"""Step 14: detect common attempts to manipulate the LLM judge."""

import re
from enum import Enum
from typing import Dict, List, Optional, Pattern, Tuple

from pydantic import BaseModel, ConfigDict

from .contracts import EvaluationInput


class AttackCategory(str, Enum):
    """Adversarial behaviors exercised by the red-team suite."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_IMPERSONATION = "role_impersonation"
    DECISION_MANIPULATION = "decision_manipulation"
    OUTPUT_HIJACK = "output_hijack"
    PROMPT_EXTRACTION = "prompt_extraction"
    ENCODED_INSTRUCTION = "encoded_instruction"
    IDENTITY_INFLUENCE = "identity_influence"
    DISTRACTION = "distraction"


class InputLocation(str, Enum):
    """Untrusted field containing a possible manipulation attempt."""

    QUESTION = "question"
    CONTEXT = "context"
    REFERENCE_ANSWER = "reference_answer"
    CANDIDATE_A = "candidate_a"
    CANDIDATE_B = "candidate_b"


class InjectionFinding(BaseModel):
    """A non-secret signal recorded for audit and human triage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    location: InputLocation
    category: AttackCategory
    detector_name: str


# Patterns are signals, not proof. They intentionally never include or return the
# matched candidate text, which may contain private production data.
_DETECTORS: List[Tuple[AttackCategory, str, Pattern[str]]] = [
    (
        AttackCategory.INSTRUCTION_OVERRIDE,
        "ignore_instructions",
        re.compile(r"\b(ignore|disregard|override)\b.{0,60}\b(instruction|rubric|above|previous|prior)\b", re.I | re.S),
    ),
    (
        AttackCategory.ROLE_IMPERSONATION,
        "role_impersonation",
        re.compile(r"(?:<\/?(?:system|assistant|developer)>|\b(?:system|developer)\s*(?:message|:)\b)", re.I),
    ),
    (
        AttackCategory.DECISION_MANIPULATION,
        "forced_decision",
        re.compile(r"\b(give|mark|rate|set|choose|output)\b.{0,50}\b(PASS|A_WINS|B_WINS|score\s*(?:of\s*)?5|highest score)\b", re.I | re.S),
    ),
    (
        AttackCategory.OUTPUT_HIJACK,
        "output_format_hijack",
        re.compile(r"\b(return|output|respond)\b.{0,50}\b(JSON|schema|markdown|XML|case_id)\b", re.I | re.S),
    ),
    (
        AttackCategory.PROMPT_EXTRACTION,
        "prompt_extraction",
        re.compile(r"\b(reveal|repeat|print|show|expose)\b.{0,60}\b(system prompt|hidden prompt|rubric|instructions)\b", re.I | re.S),
    ),
    (
        AttackCategory.ENCODED_INSTRUCTION,
        "encoded_instruction",
        re.compile(r"\b(base64|decode this|rot13|hex-encoded)\b", re.I),
    ),
    (
        AttackCategory.IDENTITY_INFLUENCE,
        "model_identity_claim",
        re.compile(r"\b(produced|written|generated)\s+by\s+(GPT|OpenAI|Terra|Luna|Claude|Gemini)\b", re.I),
    ),
]


def detect_prompt_injection(evaluation_input: EvaluationInput) -> List[InjectionFinding]:
    """Return suspicious-pattern metadata without blocking or changing the input."""

    fields: Dict[InputLocation, Optional[str]] = {
        InputLocation.QUESTION: evaluation_input.question,
        InputLocation.CONTEXT: evaluation_input.context,
        InputLocation.REFERENCE_ANSWER: evaluation_input.reference_answer,
        InputLocation.CANDIDATE_A: evaluation_input.candidate_answer,
        InputLocation.CANDIDATE_B: evaluation_input.candidate_b,
    }
    findings = []
    seen = set()
    for location, value in fields.items():
        if not value:
            continue
        for category, detector_name, pattern in _DETECTORS:
            identity = (location, category, detector_name)
            if pattern.search(value) and identity not in seen:
                findings.append(
                    InjectionFinding(
                        location=location,
                        category=category,
                        detector_name=detector_name,
                    )
                )
                seen.add(identity)
    return findings
