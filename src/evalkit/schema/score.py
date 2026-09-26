"""A Score is one grader's verdict on one recording."""

from __future__ import annotations

from enum import Enum

from evalkit.schema.case import Frozen


class Severity(str, Enum):
    """How much a failure matters.

    CRITICAL  any failure blocks the release, no matter the overall score
    MAJOR     counts toward the pass threshold
    MINOR     reported only, never blocks
    """
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class Score(Frozen):
    grader: str                  # "tools.selection"
    grader_version: str
    value: float                 # 0.0 - 1.0
    passed: bool | None          # None == abstained (grader could not judge)
    severity: Severity
    abstained: bool = False
    evidence: dict = {}          # EXACTLY what was observed - this is what you read when debugging
    explanation: str = ""


class CaseResult(Frozen):
    case_id: str
    trial_index: int
    scores: list[Score]
    passed: bool
    stop_reason: str

    @property
    def review_reasons(self) -> list[str]:
        """Graders that stepped aside because a HUMAN must decide.

        An ordinary abstain ("this check does not apply") is not one of
        these. A judge panel that disagreed is: it abstains so the models'
        split is not averaged into false confidence - and decide() skips
        abstains, so without this the case would quietly count as a pass.
        Once a human verdict is recorded (evalkit.review), nothing is pending.
        """
        if any(s.grader == "human.review" and not s.abstained for s in self.scores):
            return []
        return [f"{s.grader}: {s.explanation}" for s in self.scores
                if s.abstained and s.evidence.get("requires_human_review") is True]

    @property
    def needs_review(self) -> bool:
        """Only a case that would otherwise pass waits on a human. One that
        already failed another check fails whatever the human decides."""
        return self.passed and bool(self.review_reasons)
