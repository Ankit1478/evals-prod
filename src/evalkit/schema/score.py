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
