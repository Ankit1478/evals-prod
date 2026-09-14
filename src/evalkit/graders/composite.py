"""Combine many Scores into ONE pass/fail for the case.

The rule is severity-ordered, not an average. Averaging lets four polite
MINOR passes outvote 'cancelled the wrong customer's order'.
"""

from __future__ import annotations

from evalkit.graders.base import Grader
from evalkit.graders.output import OutputBehavior, OutputHonesty
from evalkit.graders.tools import ToolArguments, ToolExecution, ToolSelection
from evalkit.schema.case import Case
from evalkit.schema.score import CaseResult, Score, Severity
from evalkit.schema.trajectory import Trajectory

DEFAULT_GRADERS: list[Grader] = [
    ToolSelection(),
    ToolArguments(),
    ToolExecution(),
    OutputHonesty(),
    OutputBehavior(),
]

MAJOR_THRESHOLD = 1.0      # every MAJOR check must pass, for now


def grade_all(case: Case, traj: Trajectory,
              graders: list[Grader] | None = None) -> CaseResult:
    graders = graders or DEFAULT_GRADERS
    scores: list[Score] = [g.grade(case, traj) for g in graders]
    return CaseResult(
        case_id=case.id,
        trial_index=traj.trial_index,
        scores=scores,
        passed=decide(scores),
        stop_reason=traj.stop_reason.value,
    )


def decide(scores: list[Score]) -> bool:
    """1. any CRITICAL failure  -> fail, full stop
       2. else MAJOR average    -> must clear the threshold
       3. MINOR                 -> reported, never gates
    """
    graded = [s for s in scores if not s.abstained]

    if any(s.severity is Severity.CRITICAL and s.passed is False for s in graded):
        return False

    majors = [s for s in graded if s.severity is Severity.MAJOR]
    if majors:
        avg = sum(s.value for s in majors) / len(majors)
        return avg >= MAJOR_THRESHOLD
    return True
