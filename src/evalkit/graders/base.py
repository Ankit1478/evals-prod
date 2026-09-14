"""A Grader looks at (case, recording) and returns one Score.

Graders are PURE: no network, no database, no clock, no randomness. Same
inputs -> same score, forever. That is what makes them testable with fixtures,
and a grader you cannot test is a grader you cannot trust.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evalkit.schema.case import Case
from evalkit.schema.score import Score, Severity
from evalkit.schema.trajectory import Trajectory


@runtime_checkable
class Grader(Protocol):
    name: str
    version: str
    severity: Severity

    def grade(self, case: Case, traj: Trajectory) -> Score: ...


def score(grader: "Grader", value: float, passed: bool,
          evidence: dict, explanation: str) -> Score:
    """Small helper so every grader builds a Score the same way."""
    return Score(
        grader=grader.name,
        grader_version=grader.version,
        value=value,
        passed=passed,
        severity=grader.severity,
        evidence=evidence,
        explanation=explanation,
    )
