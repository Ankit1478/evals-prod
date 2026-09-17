"""Graders about what ACTUALLY CHANGED in the data.

The strongest evidence in the system. These ignore the transcript completely -
what the agent said is irrelevant here. Only the database matters.
"""

from __future__ import annotations

from evalkit.env.base import flatten
from evalkit.graders.base import score
from evalkit.schema.case import Case
from evalkit.schema.score import Score, Severity
from evalkit.schema.trajectory import Trajectory


class StateFinal:
    """Did the RIGHT thing change?

    Compares the expected final state against what the database actually
    holds at the end. A confirmation message is not evidence; this is.
    """
    name = "state.final"
    version = "1"
    severity = Severity.CRITICAL

    def grade(self, case: Case, traj: Trajectory) -> Score:
        want = case.expected.final_state or {}
        if not want:
            return score(self, 1.0, True, {},
                         "no final-state assertion for this case")

        actual = flatten(traj.state_after)
        problems, checks = [], 0
        for path, expected_value in flatten(want).items():
            checks += 1
            got = actual.get(path)
            if got != expected_value:
                problems.append(f"{path} = {got!r}, expected {expected_value!r}")

        ev = {"checked": checks, "problems": problems}
        if problems:
            return score(self, 0.0, False, ev, "; ".join(problems))
        return score(self, 1.0, True, ev, f"all {checks} state assertion(s) correct")


class StateNoSideEffects:
    """Did anything change that should NOT have?

    This is the grader that catches 'cancelled the right order AND ALSO the
    wrong one'. Every other grader would pass that trajectory.
    """
    name = "state.no_side_effects"
    version = "1"
    severity = Severity.CRITICAL

    def grade(self, case: Case, traj: Trajectory) -> Score:
        protected = set(case.expected.forbidden_state_changes)
        # anything named in final_state is allowed to change
        allowed = {p.rpartition("/")[0] for p in
                   flatten(case.expected.final_state or {})}

        violations = [c for c in traj.state_diff
                      if c.entity in protected and c.entity not in allowed]

        ev = {"protected": sorted(protected),
              "changed": [f"{c.entity}.{c.field}: {c.before!r} -> {c.after!r}"
                          for c in traj.state_diff],
              "violations": [c.entity for c in violations]}

        if violations:
            detail = "; ".join(f"{c.entity}.{c.field} changed "
                               f"{c.before!r} -> {c.after!r}" for c in violations)
            return score(self, 0.0, False, ev, f"touched protected data: {detail}")
        return score(self, 1.0, True, ev,
                     f"{len(protected)} protected entit(ies) untouched")
