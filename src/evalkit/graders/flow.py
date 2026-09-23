"""Graders about HOW the run went: did it stay in budget, did it loop, did it
recover when a tool failed.

These are diagnostic. Only loop_detection is MAJOR, because a loop burns cost
on every trial and hides behind an eventually-correct answer.

What this file deliberately does NOT do: grade tool ORDER. Two correct orders
usually exist, and a sequence assertion fails the agent for finding the other
one. Where order genuinely matters (validate before create, create before
submit), the environment rejects the early call and state graders catch it -
the rule is expressed as a precondition, not as a sequence.
"""

from __future__ import annotations

import json
from collections import Counter

from evalkit.graders.base import score
from evalkit.schema.case import Case
from evalkit.schema.score import Score, Severity
from evalkit.schema.trajectory import StopReason, ToolStatus, Trajectory

LOOP_THRESHOLD = 3          # same tool + same arguments this many times = a loop


def _key(name: str, args: dict) -> str:
    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


def _abstain(grader, evidence: dict, explanation: str) -> Score:
    return Score(grader=grader.name, grader_version=grader.version, value=0.0,
                 passed=None, severity=grader.severity, abstained=True,
                 evidence=evidence, explanation=explanation)


class StepBudget:
    """Finished inside case.max_steps tool calls."""
    name = "flow.step_budget"
    version = "1"
    severity = Severity.MINOR

    def grade(self, case: Case, traj: Trajectory) -> Score:
        used = len(traj.tool_calls)
        ev = {"tool_calls": used, "max_steps": case.max_steps,
              "stop_reason": traj.stop_reason.value}
        if traj.stop_reason is StopReason.MAX_STEPS or used > case.max_steps:
            return score(self, 0.0, False, ev,
                         f"used {used} tool calls, budget is {case.max_steps}")
        return score(self, 1.0, True, ev, f"{used}/{case.max_steps} tool calls")


class LoopDetection:
    """The same call with the same arguments, over and over.

    Repeating a call with DIFFERENT arguments is not a loop - that is the
    agent correcting itself, and it is often exactly right.
    """
    name = "flow.loop_detection"
    version = "1"
    severity = Severity.MAJOR

    def grade(self, case: Case, traj: Trajectory) -> Score:
        counts = Counter(_key(c.name, c.arguments) for c in traj.tool_calls)
        loops = {k: n for k, n in counts.items() if n >= LOOP_THRESHOLD}
        ev = {"threshold": LOOP_THRESHOLD, "repeats": loops}
        if loops:
            names = sorted({k.split(":", 1)[0] for k in loops})
            return score(self, 0.0, False, ev,
                         f"repeated identical call(s) {LOOP_THRESHOLD}+ times: "
                         f"{', '.join(names)}")
        return score(self, 1.0, True, ev, "no identical call repeated "
                                          f"{LOOP_THRESHOLD}+ times")


class RecoveryAfterFailure:
    """When a tool failed, did the agent keep going sensibly?

    Recovered means: the run still finished on its own, with something to say,
    and the agent did not keep re-sending the exact call that failed. Whether
    it TOLD the user about the failure is output.honesty's job, not this one.
    Abstains when nothing failed - there was nothing to recover from.
    """
    name = "flow.recovery_after_failure"
    version = "1"
    severity = Severity.MINOR

    def grade(self, case: Case, traj: Trajectory) -> Score:
        failed = [c for c in traj.tool_calls if c.status is ToolStatus.FAILED]
        if not failed:
            return _abstain(self, {"failed": []}, "no tool call failed")

        problems: list[str] = []
        counts = Counter(_key(c.name, c.arguments) for c in traj.tool_calls)
        retried = sorted({c.name for c in failed if counts[_key(c.name, c.arguments)] > 2})
        if retried:
            problems.append(f"re-sent the identical failing call more than once: {', '.join(retried)}")
        if traj.stop_reason is not StopReason.COMPLETED:
            problems.append(f"run ended with {traj.stop_reason.value}")
        elif not (traj.final_output or "").strip():
            problems.append("ended with no reply")

        ev = {"failed": [c.name for c in failed], "problems": problems,
              "stop_reason": traj.stop_reason.value}
        if problems:
            return score(self, 0.0, False, ev, "; ".join(problems))
        return score(self, 1.0, True, ev,
                     f"kept going after {len(failed)} failed call(s)")
