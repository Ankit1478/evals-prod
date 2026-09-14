"""Graders about TOOLS: did the agent pick the right ones, with the right
arguments, the right number of times."""

from __future__ import annotations

import re

from evalkit.graders.base import score
from evalkit.schema.case import Case
from evalkit.schema.score import Score, Severity
from evalkit.schema.trajectory import ToolStatus, Trajectory


class ToolSelection:
    """Right tool? And - just as important - no FORBIDDEN tool?

    CRITICAL, and it runs first: correct arguments cannot rescue the wrong
    tool. Cancelling the wrong order politely is still cancelling the wrong
    order.
    """
    name = "tools.selection"
    version = "1"
    severity = Severity.CRITICAL

    def grade(self, case: Case, traj: Trajectory) -> Score:
        used = [c.name for c in traj.tool_calls]
        used_set = set(used)
        required = {c.tool for c in case.expected.required_calls}
        forbidden = set(case.expected.forbidden_tools)

        missing = sorted(required - used_set)
        illegal = sorted(forbidden & used_set)

        ev = {"used": used, "required": sorted(required),
              "forbidden": sorted(forbidden), "missing": missing,
              "forbidden_used": illegal}

        if illegal:
            return score(self, 0.0, False, ev,
                         f"used forbidden tool(s): {', '.join(illegal)}")
        if missing:
            found = len(required) - len(missing)
            v = found / len(required) if required else 0.0
            return score(self, v, False, ev,
                         f"missing required tool(s): {', '.join(missing)}")
        return score(self, 1.0, True, ev, "correct tools, none forbidden")


def _constraint_ok(args: dict, constraint: str) -> bool:
    """Check something like 'amount_inr <= 2500'.

    Parsed by hand on purpose. A grader must NEVER eval() a string - tool
    output is untrusted input.
    """
    m = re.match(r"^\s*(\w+)\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$", constraint)
    if not m:
        return True                       # unparseable constraint: ignore, do not crash
    field, op, raw = m.groups()
    if field not in args:
        return False
    try:
        left, right = float(args[field]), float(raw)
    except (TypeError, ValueError):
        return False
    return {"<=": left <= right, ">=": left >= right, "==": left == right,
            "!=": left != right, "<": left < right, ">": left > right}[op]


class ToolArguments:
    """Right tool, WRONG order id is the most expensive agent bug there is.
    That is why this is CRITICAL, not minor."""
    name = "tools.arguments"
    version = "1"
    severity = Severity.CRITICAL

    def grade(self, case: Case, traj: Trajectory) -> Score:
        problems: list[str] = []
        checked = 0

        for exp in case.expected.required_calls:
            actual = [c for c in traj.tool_calls if c.name == exp.tool]
            if not actual:
                continue                   # missing tool is ToolSelection's job

            for call in actual:
                if exp.args_equal:
                    checked += 1
                    for k, want in exp.args_equal.items():
                        got = call.arguments.get(k)
                        if got != want:
                            problems.append(
                                f"{exp.tool}.{k} = {got!r}, expected {want!r}")
                if exp.args_match:
                    checked += 1
                    for k, pattern in exp.args_match.items():
                        got = str(call.arguments.get(k, ""))
                        if not re.search(pattern, got):
                            problems.append(
                                f"{exp.tool}.{k} = {got!r} does not match /{pattern}/")
                for c in exp.args_constraints or []:
                    checked += 1
                    if not _constraint_ok(call.arguments, c):
                        problems.append(f"{exp.tool} violates constraint '{c}'")

        ev = {"checks": checked, "problems": problems,
              "calls": [{"tool": c.name, "args": c.arguments} for c in traj.tool_calls]}
        if problems:
            return score(self, 0.0, False, ev, "; ".join(problems))
        return score(self, 1.0, True, ev,
                     f"all {checked} argument check(s) correct" if checked
                     else "no argument checks for this case")


class ToolExecution:
    """Called the right NUMBER of times, and did the calls actually land?

    max_times is the one that saves money: two refunds for one request is a
    double charge, and both calls look individually correct.
    """
    name = "tools.execution"
    version = "1"
    severity = Severity.MAJOR

    def grade(self, case: Case, traj: Trajectory) -> Score:
        problems: list[str] = []
        for exp in case.expected.required_calls:
            n = sum(1 for c in traj.tool_calls if c.name == exp.tool)
            if n < exp.min_times:
                problems.append(f"{exp.tool} called {n}x, expected at least {exp.min_times}")
            if exp.max_times is not None and n > exp.max_times:
                problems.append(f"{exp.tool} called {n}x, at most {exp.max_times} allowed")

        failed = [c.name for c in traj.tool_calls if c.status is ToolStatus.FAILED]
        committed = [c.name for c in traj.tool_calls if c.status is ToolStatus.COMMITTED]
        ev = {"failed": failed, "committed": committed, "problems": problems}

        if problems:
            return score(self, 0.0, False, ev, "; ".join(problems))
        note = "call counts correct"
        if failed:
            note += f" (note: {len(failed)} tool call(s) failed upstream)"
        return score(self, 1.0, True, ev, note)
