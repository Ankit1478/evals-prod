"""Builders so each test says only what it is actually testing.

Everything else gets a sane default. A test that has to spell out 20 fields
is a test nobody reads.
"""

from __future__ import annotations

from datetime import date

import pytest

from evalkit.schema.case import Case, CaseInput, Expected, Message
from evalkit.schema.trajectory import (Step, StepType, StopReason, ToolCall,
                                       ToolStatus, Trajectory)

WORLD = {
    "orders": {
        "123": {"customer_id": "u1", "status": "active", "amount_inr": 2500},
        "456": {"customer_id": "u9", "status": "active", "amount_inr": 8000},
    },
    "session": {"customer_id": "u1"},
}


def make_case(expected: dict | None = None, ask: str = "Cancel order 123.",
              **kw) -> Case:
    return Case(
        id=kw.pop("id", "t_case_01"),
        suite="test",
        kind=kw.pop("kind", "regression"),
        input=CaseInput(messages=[Message(role="user", content=ask)]),
        initial_state=kw.pop("initial_state", WORLD),
        expected=Expected(**(expected or {})),
        added_at=date(2026, 1, 1),
        review_status="approved",
        **kw,
    )


def call(name: str, args: dict | None = None,
         status: ToolStatus = ToolStatus.COMMITTED, **kw) -> ToolCall:
    return ToolCall(id=f"c-{name}", name=name, arguments=args or {},
                    status=status, **kw)


def make_traj(calls: list[ToolCall] | None = None, say: str = "Done.",
              state_before: dict | None = None, state_after: dict | None = None,
              **kw) -> Trajectory:
    from evalkit.env.base import compute_diff
    calls = calls or []
    before = WORLD if state_before is None else state_before
    after = before if state_after is None else state_after
    steps = [Step(index=i, type=StepType.TOOL_CALL, tool_call=c)
             for i, c in enumerate(calls)]
    steps.append(Step(index=len(steps), type=StepType.ASSISTANT, content=say))
    return Trajectory(
        run_id="t_run", case_id=kw.pop("case_id", "t_case_01"),
        steps=steps, tool_calls=calls, final_output=say,
        state_before=before, state_after=after,
        state_diff=compute_diff(before, after),
        stop_reason=kw.pop("stop_reason", StopReason.COMPLETED), **kw,
    )


def cancelled(order_id: str, base: dict | None = None) -> dict:
    """A copy of the world where one order is cancelled."""
    import copy
    w = copy.deepcopy(base or WORLD)
    w["orders"][order_id]["status"] = "cancelled"
    return w


@pytest.fixture
def world() -> dict:
    return WORLD
