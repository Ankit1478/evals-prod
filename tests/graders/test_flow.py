"""flow.* - budget, loops, recovery. Known-pass and known-fail for each."""

from evalkit.graders.flow import LoopDetection, RecoveryAfterFailure, StepBudget
from evalkit.schema.trajectory import StopReason, ToolStatus
from tests.conftest import call, make_case, make_traj


def test_step_budget_passes_inside_budget():
    s = StepBudget().grade(make_case(max_steps=5), make_traj([call("lookup_order")]))
    assert s.passed


def test_step_budget_fails_on_max_steps_stop():
    traj = make_traj([call("lookup_order")], stop_reason=StopReason.MAX_STEPS)
    assert StepBudget().grade(make_case(), traj).passed is False


def test_same_call_three_times_is_a_loop():
    calls = [call("lookup_order", {"order_id": "123"}) for _ in range(3)]
    s = LoopDetection().grade(make_case(), make_traj(calls))
    assert s.passed is False and "lookup_order" in s.explanation


def test_same_tool_different_args_is_not_a_loop():
    calls = [call("lookup_order", {"order_id": str(i)}) for i in range(5)]
    assert LoopDetection().grade(make_case(), make_traj(calls)).passed


def test_recovery_abstains_when_nothing_failed():
    s = RecoveryAfterFailure().grade(make_case(), make_traj([call("lookup_order")]))
    assert s.abstained and s.passed is None


def test_recovery_passes_when_agent_carries_on():
    calls = [call("cancel_order", {"order_id": "123"}, status=ToolStatus.FAILED),
             call("escalate", {"reason": "cancel failed"})]
    s = RecoveryAfterFailure().grade(make_case(), make_traj(calls, say="Cancel failed; escalated."))
    assert s.passed


def test_recovery_fails_when_agent_keeps_resending_the_failed_call():
    calls = [call("cancel_order", {"order_id": "123"}, status=ToolStatus.FAILED)
             for _ in range(3)]
    s = RecoveryAfterFailure().grade(make_case(), make_traj(calls))
    assert s.passed is False and "re-sent" in s.explanation


def test_recovery_fails_when_run_errors_out():
    calls = [call("cancel_order", {"order_id": "123"}, status=ToolStatus.FAILED)]
    traj = make_traj(calls, stop_reason=StopReason.ERROR)
    assert RecoveryAfterFailure().grade(make_case(), traj).passed is False
