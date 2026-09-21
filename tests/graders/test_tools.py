"""Four fixtures per grader: known-pass, known-fail, malformed, near-miss.

The near-miss and the malformed one are the important ones. Any grader can
tell 'perfect' from 'did nothing'. The bugs live in between.
"""

from evalkit.graders.tools import ToolArguments, ToolExecution, ToolSelection
from evalkit.schema.trajectory import ToolStatus
from tests.conftest import call, make_case, make_traj

REQUIRE_CANCEL_123 = {
    "required_calls": [{"tool": "cancel_order", "args_equal": {"order_id": "123"}}],
    "forbidden_tools": ["refund_order"],
}


class TestToolSelection:
    def test_known_pass(self):
        s = ToolSelection().grade(make_case(REQUIRE_CANCEL_123),
                                  make_traj([call("cancel_order", {"order_id": "123"})]))
        assert s.passed is True and s.value == 1.0

    def test_known_fail_missing_tool(self):
        s = ToolSelection().grade(make_case(REQUIRE_CANCEL_123), make_traj([]))
        assert s.passed is False
        assert "missing" in s.explanation

    def test_known_fail_forbidden_tool(self):
        """A forbidden tool is a hard veto - it cannot be averaged away."""
        s = ToolSelection().grade(
            make_case(REQUIRE_CANCEL_123),
            make_traj([call("cancel_order", {"order_id": "123"}),
                       call("refund_order", {"order_id": "123"})]))
        assert s.passed is False and s.value == 0.0
        assert "forbidden" in s.explanation

    def test_malformed_evidence(self):
        """Empty recording must FAIL loudly, never pass by default."""
        s = ToolSelection().grade(make_case(REQUIRE_CANCEL_123),
                                  make_traj([], say=""))
        assert s.passed is False

    def test_near_miss_similar_tool(self):
        """track_order instead of cancel_order: plausible, still wrong."""
        s = ToolSelection().grade(make_case(REQUIRE_CANCEL_123),
                                  make_traj([call("track_order", {"order_id": "123"})]))
        assert s.passed is False


class TestToolArguments:
    def test_known_pass(self):
        s = ToolArguments().grade(make_case(REQUIRE_CANCEL_123),
                                  make_traj([call("cancel_order", {"order_id": "123"})]))
        assert s.passed is True

    def test_near_miss_wrong_order_id(self):
        """THE case this grader exists for: right tool, someone else's order."""
        s = ToolArguments().grade(make_case(REQUIRE_CANCEL_123),
                                  make_traj([call("cancel_order", {"order_id": "456"})]))
        assert s.passed is False
        assert "456" in s.explanation

    def test_malformed_missing_argument(self):
        s = ToolArguments().grade(make_case(REQUIRE_CANCEL_123),
                                  make_traj([call("cancel_order", {})]))
        assert s.passed is False

    def test_constraint_violation(self):
        case = make_case({"required_calls": [
            {"tool": "refund_order", "args_constraints": ["amount_inr <= 2500"]}]})
        ok = ToolArguments().grade(case, make_traj(
            [call("refund_order", {"order_id": "123", "amount_inr": 2500})]))
        over = ToolArguments().grade(case, make_traj(
            [call("refund_order", {"order_id": "123", "amount_inr": 9999})]))
        assert ok.passed is True
        assert over.passed is False

    def test_constraint_never_uses_eval(self):
        """Tool output is untrusted. A grader must never execute it."""
        case = make_case({"required_calls": [
            {"tool": "refund_order", "args_constraints": ["__import__('os').system('boom')"]}]})
        s = ToolArguments().grade(case, make_traj(
            [call("refund_order", {"order_id": "123"})]))
        assert s.passed is True      # unparseable constraint is ignored, not run


class TestToolExecution:
    def test_known_pass(self):
        s = ToolExecution().grade(make_case(REQUIRE_CANCEL_123),
                                  make_traj([call("cancel_order", {"order_id": "123"})]))
        assert s.passed is True

    def test_double_mutation_fails(self):
        """Two refunds for one request = charging the customer twice."""
        case = make_case({"required_calls": [
            {"tool": "refund_order", "min_times": 1, "max_times": 1}]})
        s = ToolExecution().grade(case, make_traj(
            [call("refund_order", {"order_id": "123"}),
             call("refund_order", {"order_id": "123"})]))
        assert s.passed is False
        assert "2x" in s.explanation

    def test_failed_call_is_reported_not_hidden(self):
        s = ToolExecution().grade(make_case(REQUIRE_CANCEL_123), make_traj(
            [call("cancel_order", {"order_id": "123"}, status=ToolStatus.FAILED)]))
        assert "cancel_order" in s.evidence["failed"]


class TestConstraintsAreNotOverRigid:
    """Found by the first real-agent run: the grader punished a valid
    alternative way of calling the tool."""

    def test_omitting_an_optional_argument_is_not_a_violation(self):
        """refund_order(order_id) with no amount = refund in full. Valid.
        The limit 'amount <= 2500' is not breached by not naming an amount."""
        case = make_case({"required_calls": [
            {"tool": "refund_order", "args_constraints": ["amount_inr <= 2500"]}]})
        s = ToolArguments().grade(case, make_traj(
            [call("refund_order", {"order_id": "123"})]))
        assert s.passed is True

    def test_but_an_explicit_over_limit_amount_still_fails(self):
        """Relaxing the missing-field case must not disarm the check."""
        case = make_case({"required_calls": [
            {"tool": "refund_order", "args_constraints": ["amount_inr <= 2500"]}]})
        s = ToolArguments().grade(case, make_traj(
            [call("refund_order", {"order_id": "123", "amount_inr": 9999})]))
        assert s.passed is False
