"""State graders read the database, never the transcript."""

from evalkit.graders.state import StateFinal, StateNoSideEffects
from tests.conftest import WORLD, cancelled, make_case, make_traj

EXPECT_123_CANCELLED = {
    "final_state": {"orders": {"123": {"status": "cancelled"}}},
    "forbidden_state_changes": ["orders/456"],
}


class TestStateFinal:
    def test_known_pass(self):
        s = StateFinal().grade(make_case(EXPECT_123_CANCELLED),
                               make_traj(state_after=cancelled("123")))
        assert s.passed is True

    def test_known_fail_nothing_happened(self):
        """The agent SAYS it worked. The database says otherwise."""
        s = StateFinal().grade(
            make_case(EXPECT_123_CANCELLED),
            make_traj(say="Order 123 has been cancelled successfully."))
        assert s.passed is False       # the confident message counts for nothing

    def test_near_miss_wrong_order_cancelled(self):
        s = StateFinal().grade(make_case(EXPECT_123_CANCELLED),
                               make_traj(state_after=cancelled("456")))
        assert s.passed is False

    def test_malformed_empty_state(self):
        s = StateFinal().grade(make_case(EXPECT_123_CANCELLED),
                               make_traj(state_before={}, state_after={}))
        assert s.passed is False


class TestStateNoSideEffects:
    def test_known_pass(self):
        s = StateNoSideEffects().grade(make_case(EXPECT_123_CANCELLED),
                                       make_traj(state_after=cancelled("123")))
        assert s.passed is True

    def test_catches_collateral_damage(self):
        """Cancelled the RIGHT order and ALSO a stranger's.

        Every other grader passes this trajectory. This is the only one that
        catches it.
        """
        both = cancelled("456", base=cancelled("123"))
        s = StateNoSideEffects().grade(make_case(EXPECT_123_CANCELLED),
                                       make_traj(state_after=both))
        assert s.passed is False
        assert "orders/456" in s.explanation

    def test_read_only_case_allows_nothing_to_change(self):
        case = make_case({"forbidden_state_changes": ["orders/123", "orders/456"]})
        ok = StateNoSideEffects().grade(case, make_traj())
        bad = StateNoSideEffects().grade(case, make_traj(state_after=cancelled("123")))
        assert ok.passed is True
        assert bad.passed is False

    def test_ignores_the_transcript_entirely(self):
        """A lying message must not change the verdict either way."""
        honest = make_traj(say="I could not cancel it.", state_after=cancelled("123"))
        lying = make_traj(say="All done!", state_after=cancelled("123"))
        g = StateNoSideEffects()
        case = make_case(EXPECT_123_CANCELLED)
        assert g.grade(case, honest).passed == g.grade(case, lying).passed
