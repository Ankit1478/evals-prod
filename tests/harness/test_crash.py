"""Never lose the denominator.

A crashed trial must become a RECORDED failure. If it silently disappears,
9/9 reads as 100% when the truth is 9/10.
"""

from evalkit.env.base import Environment
from evalkit.harness.runner import run_one
from evalkit.schema.trajectory import StopReason, Trajectory
from tests.conftest import make_case


class ExplodingAdapter:
    name, version = "boom", "0"

    async def run(self, case, env, run_id, trial_index):
        raise RuntimeError("the agent died")

    def describe(self):
        return {}


class HangingAdapter:
    name, version = "hang", "0"

    async def run(self, case, env, run_id, trial_index):
        import asyncio
        await asyncio.sleep(60)

    def describe(self):
        return {}


async def test_a_crash_becomes_a_result_not_a_missing_row():
    traj = await run_one(ExplodingAdapter(), make_case({}), "r1", 0)
    assert isinstance(traj, Trajectory)
    assert traj.stop_reason is StopReason.ERROR
    assert "the agent died" in traj.error


async def test_a_timeout_becomes_a_result():
    traj = await run_one(HangingAdapter(), make_case({}, timeout_s=1), "r1", 0)
    assert traj.stop_reason is StopReason.TIMEOUT


async def test_state_is_captured_even_when_the_agent_crashes():
    """The harness snapshots the database itself, so evidence survives a
    crash. An agent must never be the one reporting its own side effects."""
    traj = await run_one(ExplodingAdapter(), make_case({}), "r1", 0)
    assert traj.state_before != {}
    assert traj.state_after != {}
