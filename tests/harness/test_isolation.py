"""Trial N+1 must not see trial N's damage."""

import json

from evalkit.adapters.echo import EchoAdapter
from evalkit.harness.runner import run_one
from tests.conftest import make_case


async def test_each_trial_gets_a_clean_database(tmp_path):
    case = make_case({})
    script = tmp_path / "s.json"
    script.write_text(json.dumps({case.id: {
        "actions": [{"tool": "cancel_order", "args": {"order_id": "123"}}],
        "say": "done"}}))
    adapter = EchoAdapter(script)

    for trial in range(3):
        traj = await run_one(adapter, case, "r1", trial)
        # every trial must START with 123 active, not inherit the last cancel
        assert traj.state_before["orders"]["123"]["status"] == "active"
        assert traj.state_after["orders"]["123"]["status"] == "cancelled"
