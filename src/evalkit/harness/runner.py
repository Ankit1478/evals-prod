"""The RUNNER: take every case, run the agent on it, save the recording.

It does NOT grade. Grading is Box 4. Keeping running and grading separate is
what lets you re-grade an old run with a fixed grader, without re-paying to
run the agent again.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from evalkit.adapters.base import Adapter
from evalkit.schema.case import Case
from evalkit.schema.trajectory import StopReason, Trajectory


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{stamp}_{uuid.uuid4().hex[:6]}"


async def run_one(adapter: Adapter, case: Case, run_id: str,
                  trial_index: int = 0) -> Trajectory:
    """Run ONE case. Always returns a Trajectory - never raises.

    This is the rule that protects the denominator: a crash is a RECORDED
    failure, not a missing row. A missing row silently shrinks the total and
    makes your score look better than it is.
    """
    try:
        return await asyncio.wait_for(
            adapter.run(case, run_id, trial_index),
            timeout=case.timeout_s,
        )
    except asyncio.TimeoutError:
        return Trajectory(
            run_id=run_id, case_id=case.id, trial_index=trial_index,
            stop_reason=StopReason.TIMEOUT,
            error=f"exceeded timeout_s={case.timeout_s}",
        )
    except Exception as e:
        return Trajectory(
            run_id=run_id, case_id=case.id, trial_index=trial_index,
            stop_reason=StopReason.ERROR,
            error=f"{type(e).__name__}: {e}",
        )


async def run_suite(adapter: Adapter, cases: list[Case], runs_dir: Path,
                    trials: int = 1) -> tuple[str, list[Trajectory]]:
    """Run every case `trials` times. Write everything to runs/<run_id>/."""
    run_id = new_run_id()
    out = runs_dir / run_id
    (out / "trajectories").mkdir(parents=True, exist_ok=True)

    # The manifest is written BEFORE anything runs and never changed after.
    # It is what makes a run reproducible six months later.
    manifest = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "suite": cases[0].suite if cases else "unknown",
        "adapter": adapter.describe(),
        "case_count": len(cases),
        "trials_per_case": trials,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    trajectories: list[Trajectory] = []
    for case in cases:
        for trial in range(trials):
            traj = await run_one(adapter, case, run_id, trial)
            trajectories.append(traj)
            path = out / "trajectories" / f"{case.id}__trial{trial}.json"
            path.write_text(traj.model_dump_json(indent=2))

    return run_id, trajectories
