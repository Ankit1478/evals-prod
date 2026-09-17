"""An Adapter connects the harness to ONE kind of agent.

HTTP agent, Python function, CLI tool, MCP server - each gets an adapter.
The runner never knows which one it is talking to. That is the whole point:
swapping the agent must never mean changing the harness.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evalkit.env.base import Environment
from evalkit.schema.case import Case
from evalkit.schema.trajectory import Trajectory


@runtime_checkable
class Adapter(Protocol):
    name: str
    version: str

    async def run(self, case: Case, env: Environment, run_id: str,
                  trial_index: int) -> Trajectory:
        """Run the agent on one case and return the recording.

        The adapter MUST NOT read case.expected. That is the answer key.
        """
        ...

    def describe(self) -> dict:
        """Goes into the run manifest, so a run is reproducible."""
        ...
