"""The Store is where runs live. Callers never touch paths.

Phase 1 is plain files (jsonl.py). A database later implements the same six
methods, and nothing that calls them changes.

The one rule every Store enforces: a run is APPEND-ONLY, and once finished it
is never modified. Re-grading an old run writes a NEW run. Otherwise a grader
change can silently rewrite history, and every comparison against the old
run becomes a comparison against something that no longer exists.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evalkit.schema.case import Frozen
from evalkit.schema.score import CaseResult
from evalkit.schema.trajectory import Trajectory


class StoreError(RuntimeError):
    """Tried to overwrite, or write to a finished run."""


class RunSummary(Frozen):
    run_id: str
    suite: str
    started_at: str
    finished_at: str | None = None      # None -> still running, or crashed
    cases: int = 0
    trials: int = 1
    results: int = 0
    passed: int = 0


class Run(Frozen):
    manifest: dict
    trajectories: list[Trajectory] = []
    results: list[CaseResult] = []
    kinds: dict[str, str] = {}          # case_id -> regression | capability | adversarial
    summary: RunSummary | None = None


@runtime_checkable
class Store(Protocol):
    def start_run(self, manifest: dict) -> str: ...
    def write_trajectory(self, traj: Trajectory) -> str: ...
    def write_result(self, result: CaseResult, kind: str = "regression",
                     run_id: str | None = None) -> None: ...
    def finish_run(self, summary: RunSummary) -> None: ...
    def load_run(self, run_id: str) -> Run: ...
    def list_runs(self, suite: str | None = None, limit: int = 20) -> list[RunSummary]: ...
