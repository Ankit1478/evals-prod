"""JsonlStore - runs as folders of files, under runs/.

    runs/<run_id>/
        manifest.json                      written once, before anything runs
        trajectories/<case>__trial<N>.json one file each, never overwritten
        scores.jsonl                       one row per graded trial, appended
        summary.json                       written once; its presence = finished

The layout is the one `ef gate`, `ef diff` and `ef report` already read, so
old runs load unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from evalkit.schema.score import CaseResult
from evalkit.schema.trajectory import Trajectory
from evalkit.store.base import Run, RunSummary, StoreError


class JsonlStore:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)
        # A CaseResult does not carry its run id, so write_result writes to
        # the run this store started unless told otherwise.
        self._current: str | None = None

    def _dir(self, run_id: str) -> Path:
        d = self.root / run_id
        if not (d / "manifest.json").exists():
            raise StoreError(f"no such run: {run_id}")
        return d

    def _open_dir(self, run_id: str) -> Path:
        d = self._dir(run_id)
        if (d / "summary.json").exists():
            raise StoreError(f"run {run_id} is finished - write a new run instead")
        return d

    # ---- writing -------------------------------------------------------------

    def start_run(self, manifest: dict) -> str:
        run_id = manifest.get("run_id")
        if not run_id:
            raise StoreError("manifest needs a run_id")
        d = self.root / run_id
        if d.exists():
            raise StoreError(f"run {run_id} already exists - run ids are never reused")
        (d / "trajectories").mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        self._current = run_id
        return run_id

    def write_trajectory(self, traj: Trajectory) -> str:
        d = self._open_dir(traj.run_id)
        path = d / "trajectories" / f"{traj.case_id}__trial{traj.trial_index}.json"
        if path.exists():
            raise StoreError(f"{path.name} already written for run {traj.run_id}")
        path.write_text(traj.model_dump_json(indent=2))
        return str(path)

    def write_result(self, result: CaseResult, kind: str = "regression",
                     run_id: str | None = None) -> None:
        run_id = run_id or self._current
        if run_id is None:
            raise StoreError("no open run - call start_run first, or pass run_id")
        d = self._open_dir(run_id)
        row = result.model_dump(mode="json")
        row["kind"] = kind
        with (d / "scores.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    def finish_run(self, summary: RunSummary) -> None:
        d = self._open_dir(summary.run_id)
        (d / "summary.json").write_text(summary.model_dump_json(indent=2))

    # ---- reading -------------------------------------------------------------

    def load_run(self, run_id: str) -> Run:
        d = self._dir(run_id)
        manifest = json.loads((d / "manifest.json").read_text())
        trajs = [Trajectory.model_validate_json(p.read_text())
                 for p in sorted((d / "trajectories").glob("*.json"))]
        results: list[CaseResult] = []
        kinds: dict[str, str] = {}
        scores = d / "scores.jsonl"
        if scores.exists():
            for line in scores.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                kinds[row["case_id"]] = row.pop("kind", "regression")
                results.append(CaseResult.model_validate(row))
        summary_path = d / "summary.json"
        summary = (RunSummary.model_validate_json(summary_path.read_text())
                   if summary_path.exists() else None)
        return Run(manifest=manifest, trajectories=trajs, results=results,
                   kinds=kinds, summary=summary)

    def list_runs(self, suite: str | None = None, limit: int = 20) -> list[RunSummary]:
        """Newest first. Unfinished runs are listed too, with finished_at=None."""
        if not self.root.exists():
            return []
        out: list[RunSummary] = []
        for d in sorted((p for p in self.root.iterdir() if p.is_dir()), reverse=True):
            summary_path, manifest_path = d / "summary.json", d / "manifest.json"
            if summary_path.exists():
                s = RunSummary.model_validate_json(summary_path.read_text())
            elif manifest_path.exists():
                # Unfinished - or written before the store existed. Count
                # whatever results it has, so old runs still show a score.
                m = json.loads(manifest_path.read_text())
                scores = d / "scores.jsonl"
                rows = ([json.loads(x) for x in scores.read_text().splitlines() if x.strip()]
                        if scores.exists() else [])
                s = RunSummary(run_id=m.get("run_id", d.name),
                               suite=m.get("suite", "unknown"),
                               started_at=m.get("started_at", ""),
                               cases=m.get("case_count", 0),
                               trials=m.get("trials_per_case", 1),
                               results=len(rows),
                               passed=sum(1 for r in rows if r.get("passed")))
            else:
                continue
            if suite and s.suite != suite:
                continue
            out.append(s)
            if len(out) >= limit:
                break
        return out
