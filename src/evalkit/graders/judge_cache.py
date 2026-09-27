"""Reuse a judge's verdict when it would be asked the exact same thing again.

The echo agent gives the same reply on every run, and a real agent often
repeats itself across trials - paying a judge each time for an identical
question buys nothing. The cache key is everything the judge sees and
everything that configures it (rubric text, model(s), question, tool
facts, reference, reply), so change any of it and the judge is asked
fresh.

What is NOT cached: an abstain caused by an error, missing credentials or a
refusal - those are transient, and caching them would freeze a failure. A
split verdict (judges disagreed, human review) IS cached: it is a real
answer, and asking again would just pay to disagree again.

EVAL_JUDGE_CACHE=0 in .env turns it off. `ef judge-stability` never uses
it - that command exists to measure how much the judge varies.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from evalkit.schema.case import Case
from evalkit.schema.score import Score
from evalkit.schema.trajectory import Trajectory

DEFAULT_PATH = Path(".evalcache") / "judge_verdicts.jsonl"


def cache_enabled() -> bool:
    return os.environ.get("EVAL_JUDGE_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _cacheable(s: Score) -> bool:
    return not s.abstained or s.evidence.get("requires_human_review") is True


class CachedJudge:
    """Wraps any judge that has `cache_key(case, traj) -> dict | None`."""

    def __init__(self, judge: Any, path: Path = DEFAULT_PATH):
        self._judge = judge
        self.name, self.version, self.severity = judge.name, judge.version, judge.severity
        self._path = Path(path)
        self._hits = 0
        self._store: dict[str, dict] = {}
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    self._store[row["key"]] = row["score"]
                except (ValueError, KeyError):
                    continue          # a torn line from a killed run: skip it, keep the rest

    @property
    def hits(self) -> int:
        return self._hits

    def _key(self, case: Case, traj: Trajectory) -> str | None:
        inputs = self._judge.cache_key(case, traj)
        if inputs is None:
            return None
        blob = json.dumps({"judge": self.name, "version": self.version, **inputs},
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    async def grade(self, case: Case, traj: Trajectory) -> Score:
        key = self._key(case, traj)
        if key is not None and key in self._store:
            self._hits += 1
            cached = Score.model_validate(self._store[key])
            return cached.model_copy(update={"evidence": {**cached.evidence, "cached": True}})

        s = await self._judge.grade(case, traj)
        if key is not None and _cacheable(s):
            self._store[key] = s.model_dump(mode="json")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as f:
                f.write(json.dumps({"key": key, "score": self._store[key]}) + "\n")
        return s
