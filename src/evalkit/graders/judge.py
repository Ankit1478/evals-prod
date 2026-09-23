"""judge.rubric - wraps a hosted judge model (TypeSafe `jev`) behind the
same Score shape every other grader uses.

This is the ONE async, networked, non-deterministic grader in the system,
and it exists only for what a rule cannot check (tone, quality) - never for
anything a regex or a state diff can already decide with certainty. It
ABSTAINS rather than guesses whenever it cannot get a clean answer: an
abstained score is excluded from the pass/fail average entirely (see
graders/composite.py's `decide()`), while a confidently wrong guess would
not be - so silence here is always the safer failure.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Awaitable, Callable

from evalkit.graders.base import score
from evalkit.graders.composite import decide
from evalkit.schema.case import Case
from evalkit.schema.score import CaseResult, Score, Severity
from evalkit.schema.trajectory import Trajectory

Transport = Callable[[str, dict, dict], Awaitable[dict]]

JUDGE_KEY = "jev"      # rubrics name the judge they were written for


async def _typesafe_transport(url: str, headers: dict, body: dict) -> dict:
    """The real network call, isolated behind one function so tests can
    swap it out - a grader must be testable without paying for an API call
    every time the suite runs."""

    def _post() -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())

    return await asyncio.to_thread(_post)


class JevJudge:
    """One question per case: does the reply meet the named rubric?

    Only runs for cases that set `expected.rubric_id` - every case without
    one is untouched, so adding this grader cannot change any existing
    score.
    """
    name = "judge.rubric"
    version = "1"
    severity = Severity.MAJOR

    def __init__(self, rubric_dir: Path, transport: Transport = _typesafe_transport):
        self.rubric_dir = Path(rubric_dir)
        self._transport = transport

    def _abstain(self, evidence: dict, explanation: str) -> Score:
        return Score(grader=self.name, grader_version=self.version, value=0.0,
                    passed=None, severity=self.severity, abstained=True,
                    evidence=evidence, explanation=explanation)

    async def grade(self, case: Case, traj: Trajectory) -> Score:
        rubric_id = case.expected.rubric_id
        if rubric_id is None:
            return self._abstain({}, "no rubric on this case - judge does not apply")

        rubric_path = self.rubric_dir / f"{rubric_id}.json"
        if not rubric_path.exists():
            return self._abstain({"rubric_id": rubric_id},
                                 f"no rubric file at {rubric_path}")
        try:
            rubric: dict[str, Any] = json.loads(rubric_path.read_text())
        except json.JSONDecodeError as e:
            return self._abstain({"rubric_id": rubric_id}, f"rubric file is not valid JSON: {e}")

        # More than one judge lives here now, so a rubric names the one it
        # was written for. Anything addressed elsewhere is not ours to score.
        addressed_to = rubric.get("judge", JUDGE_KEY)
        if addressed_to != JUDGE_KEY:
            return self._abstain(
                {"rubric_id": rubric_id},
                f"rubric '{rubric_id}' is addressed to judge "
                f"'{addressed_to}', not {JUDGE_KEY}")

        token = os.environ.get("JEV")
        if not token:
            return self._abstain(
                {"rubric_id": rubric_id},
                "JEV not set in .env - add it to enable the judge")

        question = "meets_rubric"
        body = {
            "state": traj.final_output or "",
            "model": os.environ.get("JEV_MODEL") or "jev-latest",
            "questions": {
                question: {
                    "type": "noul",
                    "instructions": rubric["instructions"],
                    "criteria": rubric["criteria"],
                }
            },
        }
        url = "https://api.typesafe.ai/v1/systemone"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            resp = await self._transport(url, headers, body)
            noul = float(resp["answers"][question]["noul"])
        except Exception as e:
            # Network error, bad status, unexpected shape - all the same
            # outcome here: abstain, never crash the run over a judge call.
            return self._abstain({"error": f"{type(e).__name__}: {e}"},
                                 "judge call failed or returned an "
                                 "unexpected shape - abstaining, not guessing")

        passed = noul >= 0.5
        return score(self, noul, passed,
                    {"rubric_id": rubric_id, "raw_noul": noul, "reply": traj.final_output},
                    f"jev scored {noul:.2f} against rubric '{rubric_id}'"
                    + ("" if passed else " - below the 0.5 bar"))


async def augment_with_judge(result: CaseResult, case: Case, traj: Trajectory,
                             judge: Any) -> CaseResult:
    """Fold a judge's Score into an already-graded CaseResult.

    `judge` is anything with an async `grade(case, traj) -> Score` - the jev
    judge here, or the wrapped LLM_AS_JUDGE subsystem. Cases without a rubric
    are returned completely untouched, so adding a judge can never change a
    score for a case that never asked for one.
    """
    if case.expected.rubric_id is None:
        return result
    judge_score = await judge.grade(case, traj)
    scores = [*result.scores, judge_score]
    return result.model_copy(update={"scores": scores, "passed": decide(scores)})
