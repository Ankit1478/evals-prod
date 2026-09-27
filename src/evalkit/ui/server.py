"""`ef ui` - a local dashboard over everything the eval records.

Read-only views of runs, cases, conversations, graders, judges, gates and
trends - plus the two places a human writes: review verdicts
(reviews.jsonl) and gold labels (gold_labels.jsonl). Both writes go through
the same functions as `ef review` / `ef label`, so the files stay identical
whichever tool wrote them.

Standard library only (http.server), bound to 127.0.0.1 by default: this
serves agent transcripts and writes to the suite, so it is not meant to be
exposed on a network without putting auth in front of it.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from evalkit.judgeops import (KAPPA_FLOOR, GoldLabel, agreement, judged_replies,
                              labels_by_reply, load_gold_label_rows, reply_of,
                              suggest_threshold, threshold_pairs)
from evalkit.review import (REVIEWS_FILE, append_review, apply_reviews, disputed,
                            load_reviews, new_review, reply_hash)
from evalkit.schema.case import Case, load_cases
from evalkit.schema.score import CaseResult
from evalkit.stats.gate import check_rubric_approval, load_results, run_gate
from evalkit.stats.summary import summarise_cases, summarise_suite

INDEX = Path(__file__).with_name("index.html")
_RUN_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
_MAX_BODY = 64 * 1024


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def status_of(r: CaseResult) -> str:
    return "REVIEW" if r.needs_review else ("PASS" if r.passed else "FAIL")


class Dashboard:
    """The data behind every page. Pure functions of what is on disk."""

    def __init__(self, runs_dir: Path, suite_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.suite_dir = Path(suite_dir)

    # ---- loading ----------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        if not _RUN_ID.match(run_id or "") or ".." in run_id:
            raise ApiError(400, "bad run id")
        d = self.runs_dir / run_id
        if not (d / "manifest.json").exists():
            raise ApiError(404, f"no run {run_id}")
        return d

    def _results(self, run_id: str) -> tuple[list[CaseResult], dict[str, str]]:
        d = self._run_dir(run_id)
        if not (d / "scores.jsonl").exists():
            return [], {}
        results, kinds = load_results(d)
        return apply_reviews(results, load_reviews(self.suite_dir / REVIEWS_FILE)), kinds

    def _cases(self) -> dict[str, Case]:
        out: dict[str, Case] = {}
        for p in sorted((self.suite_dir / "cases").glob("*.jsonl")):
            try:
                out.update({c.id: c for c in load_cases(p)})
            except ValueError:
                continue
        return out

    def _thresholds(self) -> dict:
        p = self.suite_dir / "thresholds.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _run_ids(self) -> list[str]:
        if not self.runs_dir.exists():
            return []
        return sorted((d.name for d in self.runs_dir.iterdir()
                       if (d / "manifest.json").exists()), reverse=True)

    def reviewer(self) -> str:
        try:
            return subprocess.run(["git", "config", "user.name"], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    # ---- pages ------------------------------------------------------------

    def overview(self) -> dict:
        runs = []
        thresholds = self._thresholds()
        for run_id in self._run_ids():
            d = self.runs_dir / run_id
            manifest = json.loads((d / "manifest.json").read_text())
            results, kinds = self._results(run_id)
            per_case = summarise_cases(results, kinds)
            verdict = (run_gate(results, kinds, thresholds, suite_dir=self.suite_dir)
                       if results else None)
            adapter = manifest.get("adapter", {})
            runs.append({
                "run_id": run_id,
                "suite": manifest.get("suite"),
                "started_at": manifest.get("started_at"),
                "finished": (d / "summary.json").exists(),
                "adapter": adapter.get("adapter"),
                "model": adapter.get("model"),
                "trials": manifest.get("trials_per_case", 1),
                "cases": len(per_case),
                "reliable": sum(1 for c in per_case if c.pass_hat_k == 1.0),
                "review": sum(1 for c in per_case if c.pending),
                "per_kind": {s.kind: s.pass_hat_k for s in summarise_suite(per_case)},
                "gate": ({"exit_code": verdict.exit_code, "blocked_by": verdict.blocked_by}
                         if verdict else None),
            })
        return {"suite": str(self.suite_dir), "runs": runs}

    def run(self, run_id: str) -> dict:
        d = self._run_dir(run_id)
        manifest = json.loads((d / "manifest.json").read_text())
        results, kinds = self._results(run_id)
        cases = self._cases()
        per_case = summarise_cases(results, kinds)
        verdict = (run_gate(results, kinds, self._thresholds(), suite_dir=self.suite_dir)
                   if results else None)
        rows = []
        for r in results:
            c = cases.get(r.case_id)
            rows.append({
                "case_id": r.case_id, "trial": r.trial_index,
                "kind": kinds.get(r.case_id, "regression"),
                "status": status_of(r), "stop_reason": r.stop_reason,
                "ask": c.input.messages[0].content if c else "",
                "multi_turn": bool(c and c.input.max_turns > 1),
                "judged": any(s.grader.startswith("judge.") and not s.abstained
                              for s in r.scores) or r.needs_review,
                "why": ([f"{s.grader}: {s.explanation}" for s in r.scores if s.passed is False]
                        or r.review_reasons),
            })
        tokens = [s.evidence["usage"] for r in results for s in r.scores
                  if isinstance(s.evidence.get("usage"), dict) and not s.evidence.get("cached")]
        return {
            "run_id": run_id, "manifest": manifest, "finished": (d / "summary.json").exists(),
            "gate": ({"passed": verdict.passed, "exit_code": verdict.exit_code,
                      "blocked_by": verdict.blocked_by,
                      "stages": [s.model_dump() for s in verdict.stages]} if verdict else None),
            "suites": [s.model_dump() for s in summarise_suite(per_case)],
            "cases": rows,
            "judge_tokens": {"input": sum(u["input_tokens"] for u in tokens),
                             "output": sum(u["output_tokens"] for u in tokens)},
        }

    def case(self, run_id: str, case_id: str, trial: int) -> dict:
        d = self._run_dir(run_id)
        path = d / "trajectories" / f"{case_id}__trial{trial}.json"
        if not re.match(r"^[A-Za-z0-9_-]+$", case_id) or not path.exists():
            raise ApiError(404, f"no recording for {case_id} trial {trial}")
        traj = json.loads(path.read_text())
        results, kinds = self._results(run_id)
        result = next((r for r in results
                       if r.case_id == case_id and r.trial_index == trial), None)
        c = self._cases().get(case_id)
        return {
            "run_id": run_id, "case_id": case_id, "trial": trial,
            "kind": kinds.get(case_id), "status": status_of(result) if result else None,
            "case": c.model_dump(mode="json") if c else None,
            "trajectory": traj,
            "scores": [s.model_dump(mode="json") for s in result.scores] if result else [],
        }

    def review_queue(self, run_id: str) -> dict:
        results, _ = self._results(run_id)
        cases = self._cases()
        items = []
        for r in results:
            if not r.needs_review:
                continue
            ev = disputed(r).evidence
            c = cases.get(r.case_id)
            items.append({"case_id": r.case_id, "trial": r.trial_index,
                          "ask": c.input.messages[0].content if c else "",
                          "reply": ev.get("reply"), "judges": ev.get("per_model"),
                          "why": r.review_reasons})
        return {"run_id": run_id, "items": items, "reviewer": self.reviewer()}

    def label_queue(self, run_id: str) -> dict:
        """Judged replies to label. The judges' verdicts are deliberately
        absent: a labeller who has seen them anchors on them."""
        results, _ = self._results(run_id)
        cases = self._cases()
        rows = load_gold_label_rows(self.suite_dir / "gold_labels.jsonl")
        done = labels_by_reply(rows, results)
        items = []
        for r in results:
            reply = reply_of(r)
            if reply is None:
                continue
            c = cases.get(r.case_id)
            key = (r.case_id, reply_hash(reply))
            rubric_id = c.expected.rubric_id if c else None
            items.append({"case_id": r.case_id, "trial": r.trial_index,
                          "ask": c.input.messages[0].content if c else "",
                          "kind": c.kind if c else None,
                          "notes": c.notes if c else "",
                          "reference": c.expected.reference_answer if c else None,
                          "rubric": rubric_id, "guide": self._rubric_guide(rubric_id),
                          "reply": reply, "label": done.get(key),
                          **self._what_the_judge_saw(run_id, r)})
        guide_path = self.suite_dir / "labeling_guide.json"
        return {"run_id": run_id, "items": items, "reviewer": self.reviewer(),
                "total_labels": len(rows),
                "guide": json.loads(guide_path.read_text()) if guide_path.exists() else None}

    def _rubric_guide(self, rubric_id: str | None) -> dict | None:
        """PASS/FAIL guidance for a labeller, taken from what the judge itself
        uses: the jev rubric's own criteria, or the vendored rubric's criteria
        and pass rule. Never a second, hand-written copy that could drift."""
        if not rubric_id:
            return None
        path = self.suite_dir / "rubrics" / f"{rubric_id}.json"
        if not path.exists():
            return None
        rubric = json.loads(path.read_text())
        if rubric.get("judge") == "llm_as_judge":
            from llm_judge.contracts import CRITERIA, TASK_DEFINITION
            return {
                "question": "Is this a correct, useful answer to what the customer asked?",
                "pass": ("It matches the tool facts (and the reference, if shown), answers the "
                         "question directly, covers what matters, and is clear."),
                "fail": ("It says something no tool returned (an invented date, status or "
                         "promise), gets the facts or the order wrong, misses the main point, "
                         "or is confusing."),
                "criteria": [{"name": k.value, "weight": v.weight, "meaning": v.description}
                             for k, v in CRITERIA.items()],
                "rule": (f"The judge passes a reply when its weighted score is at least "
                         f"{TASK_DEFINITION.pass_threshold} of {TASK_DEFINITION.score_max} and both "
                         f"correctness and relevance score {TASK_DEFINITION.minimum_critical_score}+."),
            }
        crit = rubric.get("criteria", {})
        return {"question": rubric.get("instructions", ""),
                "pass": crit.get("true", ""), "fail": crit.get("false", "")}

    def _what_the_judge_saw(self, run_id: str, r: CaseResult) -> dict:
        """The exchange and the tool facts - what a labeller needs to tell a
        grounded reply from an invented one. Never the judges' verdicts."""
        path = self.runs_dir / run_id / "trajectories" / f"{r.case_id}__trial{r.trial_index}.json"
        if not path.exists():
            return {"transcript": [], "tools": []}
        t = json.loads(path.read_text())
        turns = [{"role": s["type"], "content": s["content"]} for s in t.get("steps", [])
                 if s.get("type") in ("user", "assistant") and s.get("content")]
        if turns and turns[-1]["role"] == "assistant":
            turns = turns[:-1]                       # the reply being labelled
        tools = [{"name": tc["name"], "arguments": tc.get("arguments"),
                  "result": tc.get("result"), "error": tc.get("error")}
                 for tc in t.get("tool_calls", [])]
        return {"transcript": turns if len(turns) > 1 else [], "tools": tools}

    def judges(self, run_id: str) -> dict:
        results, _ = self._results(run_id)
        ok, detail = check_rubric_approval(self.suite_dir / "rubric_approval.json")
        rows = load_gold_label_rows(self.suite_dir / "gold_labels.jsonl")
        human = labels_by_reply(rows, results)
        graders = sorted({s.grader for r in results for s in r.scores
                          if s.grader.startswith("judge.") and not s.abstained})
        per_judge = []
        for g in graders:
            rep = agreement(judged_replies(results, g), human)
            pairs = threshold_pairs(results, g, human)
            fit = suggest_threshold(pairs) if pairs else None
            # A rubric addressed to the other judge is "not mine", not "could not decide".
            verdicts = [s for r in results for s in r.scores if s.grader == g
                        and not (s.abstained and ("addressed to judge" in s.explanation
                                                  or "no rubric" in s.explanation))]
            per_judge.append({
                "grader": g, "judged": sum(1 for s in verdicts if not s.abstained),
                "abstained": sum(1 for s in verdicts if s.abstained),
                "labelled": rep.cases, "agreed": rep.agreed,
                "agreement_rate": rep.agreement_rate, "kappa": rep.kappa,
                "meets_floor": rep.meets_floor, "threshold_pairs": len(pairs),
                "threshold_fit": ({"threshold": fit[0], "agreement": fit[1]} if fit else None),
            })
        return {"run_id": run_id, "rubric_approval": {"ok": ok, "detail": detail},
                "labels_total": len(rows), "labels_in_run": len(human),
                "kappa_floor": KAPPA_FLOOR, "judges": per_judge}

    # ---- writes -----------------------------------------------------------

    def _find(self, run_id: str, case_id: str, trial: int) -> CaseResult:
        results, _ = self._results(run_id)
        r = next((r for r in results if r.case_id == case_id and r.trial_index == trial), None)
        if r is None:
            raise ApiError(404, f"no result for {case_id} trial {trial}")
        return r

    @staticmethod
    def _decision(body: dict) -> str:
        decision = str(body.get("decision", "")).upper()
        if decision not in {"PASS", "FAIL"}:
            raise ApiError(400, "decision must be PASS or FAIL")
        if not str(body.get("reviewer", "")).strip():
            raise ApiError(400, "reviewer name is required")
        return decision

    def post_review(self, body: dict) -> dict:
        decision = self._decision(body)
        r = self._find(body.get("run_id", ""), body.get("case_id", ""), int(body.get("trial", 0)))
        if not r.needs_review:
            raise ApiError(409, "this case is not waiting on a review")
        rv = new_review(r, decision, str(body["reviewer"]), str(body.get("note", "")))
        append_review(self.suite_dir / REVIEWS_FILE, rv)
        return {"ok": True, "review": rv.model_dump()}

    def post_label(self, body: dict) -> dict:
        decision = self._decision(body)
        r = self._find(body.get("run_id", ""), body.get("case_id", ""), int(body.get("trial", 0)))
        reply = reply_of(r)
        if reply is None:
            raise ApiError(409, "this case has no judged reply to label")
        row = GoldLabel(case_id=r.case_id, human_decision=decision,
                        reply_sha256=reply_hash(reply), reviewer=str(body["reviewer"]).strip(),
                        note=str(body.get("note", "")),
                        labeled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with (self.suite_dir / "gold_labels.jsonl").open("a") as f:
            f.write(row.model_dump_json() + "\n")
        return {"ok": True, "label": row.model_dump()}


def make_handler(dash: Dashboard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:       # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            self._send(status, json.dumps(data, default=str).encode(), "application/json")

        def _route(self, method: str) -> None:
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if method == "GET" and not parts:
                    self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
                    return
                if parts[:1] != ["api"]:
                    raise ApiError(404, "not found")
                api = parts[1:]
                if method == "GET":
                    if api == ["overview"]:
                        data = dash.overview()
                    elif len(api) == 2 and api[0] == "runs":
                        data = dash.run(api[1])
                    elif len(api) == 5 and api[0] == "runs" and api[2] == "cases":
                        data = dash.case(api[1], api[3], int(api[4]))
                    elif api == ["review"]:
                        data = dash.review_queue(q.get("run", ""))
                    elif api == ["label"]:
                        data = dash.label_queue(q.get("run", ""))
                    elif api == ["judges"]:
                        data = dash.judges(q.get("run", ""))
                    else:
                        raise ApiError(404, "not found")
                    self._json(200, data)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                if length > _MAX_BODY:
                    raise ApiError(413, "body too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                if api == ["review"]:
                    self._json(200, dash.post_review(body))
                elif api == ["label"]:
                    self._json(200, dash.post_label(body))
                else:
                    raise ApiError(404, "not found")
            except ApiError as e:
                self._json(e.status, {"error": str(e)})
            except (ValueError, KeyError) as e:
                self._json(400, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

    return Handler


def serve(runs_dir: Path, suite_dir: Path, host: str = "127.0.0.1",
          port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(Dashboard(runs_dir, suite_dir)))
