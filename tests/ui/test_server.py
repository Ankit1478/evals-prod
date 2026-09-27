"""The dashboard must show what is on disk, write labels/reviews in the same
format as the CLI, hide judge verdicts while labelling, and refuse paths
outside runs/."""

import json
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from evalkit.judgeops import load_gold_label_rows
from evalkit.review import load_reviews, reply_hash
from evalkit.schema.score import CaseResult, Score, Severity
from evalkit.ui.server import serve

from tests.conftest import make_traj

RUN = "2026-01-01T00-00-00_abc123"
LABEL_CASE = "order_cancel_empathy_11"      # a real suite case id, so the case page resolves
SPLIT_CASE = "order_lookup_quality_13"


def _result(case_id, scores, passed=True):
    return CaseResult(case_id=case_id, trial_index=0, stop_reason="completed",
                      passed=passed, scores=scores)


@pytest.fixture
def site(tmp_path):
    suite = tmp_path / "suite"
    shutil.copytree(Path("suites/support-agent"), suite)
    for f in ("gold_labels.jsonl", "reviews.jsonl"):
        (suite / f).unlink(missing_ok=True)
    run = tmp_path / "runs" / RUN
    (run / "trajectories").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "run_id": RUN, "suite": "support-agent", "started_at": "2026-01-01T00:00:00+00:00",
        "adapter": {"adapter": "inprocess", "model": "openai:gpt-x"}, "trials_per_case": 1}))
    judged = _result(LABEL_CASE, [Score(grader="judge.rubric", grader_version="1", value=0.97,
                                        passed=True, severity=Severity.MAJOR,
                                        evidence={"reply": "So sorry - I can't.", "threshold": 0.5},
                                        explanation="jev scored 0.97")])
    split = _result(SPLIT_CASE, [Score(grader="judge.answer_quality", grader_version="1", value=0,
                                       passed=None, severity=Severity.MAJOR, abstained=True,
                                       evidence={"requires_human_review": True,
                                                 "reply": "It arrives Friday.",
                                                 "per_model": {"a": "PASS", "b": "FAIL"}},
                                       explanation="a and b disagreed")])
    rows = [{**judged.model_dump(mode="json"), "kind": "capability"},
            {**split.model_dump(mode="json"), "kind": "capability"}]
    (run / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    traj = make_traj(say="So sorry - I can't.", case_id=LABEL_CASE)
    (run / "trajectories" / f"{LABEL_CASE}__trial0.json").write_text(traj.model_dump_json())

    server = serve(tmp_path / "runs", suite, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path) as r:
            return json.loads(r.read())

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    yield {"get": get, "post": post, "suite": suite, "base": base}
    server.shutdown()
    server.server_close()


def test_index_page_is_served(site):
    with urllib.request.urlopen(site["base"] + "/") as r:
        assert "Evals Dashboard" in r.read().decode()


def test_overview_lists_the_run_with_its_gate(site):
    d = site["get"]("/api/overview")
    [run] = d["runs"]
    assert run["run_id"] == RUN and run["cases"] == 2 and run["review"] == 1
    assert run["gate"]["exit_code"] == 4                    # the split case blocks


def test_run_page_shows_status_per_case(site):
    d = site["get"](f"/api/runs/{RUN}")
    status = {c["case_id"]: c["status"] for c in d["cases"]}
    assert status == {LABEL_CASE: "PASS", SPLIT_CASE: "REVIEW"}
    assert d["gate"]["blocked_by"] == "1b REVIEW"


def test_case_page_has_conversation_scores_and_answer_key(site):
    d = site["get"](f"/api/runs/{RUN}/cases/{LABEL_CASE}/0")
    assert d["trajectory"]["final_output"] == "So sorry - I can't."
    assert d["scores"][0]["grader"] == "judge.rubric"
    assert d["case"]["expected"]["rubric_id"] == "empathetic_refusal"


def test_label_queue_hides_the_judges_verdict(site):
    d = site["get"](f"/api/label?run={RUN}")
    blob = json.dumps(d)
    assert "So sorry" in blob
    assert "0.97" not in blob and "jev scored" not in blob and "per_model" not in blob


def test_a_label_from_the_ui_is_pinned_like_ef_label(site):
    code, _ = site["post"]("/api/label", {"run_id": RUN, "case_id": LABEL_CASE, "trial": 0,
                                          "decision": "pass", "note": "warm", "reviewer": "Ankit"})
    assert code == 200
    [row] = load_gold_label_rows(site["suite"] / "gold_labels.jsonl")
    assert row.reply_sha256 == reply_hash("So sorry - I can't.") and row.reviewer == "Ankit"
    assert site["get"](f"/api/label?run={RUN}")["items"][0]["label"] == "PASS"


def test_a_review_from_the_ui_settles_the_case(site):
    code, _ = site["post"]("/api/review", {"run_id": RUN, "case_id": SPLIT_CASE, "trial": 0,
                                           "decision": "FAIL", "note": "invented date",
                                           "reviewer": "Ankit"})
    assert code == 200
    assert len(load_reviews(site["suite"] / "reviews.jsonl")) == 1
    status = {c["case_id"]: c["status"] for c in site["get"](f"/api/runs/{RUN}")["cases"]}
    assert status[SPLIT_CASE] == "FAIL"


def test_writes_need_a_name_and_a_real_decision(site):
    body = {"run_id": RUN, "case_id": LABEL_CASE, "trial": 0}
    assert site["post"]("/api/label", {**body, "decision": "PASS", "reviewer": " "})[0] == 400
    assert site["post"]("/api/label", {**body, "decision": "maybe", "reviewer": "A"})[0] == 400


def test_reviewing_a_case_that_is_not_split_is_refused(site):
    code, _ = site["post"]("/api/review", {"run_id": RUN, "case_id": LABEL_CASE, "trial": 0,
                                           "decision": "PASS", "reviewer": "A"})
    assert code == 409


def test_judges_page_counts_labels(site):
    site["post"]("/api/label", {"run_id": RUN, "case_id": LABEL_CASE, "trial": 0,
                                "decision": "PASS", "reviewer": "A"})
    d = site["get"](f"/api/judges?run={RUN}")
    [jev] = [j for j in d["judges"] if j["grader"] == "judge.rubric"]
    assert jev["labelled"] == 1 and jev["agreed"] == 1
    assert d["rubric_approval"]["ok"] is False             # the shipped template


@pytest.mark.parametrize("path", ["/api/runs/..%2F..%2Fetc", "/api/runs/nope",
                                  f"/api/runs/{RUN}/cases/..%2Fmanifest/0"])
def test_paths_outside_runs_are_refused(site, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        site["get"](path)
    assert e.value.code in (400, 404)


def test_label_items_carry_the_tool_facts_but_not_the_verdict(site):
    [item] = [i for i in site["get"](f"/api/label?run={RUN}")["items"] if i["case_id"] == LABEL_CASE]
    assert "tools" in item and "transcript" in item
    assert "0.97" not in json.dumps(item)


def test_label_cards_carry_guidance_from_the_rubric_and_suite(site):
    d = site["get"](f"/api/label?run={RUN}")
    [item] = [i for i in d["items"] if i["case_id"] == LABEL_CASE]
    assert item["guide"]["pass"] and item["guide"]["fail"]        # from the jev rubric's criteria
    assert item["notes"]                                           # what the case tests
    assert d["guide"]["agent_rules"]                               # suite labeling_guide.json


def test_llm_judge_guidance_comes_from_the_vendored_rubric(tmp_path):
    from evalkit.ui.server import Dashboard
    g = Dashboard(tmp_path, Path("suites/support-agent"))._rubric_guide("answer_quality")
    assert {c["name"] for c in g["criteria"]} == {"correctness", "relevance", "completeness", "clarity"}
    assert "3.5" in g["rule"]
