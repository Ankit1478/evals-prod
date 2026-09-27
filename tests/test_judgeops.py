"""Measuring the judge against humans.

kappa rather than raw agreement, because raw agreement flatters a judge
that has learned nothing: on a suite where 90% of cases pass, always
answering PASS scores 90%.
"""

import json
from pathlib import Path

import pytest

from evalkit.judgeops import (KAPPA_FLOOR, agreement, judge_decisions,
                              load_gold_labels)
from evalkit.schema.score import CaseResult, Score, Severity


def judged(case_id: str, passed: bool | None, abstained: bool = False,
           grader: str = "judge.rubric") -> CaseResult:
    s = Score(grader=grader, grader_version="1", value=1.0 if passed else 0.0,
              passed=passed, severity=Severity.MAJOR, abstained=abstained)
    return CaseResult(case_id=case_id, trial_index=0, scores=[s],
                      passed=bool(passed), stop_reason="completed")


def test_gold_labels_load_and_reject_a_bad_verdict(tmp_path):
    p = tmp_path / "gold.jsonl"
    p.write_text('{"case_id": "c0", "human_decision": "PASS"}\n'
                 '{"case_id": "c1", "human_decision": "fail"}\n')
    assert load_gold_labels(p) == {"c0": "PASS", "c1": "FAIL"}

    p.write_text('{"case_id": "c0", "human_decision": "maybe"}\n')
    with pytest.raises(ValueError):
        load_gold_labels(p)


def test_an_abstained_judge_score_is_not_counted_as_a_verdict():
    """A judge that declined to answer has no opinion to compare."""
    got = judge_decisions([judged("c0", True), judged("c1", None, abstained=True)])
    assert got == {"c0": "PASS"}


def test_only_judge_graders_are_compared_not_rule_based_ones():
    got = judge_decisions([judged("c0", True, grader="tools.selection"),
                           judged("c1", False, grader="judge.answer_quality")])
    assert got == {"c1": "FAIL"}


def test_perfect_agreement_gives_kappa_one_and_clears_the_floor():
    judge = {"c0": "PASS", "c1": "FAIL", "c2": "PASS", "c3": "FAIL"}
    rep = agreement(judge, dict(judge))
    assert rep.kappa == 1.0
    assert rep.meets_floor is True
    assert rep.agreed == 4 and rep.cases == 4


def test_a_judge_that_always_says_pass_is_caught_by_kappa():
    """9 of 10 cases really do pass. A judge that blindly answers PASS
    agrees 90% of the time and has learned nothing - kappa says so."""
    human = {f"c{i}": ("PASS" if i < 9 else "FAIL") for i in range(10)}
    judge = {f"c{i}": "PASS" for i in range(10)}
    rep = agreement(judge, human)
    assert rep.agreement_rate == 0.9          # looks great
    assert rep.kappa == 0.0                   # knows nothing
    assert rep.meets_floor is False


def test_only_cases_present_in_both_are_compared():
    rep = agreement({"c0": "PASS", "c9": "PASS"}, {"c0": "PASS"})
    assert rep.cases == 1


def test_no_shared_cases_reports_nothing_rather_than_a_fake_score():
    rep = agreement({"c0": "PASS"}, {"c1": "PASS"})
    assert rep.cases == 0
    assert rep.kappa is None
    assert rep.meets_floor is False


def test_the_floor_is_ours_not_the_subsystems():
    """Upstream has no absolute kappa floor - only a relative regression
    tolerance. This constant is a product decision made here."""
    assert KAPPA_FLOOR == 0.70


# --- pinned labels and the fitted pass line ------------------------------------

def _judged_reply(case_id, reply, passed=True, value=1.0, grader="judge.rubric", **ev):
    return CaseResult(case_id=case_id, trial_index=0, stop_reason="completed", passed=True,
                      scores=[Score(grader=grader, grader_version="1", value=value,
                                    passed=passed, severity=Severity.MAJOR,
                                    evidence={"reply": reply, **ev})])


def test_a_pinned_label_counts_only_for_the_reply_the_human_read():
    from evalkit.judgeops import GoldLabel, labels_for_run
    from evalkit.review import reply_hash

    rows = [GoldLabel(case_id="c1", human_decision="FAIL", reply_sha256=reply_hash("old reply"))]
    assert labels_for_run(rows, [_judged_reply("c1", "old reply")]) == {"c1": "FAIL"}
    assert labels_for_run(rows, [_judged_reply("c1", "a new reply")]) == {}


def test_an_unpinned_label_is_case_level():
    from evalkit.judgeops import GoldLabel, labels_for_run
    rows = [GoldLabel(case_id="c1", human_decision="PASS")]
    assert labels_for_run(rows, [_judged_reply("c1", "anything")]) == {"c1": "PASS"}


def test_the_latest_label_wins():
    from evalkit.judgeops import GoldLabel, labels_for_run
    rows = [GoldLabel(case_id="c1", human_decision="PASS"),
            GoldLabel(case_id="c1", human_decision="FAIL")]
    assert labels_for_run(rows, [_judged_reply("c1", "r")]) == {"c1": "FAIL"}


def test_the_pass_line_is_fitted_to_the_humans():
    from evalkit.judgeops import suggest_threshold
    # Humans pass everything at 0.6 and above: 0.5 would be too lenient.
    pairs = [(0.9, "PASS"), (0.8, "PASS"), (0.7, "PASS"), (0.65, "PASS"), (0.6, "PASS"),
             (0.55, "FAIL"), (0.5, "FAIL"), (0.4, "FAIL"), (0.2, "FAIL"), (0.1, "FAIL")]
    t, agree = suggest_threshold(pairs)
    assert 0.55 < t <= 0.6 and agree == 1.0


def test_no_threshold_is_suggested_from_a_handful_of_labels():
    from evalkit.judgeops import suggest_threshold
    assert suggest_threshold([(0.9, "PASS"), (0.1, "FAIL")]) is None


def test_ef_label_pins_the_reply_and_hides_the_judges_verdict(tmp_path, monkeypatch):
    import shutil

    from typer.testing import CliRunner

    from evalkit.cli import app
    from evalkit.judgeops import load_gold_label_rows
    from evalkit.review import reply_hash

    suite = tmp_path / "suite"
    shutil.copytree(Path("suites/support-agent"), suite)
    (suite / "gold_labels.jsonl").unlink(missing_ok=True)    # the real suite's labels are not ours
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    row = _judged_reply("order_cancel_empathy_11", "So sorry - I can't cancel 456.",
                        value=0.97, threshold=0.5).model_dump(mode="json")
    (run / "scores.jsonl").write_text(json.dumps({**row, "kind": "capability"}) + "\n")
    monkeypatch.chdir(tmp_path)

    listing = CliRunner().invoke(app, ["label", "--run", "r1", "--suite", str(suite)])
    assert listing.exit_code == 0
    assert "So sorry" in listing.output and "0.97" not in listing.output   # no anchoring

    out = CliRunner().invoke(app, ["label", "order_cancel_empathy_11", "--decision", "pass",
                                   "--reviewer", "Ankit", "--run", "r1", "--suite", str(suite)])
    assert out.exit_code == 0
    [label] = load_gold_label_rows(suite / "gold_labels.jsonl")
    assert label.reply_sha256 == reply_hash("So sorry - I can't cancel 456.")
    assert label.reviewer == "Ankit"
