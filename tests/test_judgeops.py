"""Measuring the judge against humans.

kappa rather than raw agreement, because raw agreement flatters a judge
that has learned nothing: on a suite where 90% of cases pass, always
answering PASS scores 90%.
"""

import json

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
