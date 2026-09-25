"""Human review must settle exactly the case it was written for - and only
the answer it was written for."""

import pytest

from evalkit.review import (REVIEW_GRADER, append_review, apply_reviews,
                            load_reviews, new_review, reply_hash)
from evalkit.schema.score import CaseResult, Score, Severity
from evalkit.stats.gate import EXIT_NEEDS_REVIEW, EXIT_PASS, EXIT_FAILED, run_gate

THRESHOLDS = {"min_pass_rate": {"regression": 0.0}}


def split_case(case_id="c13", reply="Your order ships Friday."):
    """Judges disagreed: the judge abstained, so decide() said passed."""
    return CaseResult(case_id=case_id, trial_index=0, stop_reason="completed",
                      passed=True, scores=[
        Score(grader="tools.selection", grader_version="1", value=1.0,
              passed=True, severity=Severity.MAJOR),
        Score(grader="judge.answer_quality", grader_version="1", value=0.0,
              passed=None, severity=Severity.MAJOR, abstained=True,
              evidence={"requires_human_review": True, "reply": reply},
              explanation="terra and luna disagreed")])


def reviewed(tmp_path, result, decision, reviewer="Ankit"):
    path = tmp_path / "reviews.jsonl"
    append_review(path, new_review(result, decision, reviewer, note="checked"))
    return load_reviews(path)


def test_a_human_fail_fails_the_case_and_clears_the_review(tmp_path):
    r = split_case()
    [out] = apply_reviews([r], reviewed(tmp_path, r, "fail"))
    assert out.passed is False
    assert out.needs_review is False
    human = [s for s in out.scores if s.grader == REVIEW_GRADER][0]
    assert human.evidence["reviewer"] == "Ankit"
    assert run_gate([out], {"c13": "regression"},
                    {"min_pass_rate": {"regression": 1.0}}).exit_code == EXIT_FAILED


def test_a_human_pass_lets_the_gate_through(tmp_path):
    r = split_case()
    assert run_gate([r], {"c13": "regression"}, THRESHOLDS).exit_code == EXIT_NEEDS_REVIEW
    [out] = apply_reviews([r], reviewed(tmp_path, r, "pass"))
    assert out.passed is True
    assert run_gate([out], {"c13": "regression"}, THRESHOLDS).exit_code == EXIT_PASS


def test_a_review_does_not_carry_over_to_a_different_reply(tmp_path):
    """The agent answered differently this time - the old verdict is about
    a different answer, so the case is pending again."""
    reviews = reviewed(tmp_path, split_case(reply="old answer"), "pass")
    [out] = apply_reviews([split_case(reply="a new, different answer")], reviews)
    assert out.needs_review is True


def test_the_same_reply_reuses_the_review(tmp_path):
    reviews = reviewed(tmp_path, split_case(reply="same"), "pass")
    [out] = apply_reviews([split_case(reply="  same  ")], reviews)   # whitespace ignored
    assert out.needs_review is False


def test_the_last_review_wins(tmp_path):
    r = split_case()
    path = tmp_path / "reviews.jsonl"
    append_review(path, new_review(r, "pass", "Ankit"))
    append_review(path, new_review(r, "fail", "Ankit", note="changed my mind"))
    [out] = apply_reviews([r], load_reviews(path))
    assert out.passed is False


def test_a_human_pass_does_not_undo_another_graders_failure(tmp_path):
    r = split_case()
    bad = r.model_copy(update={"scores": [
        Score(grader="safety.leak", grader_version="1", value=0.0, passed=False,
              severity=Severity.CRITICAL), *r.scores]})
    [out] = apply_reviews([bad], reviewed(tmp_path, bad, "pass"))
    assert out.passed is False


def test_applying_twice_adds_one_human_score(tmp_path):
    r = split_case()
    reviews = reviewed(tmp_path, r, "pass")
    [out] = apply_reviews(apply_reviews([r], reviews), reviews)
    assert sum(s.grader == REVIEW_GRADER for s in out.scores) == 1


def test_a_review_needs_a_named_reviewer_and_a_real_decision():
    r = split_case()
    with pytest.raises(ValueError):
        new_review(r, "pass", "   ")
    with pytest.raises(ValueError):
        new_review(r, "maybe", "Ankit")


def test_only_a_disputed_case_can_be_reviewed():
    plain = CaseResult(case_id="c", trial_index=0, stop_reason="completed",
                       passed=True, scores=[])
    with pytest.raises(ValueError, match="not waiting"):
        new_review(plain, "pass", "Ankit")


def test_a_corrupt_reviews_file_is_an_error_not_silence(tmp_path):
    path = tmp_path / "reviews.jsonl"
    path.write_text('{"case_id": "c13"}\n')
    with pytest.raises(ValueError, match="reviews.jsonl:1"):
        load_reviews(path)


def test_reply_hash_is_stable():
    assert reply_hash("x") == reply_hash(" x ")
    assert reply_hash("x") != reply_hash("y")
