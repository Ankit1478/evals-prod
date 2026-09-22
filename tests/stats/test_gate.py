"""The gate must never say 'ship it' on missing data.

A green CI badge on a crashed eval is the most common production failure in
this whole system.
"""

from evalkit.schema.score import CaseResult, Score, Severity
from evalkit.stats.gate import EXIT_FAILED, EXIT_INVALID, EXIT_PASS, run_gate

THRESHOLDS = {"critical_failures_allowed": 0, "invalid_trials_allowed": 0,
              "min_pass_rate": {"regression": 0.95, "adversarial": 1.0}}


def result(case_id, passed=True, stop="completed", critical_failed=False):
    scores = [Score(grader="g", grader_version="1", value=1.0 if passed else 0.0,
                    passed=not critical_failed,
                    severity=Severity.CRITICAL if critical_failed else Severity.MAJOR)]
    return CaseResult(case_id=case_id, trial_index=0, scores=scores,
                      passed=passed, stop_reason=stop)


def kinds(*ids, kind="regression"):
    return {i: kind for i in ids}


def test_empty_run_is_invalid_never_a_pass():
    v = run_gate([], {}, THRESHOLDS)
    assert v.passed is False
    assert v.exit_code == EXIT_INVALID


def test_a_crashed_trial_blocks_even_when_everything_else_passed():
    results = [result(f"c{i}") for i in range(9)] + [result("c9", stop="error")]
    v = run_gate(results, kinds(*[f"c{i}" for i in range(10)]), THRESHOLDS)
    assert v.exit_code == EXIT_INVALID
    assert v.blocked_by.startswith("0")       # validity, before any score


def test_validity_is_checked_before_scores():
    """Even a 100% score cannot rescue a run that did not complete."""
    results = [result("c0", stop="timeout")]
    v = run_gate(results, kinds("c0"), THRESHOLDS)
    assert v.exit_code == EXIT_INVALID


def test_one_critical_failure_vetoes_a_high_score():
    results = [result(f"c{i}") for i in range(99)]
    results.append(result("c99", passed=False, critical_failed=True))
    v = run_gate(results, kinds(*[f"c{i}" for i in range(100)]), THRESHOLDS)
    assert v.exit_code == EXIT_FAILED
    assert v.blocked_by.startswith("1")       # critical, before the threshold


def test_all_good_ships():
    results = [result(f"c{i}") for i in range(10)]
    v = run_gate(results, kinds(*[f"c{i}" for i in range(10)]), THRESHOLDS)
    assert v.passed is True and v.exit_code == EXIT_PASS


def test_threshold_blocks_a_merely_mediocre_run():
    results = [result(f"c{i}", passed=i < 8) for i in range(10)]
    v = run_gate(results, kinds(*[f"c{i}" for i in range(10)]), THRESHOLDS)
    assert v.exit_code == EXIT_FAILED
    assert v.blocked_by.startswith("2")


def test_a_flaky_case_fails_pass_hat_k():
    """Same case, 5 trials, one fails -> the case is NOT reliable."""
    results = [CaseResult(case_id="c0", trial_index=t, scores=[],
                          passed=(t != 2), stop_reason="completed")
               for t in range(5)]
    v = run_gate(results, kinds("c0"), THRESHOLDS)
    assert v.exit_code == EXIT_FAILED
    assert "flaky" in str(v.stages[-1].detail)


# --- rubric governance (vendored LLM_AS_JUDGE) -------------------------------

def test_rubric_approval_is_off_by_default_and_does_not_add_a_stage():
    v = run_gate([result("c0")], kinds("c0"), THRESHOLDS)
    assert all(not s.stage.startswith("0b") for s in v.stages)


def test_the_shipped_template_approval_is_rejected(tmp_path):
    """The example approval file matches the rubric's fingerprint but is a
    DRAFT with no reviewers - it must never pass as production sign-off."""
    import shutil
    from pathlib import Path
    shutil.copy(Path("vendor/llm_judge/config/rubric_approval.example.json"),
                tmp_path / "rubric_approval.json")
    th = {**THRESHOLDS, "require_rubric_approval": True}
    v = run_gate([result("c0")], kinds("c0"), th, suite_dir=tmp_path)
    assert v.exit_code == EXIT_INVALID
    assert v.blocked_by.startswith("0b")


def test_a_missing_approval_file_blocks_when_required(tmp_path):
    th = {**THRESHOLDS, "require_rubric_approval": True}
    v = run_gate([result("c0")], kinds("c0"), th, suite_dir=tmp_path)
    assert v.exit_code == EXIT_INVALID
    assert "no rubric approval file" in v.stages[-1].detail
