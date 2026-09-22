"""ef diff must tell a REAL change apart from noise, and must never compare
a case against itself if it does not exist in both runs.
"""

from evalkit.stats.compare import diff_suites, pair_cases
from evalkit.stats.summary import CaseSummary


def summary(case_id, kind="regression", pass_hat_k=1.0):
    return CaseSummary(case_id=case_id, kind=kind, trials=5, passed=5,
                       pass_at_1=1.0, pass_hat_k=pass_hat_k, flaky=False)


def test_a_case_missing_from_either_run_is_dropped_not_compared():
    before = [summary("c0"), summary("c1")]
    after = [summary("c0")]                        # c1 does not exist here
    deltas = pair_cases(before, after)
    assert [d.case_id for d in deltas] == ["c0"]


def test_a_consistent_improvement_across_many_cases_is_significant():
    before = [summary(f"c{i}", pass_hat_k=0.0) for i in range(20)]
    after = [summary(f"c{i}", pass_hat_k=1.0) for i in range(20)]
    result = diff_suites(before, after)[0]
    assert result.mean_delta == 1.0
    assert result.significant is True
    assert result.improved == 20 and result.regressed == 0


def test_a_single_flipped_case_among_many_is_not_significant():
    """One case improved, the other 19 unchanged - too little evidence to
    call it real, not just luck."""
    before = [summary(f"c{i}", pass_hat_k=1.0) for i in range(19)] + [summary("c19", pass_hat_k=0.0)]
    after = [summary(f"c{i}", pass_hat_k=1.0) for i in range(20)]
    result = diff_suites(before, after)[0]
    assert result.improved == 1 and result.regressed == 0
    assert result.significant is False


def test_no_change_is_never_significant():
    before = [summary(f"c{i}") for i in range(10)]
    after = [summary(f"c{i}") for i in range(10)]
    result = diff_suites(before, after)[0]
    assert result.mean_delta == 0.0
    assert result.significant is False


def test_a_regression_is_reported_as_negative_and_significant():
    before = [summary(f"c{i}", pass_hat_k=1.0) for i in range(20)]
    after = [summary(f"c{i}", pass_hat_k=0.0) for i in range(20)]
    result = diff_suites(before, after)[0]
    assert result.mean_delta == -1.0
    assert result.significant is True
    assert result.regressed == 20
