"""decide() must gate on PASSED, not on raw value.

A grader's `value` is confidence/evidence for a human to read - it is not
automatically a pass fraction. This is the regression test for a real bug:
a judge score of 0.97 (passed=True) dragged an otherwise-unanimous pass
below the MAJOR threshold because decide() was averaging raw values.
"""

from evalkit.graders.composite import decide
from evalkit.schema.score import Score, Severity


def major(value: float, passed: bool) -> Score:
    return Score(grader="g", grader_version="1", value=value, passed=passed,
                severity=Severity.MAJOR)


def critical(passed: bool) -> Score:
    return Score(grader="g", grader_version="1", value=1.0 if passed else 0.0,
                passed=passed, severity=Severity.CRITICAL)


def test_a_high_but_imperfect_confidence_score_does_not_sink_a_unanimous_pass():
    """Every grader passed. One of them (the judge) is 0.97 confident, not
    1.0. The case must still pass - a real number is not a partial failure."""
    scores = [major(1.0, True), major(1.0, True), major(0.97, True)]
    assert decide(scores) is True


def test_a_failed_major_still_fails_regardless_of_its_value():
    scores = [major(1.0, True), major(0.05, False)]
    assert decide(scores) is False


def test_critical_failure_vetoes_even_with_perfect_majors():
    scores = [major(1.0, True), critical(False)]
    assert decide(scores) is False


def test_no_majors_and_no_critical_failures_passes():
    assert decide([critical(True)]) is True
