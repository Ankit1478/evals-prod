"""Known inputs -> known values. The stats decide releases; they get checked."""

import pytest

from evalkit.stats.bootstrap import wilson
from evalkit.stats.passk import pass_at_k, pass_hat_k


@pytest.mark.parametrize("n,c,k,want", [
    (5, 5, 1, 1.0),      # always passed
    (5, 0, 1, 0.0),      # never passed
    (5, 5, 5, 1.0),
    (5, 0, 5, 0.0),
])
def test_pass_at_k_known_values(n, c, k, want):
    assert pass_at_k(n, c, k) == pytest.approx(want)


def test_pass_at_k_is_generous_pass_hat_k_is_strict():
    """4 of 5 passed: 'could it ever' says yes, 'can I ship it' says no."""
    assert pass_at_k(5, 4, 5) == pytest.approx(1.0)
    assert pass_hat_k(4, 5) < 0.5


def test_pass_hat_k_decreases_with_more_attempts():
    """A 90% agent gets worse the more chances it needs in a row."""
    p = [pass_hat_k(9, 10, k) for k in (1, 2, 5, 8)]
    assert p == sorted(p, reverse=True)
    assert p[-1] < 0.5              # 90% agent is under 50% over 8 tries


def test_one_flaky_trial_destroys_pass_hat_k():
    assert pass_hat_k(5, 5) == 1.0
    assert pass_hat_k(4, 5) < 1.0


@pytest.mark.parametrize("c,n", [(0, 10), (8, 10), (10, 10), (1, 3)])
def test_wilson_stays_inside_0_and_1(c, n):
    """The textbook normal interval returns impossible values like 104% at
    the extremes. Wilson does not - that is why it is the default."""
    lo, hi = wilson(c, n)
    assert 0.0 <= lo <= hi <= 1.0


def test_small_samples_give_wide_intervals():
    """8/10 is not '80%'. It is 'somewhere between 49% and 94%'."""
    lo, hi = wilson(8, 10)
    assert hi - lo > 0.4
    lo2, hi2 = wilson(800, 1000)
    assert hi2 - lo2 < 0.1          # 100x the cases -> a usable number
