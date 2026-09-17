"""pass@k and pass^k.

Same 5 runs, two different questions:
    pass@k  did it work AT LEAST ONCE?   -> "is it capable?"
    pass^k  did it work EVERY TIME?      -> "can I ship it?"

The gate uses pass^k. A customer gets one attempt, not five.
"""

from __future__ import annotations


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator (Chen et al. 2021).

    n = trials run, c = trials that passed, k = how many attempts we imagine
    giving the agent. Answers: with k attempts, what is the chance of at
    least one success?
    """
    if k <= 0 or n <= 0:
        return 0.0
    if n - c < k:            # too few failures to ever pick k failures in a row
        return 1.0
    prob_all_fail = 1.0
    for i in range(k):
        prob_all_fail *= (n - c - i) / (n - i)
    return 1.0 - prob_all_fail


def pass_hat_k(c: int, n: int, k: int | None = None) -> float:
    """P(all k attempts succeed). The reliability number.

    With k == n this is simply 'did every trial pass' (1.0 or 0.0), which is
    what you want per case. Averaged over cases it becomes the suite's pass^k.
    """
    if n <= 0:
        return 0.0
    k = n if k is None else k
    return (c / n) ** k
