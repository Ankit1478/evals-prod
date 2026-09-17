"""Confidence intervals.

A bare percentage is not a result. 8/10 is not '80%' - it is 'somewhere
between 49% and 94%, and we cannot tell yet'. Most '+2 points' claims are
noise, and a CI is what tells you so.
"""

from __future__ import annotations

import random

Z95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval - correct for SMALL samples.

    The textbook normal approximation breaks badly below a few hundred
    samples (and gives impossible answers like 104% at the extremes).
    Eval suites are almost always small, so Wilson is the default here.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def percentile_bootstrap(values: list[float], b: int = 10_000,
                         alpha: float = 0.05,
                         seed: int = 0) -> tuple[float, float]:
    """Resample the CASES (with replacement) b times, take the middle 95%.

    Resampling cases - not trials - is what makes the interval answer the
    question you actually care about: 'if I had drawn a different set of test
    cases, how different would this number be?'
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(b))
    lo = means[int((alpha / 2) * b)]
    hi = means[min(b - 1, int((1 - alpha / 2) * b))]
    return (lo, hi)
