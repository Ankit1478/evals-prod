"""Compare two runs and say whether the difference is REAL or just NOISE.

Paired bootstrap: pair the SAME case_id across both runs (not raw scores),
because the question that matters is "if we had drawn a different set of
test cases, would this delta survive?" - and pairing is what stops one
flaky case from being counted as two unrelated data points.
"""

from __future__ import annotations

import random

from evalkit.schema.case import Frozen
from evalkit.stats.summary import CaseSummary


class CaseDelta(Frozen):
    case_id: str
    kind: str
    before: float          # pass_hat_k in the "before" run
    after: float            # pass_hat_k in the "after" run
    delta: float


class DiffResult(Frozen):
    kind: str
    cases: int
    improved: int
    regressed: int
    mean_delta: float
    ci_low: float
    ci_high: float
    significant: bool       # CI does not cross zero


def pair_cases(before: list[CaseSummary], after: list[CaseSummary]) -> list[CaseDelta]:
    """Match cases present in BOTH runs. A case only in one run cannot be
    compared, so it is dropped rather than silently scored as unchanged."""
    by_id_after = {c.case_id: c for c in after}
    out: list[CaseDelta] = []
    for b in before:
        a = by_id_after.get(b.case_id)
        if a is None:
            continue
        out.append(CaseDelta(case_id=b.case_id, kind=b.kind,
                             before=b.pass_hat_k, after=a.pass_hat_k,
                             delta=a.pass_hat_k - b.pass_hat_k))
    return out


def bootstrap_diff(deltas: list[CaseDelta], b: int = 10_000,
                   seed: int = 0) -> tuple[float, float]:
    """Resample the paired per-case deltas (with replacement) b times and
    take the middle 95% - the same trick as bootstrap.percentile_bootstrap,
    applied to the DIFFERENCE instead of a raw rate."""
    if not deltas:
        return (0.0, 0.0)
    values = [d.delta for d in deltas]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(b))
    lo = means[int(0.025 * b)]
    hi = means[min(b - 1, int(0.975 * b))]
    return (lo, hi)


def diff_suites(before: list[CaseSummary], after: list[CaseSummary]) -> list[DiffResult]:
    deltas = pair_cases(before, after)
    out: list[DiffResult] = []
    for kind in sorted({d.kind for d in deltas}):
        group = [d for d in deltas if d.kind == kind]
        lo, hi = bootstrap_diff(group)
        out.append(DiffResult(
            kind=kind,
            cases=len(group),
            improved=sum(1 for d in group if d.delta > 0),
            regressed=sum(1 for d in group if d.delta < 0),
            mean_delta=sum(d.delta for d in group) / len(group),
            ci_low=lo, ci_high=hi,
            significant=not (lo <= 0 <= hi),
        ))
    return out
