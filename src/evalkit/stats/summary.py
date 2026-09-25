"""Turn a pile of trial results into the numbers a human reads."""

from __future__ import annotations

from evalkit.schema.case import Frozen
from evalkit.schema.score import CaseResult
from evalkit.stats.bootstrap import wilson
from evalkit.stats.passk import pass_at_k, pass_hat_k


class CaseSummary(Frozen):
    case_id: str
    kind: str
    trials: int
    passed: int
    pending: int = 0        # trials awaiting a human verdict - not passes
    pass_at_1: float
    pass_hat_k: float       # 1.0 only if EVERY trial passed
    flaky: bool             # passed sometimes, failed sometimes


class SuiteSummary(Frozen):
    kind: str
    cases: int
    pass_at_1: float        # average over cases
    pass_hat_k: float       # share of cases that passed every trial
    ci_low: float
    ci_high: float
    flaky_cases: list[str]
    pending_cases: list[str] = []


def summarise_cases(results: list[CaseResult],
                    kinds: dict[str, str]) -> list[CaseSummary]:
    by_case: dict[str, list[CaseResult]] = {}
    for r in results:
        by_case.setdefault(r.case_id, []).append(r)

    out: list[CaseSummary] = []
    for cid, trials in by_case.items():
        n = len(trials)
        # A trial awaiting human review is not a pass: its verdict is not
        # known yet. Counting it as one would report a pass rate nobody
        # has actually established.
        waiting = sum(1 for t in trials if t.needs_review)
        c = sum(1 for t in trials if t.passed and not t.needs_review)
        failed = n - c - waiting
        out.append(CaseSummary(
            case_id=cid,
            kind=kinds.get(cid, "regression"),
            trials=n,
            passed=c,
            pending=waiting,
            pass_at_1=pass_at_k(n, c, 1),
            pass_hat_k=pass_hat_k(c, n),
            flaky=c > 0 and failed > 0,   # the dangerous middle
        ))
    return out


def summarise_suite(cases: list[CaseSummary]) -> list[SuiteSummary]:
    out: list[SuiteSummary] = []
    for kind in sorted({c.kind for c in cases}):
        group = [c for c in cases if c.kind == kind]
        n = len(group)
        # pass^k at suite level = how many cases were reliable EVERY trial
        reliable = sum(1 for c in group if c.pass_hat_k == 1.0)
        lo, hi = wilson(reliable, n)
        out.append(SuiteSummary(
            kind=kind,
            cases=n,
            pass_at_1=sum(c.pass_at_1 for c in group) / n,
            pass_hat_k=reliable / n,
            ci_low=lo, ci_high=hi,
            flaky_cases=[c.case_id for c in group if c.flaky],
            pending_cases=[c.case_id for c in group if c.pending],
        ))
    return out
