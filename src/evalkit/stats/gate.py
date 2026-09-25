"""The GATE: turn results into a ship / do-not-ship decision.

Stages run IN ORDER and the first failure stops everything. Validity is
stage 0 - before any score is looked at - because the most common production
failure is a green badge on an eval that crashed and graded nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from evalkit.schema.case import Frozen
from evalkit.schema.score import CaseResult, Severity
from evalkit.stats.bootstrap import wilson
from evalkit.stats.summary import summarise_cases

# exit codes CI reads
EXIT_PASS = 0
EXIT_FAILED = 2      # the agent did not meet the bar
EXIT_INVALID = 3     # the run itself is broken - never treat as a pass
EXIT_NEEDS_REVIEW = 4  # a human must decide some cases before anyone can say

INVALID_STOPS = {"error", "timeout"}


class StageResult(Frozen):
    stage: str
    passed: bool
    detail: str


class Verdict(Frozen):
    passed: bool
    exit_code: int
    stages: list[StageResult]

    @property
    def blocked_by(self) -> str | None:
        for s in self.stages:
            if not s.passed:
                return s.stage
        return None


def check_rubric_approval(approval_path: Path) -> tuple[bool, str]:
    """Is the judge's rubric approved for production use?

    Delegates entirely to the vendored governance module, which fingerprints
    the rubric (SHA-256 over its canonical form) so that editing a single
    word of it invalidates the prior human sign-off. A judge grading against
    an unapproved rubric is not a result anyone should ship on.
    """
    from llm_judge.rubric import ACTIVE_RUBRIC
    from llm_judge.rubric_approval import (load_rubric_approval,
                                           validate_rubric_approval)

    if not approval_path.exists():
        return False, f"no rubric approval file at {approval_path}"
    try:
        approval = load_rubric_approval(approval_path)
        v = validate_rubric_approval(ACTIVE_RUBRIC, approval)
    except Exception as e:
        return False, f"rubric approval file is unusable: {type(e).__name__}: {e}"

    if v.valid_for_production:
        return True, (f"rubric '{v.rubric_name}' v{v.rubric_version} approved "
                      f"({v.approval_id})")
    return False, (f"rubric '{v.rubric_name}' v{v.rubric_version} is NOT approved "
                   f"for production - failed: {', '.join(v.failed_check_ids)}")


def run_gate(results: list[CaseResult], kinds: dict[str, str],
             thresholds: dict, suite_dir: Path | None = None) -> Verdict:
    """results  - one per case/trial
       kinds    - case_id -> "regression" | "adversarial" | "capability"
       thresholds - loaded from thresholds.json
       suite_dir  - needed only for the optional rubric-approval check
    """
    stages: list[StageResult] = []

    # ---- stage 0: VALIDITY -------------------------------------------------
    # Did the run actually happen? Checked BEFORE any score.
    invalid = [r for r in results if r.stop_reason in INVALID_STOPS]
    allowed = thresholds.get("invalid_trials_allowed", 0)
    if not results:
        stages.append(StageResult(stage="0 VALIDITY", passed=False,
                                  detail="no results at all - the run produced nothing"))
        return Verdict(passed=False, exit_code=EXIT_INVALID, stages=stages)
    if len(invalid) > allowed:
        stages.append(StageResult(
            stage="0 VALIDITY", passed=False,
            detail=f"{len(invalid)} trial(s) crashed or timed out "
                   f"({', '.join(r.case_id for r in invalid[:3])})"))
        return Verdict(passed=False, exit_code=EXIT_INVALID, stages=stages)
    stages.append(StageResult(stage="0 VALIDITY", passed=True,
                              detail=f"{len(results)} trial(s) completed, none invalid"))

    # ---- stage 0b: RUBRIC APPROVAL ----------------------------------------
    # Governance, not statistics: was the rubric the judge used actually
    # signed off by humans? Opt-in, because real approval needs real
    # reviewers - a shipped template deliberately fails this.
    if thresholds.get("require_rubric_approval", False):
        path = (suite_dir or Path(".")) / "rubric_approval.json"
        ok, detail = check_rubric_approval(path)
        stages.append(StageResult(stage="0b RUBRIC", passed=ok, detail=detail))
        if not ok:
            return Verdict(passed=False, exit_code=EXIT_INVALID, stages=stages)

    # ---- stage 1: CRITICAL -------------------------------------------------
    # One critical failure vetoes everything, whatever the overall score is.
    crit = [(r.case_id, s.grader) for r in results for s in r.scores
            if s.severity is Severity.CRITICAL and s.passed is False]
    max_crit = thresholds.get("critical_failures_allowed", 0)
    if len(crit) > max_crit:
        stages.append(StageResult(
            stage="1 CRITICAL", passed=False,
            detail=f"{len(crit)} critical failure(s): "
                   + ", ".join(f"{c}/{g}" for c, g in crit[:4])))
        return Verdict(passed=False, exit_code=EXIT_FAILED, stages=stages)
    stages.append(StageResult(stage="1 CRITICAL", passed=True,
                              detail="no critical failures"))

    # ---- stage 1b: HUMAN REVIEW --------------------------------------------
    # Before the threshold, because the threshold would count these cases as
    # passes: their verdict is not known yet, so no pass rate is either.
    pending = [r.case_id for r in results if r.needs_review]
    max_pending = thresholds.get("needs_review_allowed", 0)
    if len(pending) > max_pending:
        stages.append(StageResult(
            stage="1b REVIEW", passed=False,
            detail=f"{len(pending)} case(s) need a human verdict - the judges "
                   f"disagreed: {', '.join(pending[:4])}"))
        return Verdict(passed=False, exit_code=EXIT_NEEDS_REVIEW, stages=stages)
    stages.append(StageResult(stage="1b REVIEW", passed=True,
                              detail=f"{len(pending)} case(s) awaiting human review "
                                     f"(allowed {max_pending})"))

    # ---- stage 2: THRESHOLD ------------------------------------------------
    # Each kind of case has its own bar.
    # pass^k, not the per-trial average: a case counts only if EVERY trial
    # passed. A case that works 4 times out of 5 is not something you ship.
    mins = thresholds.get("min_pass_rate", {})
    use_ci = thresholds.get("use_ci_lower", False)
    per_case = summarise_cases(results, kinds)
    failures: list[str] = []
    details: list[str] = []

    for kind in sorted({c.kind for c in per_case}):
        group = [c for c in per_case if c.kind == kind]
        reliable = sum(1 for c in group if c.pass_hat_k == 1.0)
        rate = reliable / len(group)
        lo, _ = wilson(reliable, len(group))
        measured = lo if use_ci else rate
        need = mins.get(kind, 0.0)
        label = f"{rate:.0%}" + (f" (CI low {lo:.0%})" if use_ci else "")
        mark = "ok" if measured >= need else "BELOW"
        details.append(f"{kind} pass^k {label} (need {need:.0%}) {mark}")
        if measured < need:
            flaky = [c.case_id for c in group if c.flaky]
            note = f" [flaky: {', '.join(flaky)}]" if flaky else ""
            failures.append(f"{kind}: {label} < {need:.0%}{note}")
    if failures:
        stages.append(StageResult(stage="2 THRESHOLD", passed=False,
                                  detail="; ".join(failures)))
        return Verdict(passed=False, exit_code=EXIT_FAILED, stages=stages)
    stages.append(StageResult(stage="2 THRESHOLD", passed=True,
                              detail="; ".join(details)))

    # stage 3 (no regression vs a baseline run) arrives with Box 7 - it needs
    # statistics to tell a real change from noise.
    return Verdict(passed=True, exit_code=EXIT_PASS, stages=stages)


def load_results(run_dir: Path) -> tuple[list[CaseResult], dict[str, str]]:
    """Read scores.jsonl back off disk."""
    results, kinds = [], {}
    for line in (run_dir / "scores.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kinds[row["case_id"]] = row.pop("kind", "regression")
        results.append(CaseResult.model_validate(row))
    return results, kinds
