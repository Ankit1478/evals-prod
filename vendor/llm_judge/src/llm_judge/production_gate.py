"""Step 12: turn validated reliability evidence into a release decision."""

import argparse
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from .dataset import ReviewStatus
from .dataset_runner import CaseRunResult
from .error_analysis import (
    AnalysisTarget,
    ErrorAnalysisReport,
    calculate_error_analysis,
)
from .multi_judge import JudgeModel
from .reliability import ReliabilityReport, calculate_reliability, load_case_results
from .rubric import ACTIVE_RUBRIC
from .rubric_approval import (
    RubricApproval,
    RubricApprovalValidation,
    load_rubric_approval,
    validate_rubric_approval,
)
from .stability import (
    StabilityCaseResult,
    StabilityReport,
    calculate_stability_report,
    load_stability_results,
)


class GateDecision(str, Enum):
    """Whether the measured two-model judge is eligible for release."""

    PASSED = "PASSED"
    FAILED = "FAILED"


class GateComparator(str, Enum):
    """How one observed measurement is compared with its threshold."""

    AT_LEAST = ">="
    AT_MOST = "<="
    EQUAL = "=="


class ProductionThresholds(BaseModel):
    """Versioned starting thresholds; owners must adapt them to business risk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "1.0.0"
    minimum_completed_cases: int = Field(default=100, ge=1)
    maximum_runner_failure_rate: float = Field(default=0.01, ge=0, le=1)
    maximum_human_review_rate: float = Field(default=0.20, ge=0, le=1)
    maximum_model_disagreement_rate: float = Field(default=0.10, ge=0, le=1)
    minimum_cohens_kappa: float = Field(default=0.80, ge=-1, le=1)
    minimum_pass_fail_accuracy_lower_bound: float = Field(
        default=0.90, ge=0, le=1
    )
    maximum_false_pass_rate_upper_bound: float = Field(
        default=0.05, ge=0, le=1
    )
    maximum_false_fail_rate_upper_bound: float = Field(
        default=0.10, ge=0, le=1
    )
    minimum_pairwise_accuracy_lower_bound: float = Field(
        default=0.90, ge=0, le=1
    )
    minimum_score_correlation: float = Field(default=0.80, ge=-1, le=1)
    minimum_repeat_consistency: float = Field(default=0.95, ge=0, le=1)
    maximum_position_flip_rate: float = Field(default=0.05, ge=0, le=1)
    maximum_stability_failure_rate: float = Field(default=0.01, ge=0, le=1)


DEFAULT_PRODUCTION_THRESHOLDS = ProductionThresholds()


class ProductionGateCheck(BaseModel):
    """One transparent pass/fail rule in the release gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: str
    description: str
    source: str
    observed: Optional[float] = None
    comparator: GateComparator
    threshold: float
    passed: bool
    explanation: str


class ProductionGateReport(BaseModel):
    """Final decision plus every measurement used to reach it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: GateDecision
    threshold_version: str
    thresholds: ProductionThresholds
    rubric_approval: Optional[RubricApprovalValidation] = None
    total_checks: int = Field(ge=1)
    passed_checks: int = Field(ge=0)
    failed_checks: int = Field(ge=0)
    checks: List[ProductionGateCheck]
    failed_check_ids: List[str]
    summary: str
    warnings: List[str]


def _check(
    check_id: str,
    description: str,
    source: str,
    observed: Optional[float],
    comparator: GateComparator,
    threshold: float,
) -> ProductionGateCheck:
    """Evaluate one rule; unavailable required evidence always fails safely."""

    if observed is None:
        return ProductionGateCheck(
            check_id=check_id,
            description=description,
            source=source,
            comparator=comparator,
            threshold=threshold,
            passed=False,
            explanation="FAILED: required measurement is unavailable.",
        )
    if comparator == GateComparator.AT_LEAST:
        passed = observed >= threshold
    elif comparator == GateComparator.AT_MOST:
        passed = observed <= threshold
    else:
        passed = observed == threshold
    status = "PASSED" if passed else "FAILED"
    return ProductionGateCheck(
        check_id=check_id,
        description=description,
        source=source,
        observed=round(observed, 4),
        comparator=comparator,
        threshold=threshold,
        passed=passed,
        explanation=(
            f"{status}: observed {observed:.4f} {comparator.value} "
            f"required {threshold:.4f}."
        ),
    )


def evaluate_production_gate(
    reliability: ReliabilityReport,
    errors: ErrorAnalysisReport,
    stability: StabilityReport,
    *,
    draft_cases: int,
    matching_case_sets: bool,
    rubric_approval: Optional[RubricApprovalValidation] = None,
    thresholds: ProductionThresholds = DEFAULT_PRODUCTION_THRESHOLDS,
) -> ProductionGateReport:
    """Evaluate the saved Step 9–11 evidence against production thresholds."""

    aggregate_errors = errors.analyses[AnalysisTarget.AGGREGATE]
    pass_fail = aggregate_errors.pass_fail_overall
    pairwise = aggregate_errors.pairwise
    checks = [
        _check(
            "rubric_approval",
            "The exact active rubric must have valid human production approval",
            "rubric approval governance",
            (
                1.0
                if rubric_approval is not None
                and rubric_approval.valid_for_production
                else 0.0
            ),
            GateComparator.EQUAL,
            1.0,
        ),
        _check(
            "matching_case_sets",
            "Step 8 and Step 10 must cover the same case IDs",
            "dataset and stability results",
            1.0 if matching_case_sets else 0.0,
            GateComparator.EQUAL,
            1.0,
        ),
        _check(
            "no_draft_labels",
            "Every label must be human-reviewed",
            "dataset review_status",
            float(draft_cases),
            GateComparator.EQUAL,
            0.0,
        ),
        _check(
            "minimum_completed_cases",
            "Enough completed cases must support the decision",
            "Step 9 reliability report",
            float(reliability.completed_cases),
            GateComparator.AT_LEAST,
            float(thresholds.minimum_completed_cases),
        ),
        _check(
            "runner_failure_rate",
            "Step 8 evaluation failures must remain low",
            "Step 9 reliability report",
            reliability.failure_rate,
            GateComparator.AT_MOST,
            thresholds.maximum_runner_failure_rate,
        ),
        _check(
            "human_review_rate",
            "The combined judge's human-review workload must remain acceptable",
            "Step 9 reliability report",
            reliability.human_review_rate,
            GateComparator.AT_MOST,
            thresholds.maximum_human_review_rate,
        ),
        _check(
            "model_disagreement_rate",
            "Terra and Luna disagreement must remain low",
            "Step 9 reliability report",
            reliability.model_disagreement_rate,
            GateComparator.AT_MOST,
            thresholds.maximum_model_disagreement_rate,
        ),
        _check(
            "aggregate_cohens_kappa",
            "Combined decisions must agree with humans beyond chance",
            "Step 9 reliability report",
            reliability.aggregate_vs_human.cohens_kappa,
            GateComparator.AT_LEAST,
            thresholds.minimum_cohens_kappa,
        ),
        _check(
            "pass_fail_accuracy_lower_bound",
            "Conservative PASS/FAIL accuracy must be high enough",
            "Step 11 aggregate 95% confidence lower bound",
            pass_fail.accuracy.ci_lower,
            GateComparator.AT_LEAST,
            thresholds.minimum_pass_fail_accuracy_lower_bound,
        ),
        _check(
            "false_pass_rate_upper_bound",
            "Conservative false-pass risk must be low enough",
            "Step 11 aggregate 95% confidence upper bound",
            pass_fail.false_pass_rate.ci_upper,
            GateComparator.AT_MOST,
            thresholds.maximum_false_pass_rate_upper_bound,
        ),
        _check(
            "false_fail_rate_upper_bound",
            "Conservative false-fail risk must be low enough",
            "Step 11 aggregate 95% confidence upper bound",
            pass_fail.false_fail_rate.ci_upper,
            GateComparator.AT_MOST,
            thresholds.maximum_false_fail_rate_upper_bound,
        ),
        _check(
            "pairwise_accuracy_lower_bound",
            "Conservative pairwise accuracy must be high enough",
            "Step 11 aggregate 95% confidence lower bound",
            pairwise.accuracy.ci_lower,
            GateComparator.AT_LEAST,
            thresholds.minimum_pairwise_accuracy_lower_bound,
        ),
        _check(
            "score_correlation",
            "Combined rubric scores must track human scores",
            "Step 9 score correlation",
            reliability.score_correlation.aggregate_pearson,
            GateComparator.AT_LEAST,
            thresholds.minimum_score_correlation,
        ),
    ]
    for model in (JudgeModel.TERRA, JudgeModel.LUNA):
        summary = stability.per_model[model]
        model_name = model.name.lower()
        checks.extend(
            [
                _check(
                    f"{model_name}_repeat_consistency",
                    f"{model.value} must give consistent repeated decisions",
                    "Step 10 stability report",
                    summary.mean_repeat_consistency,
                    GateComparator.AT_LEAST,
                    thresholds.minimum_repeat_consistency,
                ),
                _check(
                    f"{model_name}_position_flip_rate",
                    f"{model.value} must not change with A/B position",
                    "Step 10 stability report",
                    summary.position_flip_rate,
                    GateComparator.AT_MOST,
                    thresholds.maximum_position_flip_rate,
                ),
                _check(
                    f"{model_name}_stability_failure_rate",
                    f"{model.value} repeated-call failures must remain low",
                    "Step 10 stability report",
                    summary.failure_rate,
                    GateComparator.AT_MOST,
                    thresholds.maximum_stability_failure_rate,
                ),
            ]
        )

    failed_ids = [check.check_id for check in checks if not check.passed]
    decision = GateDecision.FAILED if failed_ids else GateDecision.PASSED
    warnings = [
        "Default thresholds are learning-oriented starting points; owners must approve thresholds based on domain risk.",
        "A passed gate applies only to the tested rubric, prompts, models, deployments, and data distribution.",
        "Re-run the gate after changing the model, prompt, rubric, dataset, or production traffic distribution.",
    ]
    return ProductionGateReport(
        decision=decision,
        threshold_version=thresholds.version,
        thresholds=thresholds,
        rubric_approval=rubric_approval,
        total_checks=len(checks),
        passed_checks=len(checks) - len(failed_ids),
        failed_checks=len(failed_ids),
        checks=checks,
        failed_check_ids=failed_ids,
        summary=(
            "PASSED: all required production checks succeeded."
            if decision == GateDecision.PASSED
            else f"FAILED: {len(failed_ids)} required production check(s) failed."
        ),
        warnings=warnings,
    )


def build_production_gate(
    runner_results: Sequence[CaseRunResult],
    stability_results: Sequence[StabilityCaseResult],
    *,
    rubric_approval: RubricApproval,
    thresholds: ProductionThresholds = DEFAULT_PRODUCTION_THRESHOLDS,
    bootstrap_iterations: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> ProductionGateReport:
    """Build all prerequisite reports locally, then make the release decision."""

    runner_ids = {result.case_id for result in runner_results}
    stability_ids = {result.case_id for result in stability_results}
    draft_case_ids = {
        result.case_id
        for result in runner_results
        if result.review_status == ReviewStatus.DRAFT
    } | {
        result.case_id
        for result in stability_results
        if result.review_status == ReviewStatus.DRAFT
    }
    return evaluate_production_gate(
        calculate_reliability(runner_results),
        calculate_error_analysis(
            runner_results,
            bootstrap_iterations=bootstrap_iterations,
            confidence_level=confidence_level,
            random_seed=random_seed,
        ),
        calculate_stability_report(stability_results),
        draft_cases=len(draft_case_ids),
        matching_case_sets=runner_ids == stability_ids,
        rubric_approval=validate_rubric_approval(ACTIVE_RUBRIC, rubric_approval),
        thresholds=thresholds,
    )


def load_thresholds(path: Optional[Union[str, Path]]) -> ProductionThresholds:
    """Load an optional reviewed threshold policy or use the documented default."""

    if path is None:
        return DEFAULT_PRODUCTION_THRESHOLDS
    return ProductionThresholds.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def main() -> None:
    """Command-line entry point for the local Step 12 production gate."""

    parser = argparse.ArgumentParser(
        description="Evaluate whether the Terra/Luna judge passes release gates"
    )
    parser.add_argument(
        "--runner-results", required=True, help="Step 8 JSONL result file"
    )
    parser.add_argument(
        "--stability-results", required=True, help="Step 10 JSONL result file"
    )
    parser.add_argument("--thresholds", help="Optional threshold-policy JSON file")
    parser.add_argument(
        "--rubric-approval",
        required=True,
        help="Human approval JSON for the exact active rubric",
    )
    parser.add_argument("--output", help="Optional production-gate JSON report")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = build_production_gate(
        load_case_results(args.runner_results),
        load_stability_results(args.stability_results),
        rubric_approval=load_rubric_approval(args.rubric_approval),
        thresholds=load_thresholds(args.thresholds),
        bootstrap_iterations=args.bootstrap_iterations,
        confidence_level=args.confidence_level,
        random_seed=args.seed,
    )
    rendered = report.model_dump_json(indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open(
            "w" if args.overwrite else "x", encoding="utf-8"
        ) as handle:
            handle.write(rendered + "\n")
    print(rendered)
    if report.decision == GateDecision.FAILED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
