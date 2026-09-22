"""Step 16: developer-controlled calibration and held-out verification."""

import argparse
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from .contracts import EvaluationMode
from .dataset import EvaluationCase, EvaluationDataset, ReviewStatus, load_jsonl
from .dataset_runner import CaseRunResult
from .error_analysis import AnalysisTarget, ErrorAnalysisReport, calculate_error_analysis
from .reliability import ReliabilityReport, calculate_reliability, load_case_results


class DatasetSplitManifest(BaseModel):
    """Reproducible record of which cases developers may calibrate against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "1.0.0"
    random_seed: int
    calibration_fraction: float = Field(gt=0, lt=1)
    stratified_by: str = "evaluation_mode"
    source_cases: int = Field(ge=2)
    calibration_cases: int = Field(ge=1)
    heldout_cases: int = Field(ge=1)
    contains_drafts: bool
    source_fingerprint: str
    calibration_fingerprint: str
    heldout_fingerprint: str
    calibration_case_ids: List[str]
    heldout_case_ids: List[str]
    calibration_by_mode: Dict[EvaluationMode, int]
    heldout_by_mode: Dict[EvaluationMode, int]


class DatasetSplit(BaseModel):
    """The reusable calibration partition and untouched final-test partition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    calibration: EvaluationDataset
    heldout: EvaluationDataset
    manifest: DatasetSplitManifest


class MetricDirection(str, Enum):
    """Whether a healthy metric should move upward or downward."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class MetricChange(str, Enum):
    """Result of comparing one candidate measurement with its baseline."""

    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"
    UNAVAILABLE = "unavailable"


class CalibrationDecision(str, Enum):
    """Whether a proposed judge configuration should move to held-out testing."""

    ACCEPTED = "accepted_for_heldout_test"
    REJECTED = "rejected_due_to_regression"
    NEEDS_DEVELOPER_REVIEW = "needs_developer_review"


class CalibrationAcceptancePolicy(BaseModel):
    """Maximum tolerated regression for each calibration metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "1.0.0"
    accuracy_regression_tolerance: float = Field(default=0.0, ge=0, le=1)
    kappa_regression_tolerance: float = Field(default=0.02, ge=0, le=2)
    false_pass_increase_tolerance: float = Field(default=0.0, ge=0, le=1)
    false_fail_increase_tolerance: float = Field(default=0.02, ge=0, le=1)
    human_review_increase_tolerance: float = Field(default=0.05, ge=0, le=1)
    failure_increase_tolerance: float = Field(default=0.0, ge=0, le=1)
    pairwise_accuracy_regression_tolerance: float = Field(default=0.0, ge=0, le=1)
    score_correlation_regression_tolerance: float = Field(default=0.02, ge=0, le=2)


DEFAULT_CALIBRATION_POLICY = CalibrationAcceptancePolicy()


class MetricComparison(BaseModel):
    """Before/after value, effect size, direction, and regression judgment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    direction: MetricDirection
    baseline: Optional[float] = None
    candidate: Optional[float] = None
    delta: Optional[float] = None
    tolerated_regression: float = Field(ge=0)
    outcome: MetricChange
    explanation: str


class CalibrationComparisonReport(BaseModel):
    """Developer-reviewed comparison of two judge configurations on one split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_configuration: str
    candidate_configuration: str
    change_summary: str
    reviewed_by: Optional[str] = None
    split_fingerprint: str
    calibration_cases: int = Field(ge=1)
    decision: CalibrationDecision
    comparisons: List[MetricComparison]
    regressed_metrics: List[str]
    improved_metrics: List[str]
    fixed_false_pass_case_ids: List[str]
    new_false_pass_case_ids: List[str]
    fixed_false_fail_case_ids: List[str]
    new_false_fail_case_ids: List[str]
    fixed_pairwise_mismatch_case_ids: List[str]
    new_pairwise_mismatch_case_ids: List[str]
    warnings: List[str]


class HeldoutVerificationReport(BaseModel):
    """Final metrics for a locked configuration on untouched test case IDs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    configuration_version: str
    calibration_reviewed_by: str
    split_fingerprint: str
    heldout_cases: int = Field(ge=1)
    reliability: ReliabilityReport
    error_analysis: ErrorAnalysisReport
    warnings: List[str]


def _case_fingerprint(cases: Sequence[EvaluationCase]) -> str:
    payload = [case.model_dump(mode="json") for case in cases]
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _rank(seed: int, case_id: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def _count_by_mode(cases: Sequence[EvaluationCase]) -> Dict[EvaluationMode, int]:
    return {
        mode: sum(case.mode == mode for case in cases)
        for mode in EvaluationMode
    }


def split_evaluation_dataset(
    dataset: EvaluationDataset,
    *,
    calibration_fraction: float = 0.70,
    random_seed: int = 42,
    allow_drafts: bool = False,
) -> DatasetSplit:
    """Create deterministic, mode-stratified calibration and held-out datasets."""

    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    if len(dataset.cases) < 2:
        raise ValueError("At least two cases are needed for calibration and held-out data")
    if not allow_drafts:
        dataset.ensure_ready_for_production()

    calibration_ids: Set[str] = set()
    for mode in EvaluationMode:
        grouped = sorted(
            (case for case in dataset.cases if case.mode == mode),
            key=lambda case: _rank(random_seed, case.case_id),
        )
        if not grouped:
            continue
        desired = int(len(grouped) * calibration_fraction + 0.5)
        if len(grouped) > 1:
            desired = max(1, min(len(grouped) - 1, desired))
        else:
            desired = 1 if int(_rank(random_seed, grouped[0].case_id), 16) % 2 == 0 else 0
        calibration_ids.update(case.case_id for case in grouped[:desired])

    ranked_all = sorted(dataset.cases, key=lambda case: _rank(random_seed, case.case_id))
    if not calibration_ids:
        calibration_ids.add(ranked_all[0].case_id)
    if len(calibration_ids) == len(dataset.cases):
        calibration_ids.remove(ranked_all[-1].case_id)

    calibration_cases = [
        case for case in dataset.cases if case.case_id in calibration_ids
    ]
    heldout_cases = [
        case for case in dataset.cases if case.case_id not in calibration_ids
    ]
    calibration = EvaluationDataset(cases=calibration_cases)
    heldout = EvaluationDataset(cases=heldout_cases)
    manifest = DatasetSplitManifest(
        random_seed=random_seed,
        calibration_fraction=calibration_fraction,
        source_cases=len(dataset.cases),
        calibration_cases=len(calibration_cases),
        heldout_cases=len(heldout_cases),
        contains_drafts=any(
            case.review_status == ReviewStatus.DRAFT for case in dataset.cases
        ),
        source_fingerprint=_case_fingerprint(dataset.cases),
        calibration_fingerprint=_case_fingerprint(calibration_cases),
        heldout_fingerprint=_case_fingerprint(heldout_cases),
        calibration_case_ids=[case.case_id for case in calibration_cases],
        heldout_case_ids=[case.case_id for case in heldout_cases],
        calibration_by_mode=_count_by_mode(calibration_cases),
        heldout_by_mode=_count_by_mode(heldout_cases),
    )
    return DatasetSplit(
        calibration=calibration,
        heldout=heldout,
        manifest=manifest,
    )


def _write_jsonl(path: Path, cases: Sequence[EvaluationCase]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for case in cases:
            handle.write(case.model_dump_json() + "\n")


def save_dataset_split(
    split: DatasetSplit,
    *,
    calibration_path: Union[str, Path],
    heldout_path: Union[str, Path],
    manifest_path: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """Save all split artifacts while refusing accidental partial replacement."""

    destinations = [Path(calibration_path), Path(heldout_path), Path(manifest_path)]
    if len(set(destinations)) != 3:
        raise ValueError("Calibration, held-out, and manifest paths must be different")
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Split output already exists: {existing}")
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and path.exists():
            path.unlink()
    _write_jsonl(destinations[0], split.calibration.cases)
    _write_jsonl(destinations[1], split.heldout.cases)
    with destinations[2].open("x", encoding="utf-8") as handle:
        handle.write(split.manifest.model_dump_json(indent=2) + "\n")


def load_split_manifest(path: Union[str, Path]) -> DatasetSplitManifest:
    """Validate a saved partition manifest before using its protected IDs."""

    return DatasetSplitManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def _require_exact_ids(
    results: Sequence[CaseRunResult],
    expected_ids: Sequence[str],
    name: str,
) -> None:
    actual = {result.case_id for result in results}
    expected = set(expected_ids)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{name} result IDs do not match the split; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _require_same_human_labels(
    baseline_results: Sequence[CaseRunResult],
    candidate_results: Sequence[CaseRunResult],
) -> None:
    """Prevent a candidate run from looking better by changing the answer key."""

    baseline_by_id = {result.case_id: result for result in baseline_results}
    candidate_by_id = {result.case_id: result for result in candidate_results}
    changed = []
    for case_id in sorted(baseline_by_id):
        baseline = baseline_by_id[case_id]
        candidate = candidate_by_id[case_id]
        if (
            baseline.mode != candidate.mode
            or baseline.human_decision != candidate.human_decision
            or baseline.human_scores != candidate.human_scores
            or baseline.review_status != candidate.review_status
        ):
            changed.append(case_id)
    if changed:
        raise ValueError(
            f"Human labels or review metadata changed between runs: {changed}"
        )


def _metric_comparison(
    name: str,
    baseline: Optional[float],
    candidate: Optional[float],
    direction: MetricDirection,
    tolerance: float,
) -> MetricComparison:
    if baseline is None or candidate is None:
        return MetricComparison(
            metric=name,
            direction=direction,
            baseline=baseline,
            candidate=candidate,
            tolerated_regression=tolerance,
            outcome=MetricChange.UNAVAILABLE,
            explanation="Not compared because one or both measurements are unavailable.",
        )
    delta = candidate - baseline
    signed_improvement = (
        delta if direction == MetricDirection.HIGHER_IS_BETTER else -delta
    )
    if signed_improvement < -tolerance:
        outcome = MetricChange.REGRESSED
    elif signed_improvement > tolerance:
        outcome = MetricChange.IMPROVED
    else:
        outcome = MetricChange.UNCHANGED
    return MetricComparison(
        metric=name,
        direction=direction,
        baseline=round(baseline, 4),
        candidate=round(candidate, 4),
        delta=round(delta, 4),
        tolerated_regression=tolerance,
        outcome=outcome,
        explanation=(
            f"{outcome.value}: baseline={baseline:.4f}, candidate={candidate:.4f}, "
            f"delta={delta:+.4f}."
        ),
    )


def compare_calibration_runs(
    baseline_results: Sequence[CaseRunResult],
    candidate_results: Sequence[CaseRunResult],
    manifest: DatasetSplitManifest,
    *,
    baseline_configuration: str,
    candidate_configuration: str,
    change_summary: str,
    reviewed_by: Optional[str] = None,
    policy: CalibrationAcceptancePolicy = DEFAULT_CALIBRATION_POLICY,
    bootstrap_iterations: int = 2000,
) -> CalibrationComparisonReport:
    """Compare two configurations only on the reusable calibration partition."""

    if not baseline_configuration.strip() or not candidate_configuration.strip():
        raise ValueError("Both configuration versions are required")
    if not change_summary.strip():
        raise ValueError("A developer-readable change summary is required")
    _require_exact_ids(
        baseline_results, manifest.calibration_case_ids, "Baseline calibration"
    )
    _require_exact_ids(
        candidate_results, manifest.calibration_case_ids, "Candidate calibration"
    )
    _require_same_human_labels(baseline_results, candidate_results)
    baseline_reliability = calculate_reliability(baseline_results)
    candidate_reliability = calculate_reliability(candidate_results)
    baseline_errors = calculate_error_analysis(
        baseline_results,
        bootstrap_iterations=bootstrap_iterations,
    )
    candidate_errors = calculate_error_analysis(
        candidate_results,
        bootstrap_iterations=bootstrap_iterations,
    )
    baseline_aggregate = baseline_errors.analyses[AnalysisTarget.AGGREGATE]
    candidate_aggregate = candidate_errors.analyses[AnalysisTarget.AGGREGATE]

    specs: List[Tuple[str, Optional[float], Optional[float], MetricDirection, float]] = [
        (
            "aggregate_human_agreement",
            baseline_reliability.aggregate_vs_human.agreement_rate,
            candidate_reliability.aggregate_vs_human.agreement_rate,
            MetricDirection.HIGHER_IS_BETTER,
            policy.accuracy_regression_tolerance,
        ),
        (
            "aggregate_cohens_kappa",
            baseline_reliability.aggregate_vs_human.cohens_kappa,
            candidate_reliability.aggregate_vs_human.cohens_kappa,
            MetricDirection.HIGHER_IS_BETTER,
            policy.kappa_regression_tolerance,
        ),
        (
            "pass_fail_accuracy",
            baseline_aggregate.pass_fail_overall.accuracy.value,
            candidate_aggregate.pass_fail_overall.accuracy.value,
            MetricDirection.HIGHER_IS_BETTER,
            policy.accuracy_regression_tolerance,
        ),
        (
            "false_pass_rate",
            baseline_aggregate.pass_fail_overall.false_pass_rate.value,
            candidate_aggregate.pass_fail_overall.false_pass_rate.value,
            MetricDirection.LOWER_IS_BETTER,
            policy.false_pass_increase_tolerance,
        ),
        (
            "false_fail_rate",
            baseline_aggregate.pass_fail_overall.false_fail_rate.value,
            candidate_aggregate.pass_fail_overall.false_fail_rate.value,
            MetricDirection.LOWER_IS_BETTER,
            policy.false_fail_increase_tolerance,
        ),
        (
            "human_review_rate",
            baseline_reliability.human_review_rate,
            candidate_reliability.human_review_rate,
            MetricDirection.LOWER_IS_BETTER,
            policy.human_review_increase_tolerance,
        ),
        (
            "failure_rate",
            baseline_reliability.failure_rate,
            candidate_reliability.failure_rate,
            MetricDirection.LOWER_IS_BETTER,
            policy.failure_increase_tolerance,
        ),
        (
            "pairwise_accuracy",
            baseline_aggregate.pairwise.accuracy.value,
            candidate_aggregate.pairwise.accuracy.value,
            MetricDirection.HIGHER_IS_BETTER,
            policy.pairwise_accuracy_regression_tolerance,
        ),
        (
            "aggregate_score_correlation",
            baseline_reliability.score_correlation.aggregate_pearson,
            candidate_reliability.score_correlation.aggregate_pearson,
            MetricDirection.HIGHER_IS_BETTER,
            policy.score_correlation_regression_tolerance,
        ),
    ]
    comparisons = [_metric_comparison(*spec) for spec in specs]
    regressed = [
        item.metric for item in comparisons if item.outcome == MetricChange.REGRESSED
    ]
    improved = [
        item.metric for item in comparisons if item.outcome == MetricChange.IMPROVED
    ]
    if regressed:
        decision = CalibrationDecision.REJECTED
    elif not reviewed_by or not reviewed_by.strip():
        decision = CalibrationDecision.NEEDS_DEVELOPER_REVIEW
    else:
        decision = CalibrationDecision.ACCEPTED

    baseline_pass = baseline_aggregate.pass_fail_overall
    candidate_pass = candidate_aggregate.pass_fail_overall
    baseline_pair = baseline_aggregate.pairwise
    candidate_pair = candidate_aggregate.pairwise
    warnings = [
        "Calibration results may guide changes but are not final production evidence.",
        "Do not change human labels merely to improve candidate metrics.",
        "Run the accepted locked configuration once on the held-out partition.",
    ]
    if manifest.calibration_cases < 30:
        warnings.append(
            "Fewer than 30 calibration cases are available; metric changes may be unstable."
        )
    if manifest.contains_drafts:
        warnings.append("The split contains draft labels; results are learning-only.")
    return CalibrationComparisonReport(
        baseline_configuration=baseline_configuration,
        candidate_configuration=candidate_configuration,
        change_summary=change_summary,
        reviewed_by=reviewed_by,
        split_fingerprint=manifest.source_fingerprint,
        calibration_cases=manifest.calibration_cases,
        decision=decision,
        comparisons=comparisons,
        regressed_metrics=regressed,
        improved_metrics=improved,
        fixed_false_pass_case_ids=sorted(
            set(baseline_pass.false_pass_case_ids)
            - set(candidate_pass.false_pass_case_ids)
        ),
        new_false_pass_case_ids=sorted(
            set(candidate_pass.false_pass_case_ids)
            - set(baseline_pass.false_pass_case_ids)
        ),
        fixed_false_fail_case_ids=sorted(
            set(baseline_pass.false_fail_case_ids)
            - set(candidate_pass.false_fail_case_ids)
        ),
        new_false_fail_case_ids=sorted(
            set(candidate_pass.false_fail_case_ids)
            - set(baseline_pass.false_fail_case_ids)
        ),
        fixed_pairwise_mismatch_case_ids=sorted(
            set(baseline_pair.mismatch_case_ids)
            - set(candidate_pair.mismatch_case_ids)
        ),
        new_pairwise_mismatch_case_ids=sorted(
            set(candidate_pair.mismatch_case_ids)
            - set(baseline_pair.mismatch_case_ids)
        ),
        warnings=warnings,
    )


def verify_heldout_run(
    results: Sequence[CaseRunResult],
    manifest: DatasetSplitManifest,
    comparison: CalibrationComparisonReport,
    *,
    configuration_version: str,
    bootstrap_iterations: int = 2000,
) -> HeldoutVerificationReport:
    """Calculate final evidence only when results match protected held-out IDs."""

    if not configuration_version.strip():
        raise ValueError("A locked configuration version is required")
    if comparison.decision != CalibrationDecision.ACCEPTED:
        raise ValueError("Held-out verification requires an accepted calibration comparison")
    if comparison.candidate_configuration != configuration_version:
        raise ValueError(
            "Held-out configuration does not match the accepted calibration candidate"
        )
    if comparison.split_fingerprint != manifest.source_fingerprint:
        raise ValueError("Calibration comparison belongs to a different dataset split")
    if not comparison.reviewed_by:
        raise ValueError("Accepted calibration comparison is missing its developer reviewer")
    _require_exact_ids(results, manifest.heldout_case_ids, "Held-out")
    warnings = [
        "Do not tune the prompt, rubric, examples, or thresholds from held-out errors.",
        "Use the production gate to decide whether this locked configuration is releasable.",
    ]
    if manifest.heldout_cases < 30:
        warnings.append(
            "Fewer than 30 held-out cases are available; final estimates may be unstable."
        )
    if manifest.contains_drafts:
        warnings.append("The split contains draft labels; verification is learning-only.")
    return HeldoutVerificationReport(
        configuration_version=configuration_version,
        calibration_reviewed_by=comparison.reviewed_by,
        split_fingerprint=manifest.source_fingerprint,
        heldout_cases=manifest.heldout_cases,
        reliability=calculate_reliability(results),
        error_analysis=calculate_error_analysis(
            results,
            bootstrap_iterations=bootstrap_iterations,
        ),
        warnings=warnings,
    )


def _write_report(path: Union[str, Path], report: BaseModel, overwrite: bool) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w" if overwrite else "x", encoding="utf-8") as handle:
        handle.write(report.model_dump_json(indent=2) + "\n")


def main() -> None:
    """Command-line entry point for split, compare, and verify operations."""

    parser = argparse.ArgumentParser(
        description="Calibrate an LLM judge without contaminating final test data"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="Create protected partitions")
    split_parser.add_argument("--dataset", required=True)
    split_parser.add_argument("--calibration-output", required=True)
    split_parser.add_argument("--heldout-output", required=True)
    split_parser.add_argument("--manifest-output", required=True)
    split_parser.add_argument("--calibration-fraction", type=float, default=0.70)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--allow-drafts", action="store_true")
    split_parser.add_argument("--overwrite", action="store_true")

    compare_parser = subparsers.add_parser("compare", help="Compare calibration runs")
    compare_parser.add_argument("--baseline-results", required=True)
    compare_parser.add_argument("--candidate-results", required=True)
    compare_parser.add_argument("--manifest", required=True)
    compare_parser.add_argument("--baseline-version", required=True)
    compare_parser.add_argument("--candidate-version", required=True)
    compare_parser.add_argument("--change-summary", required=True)
    compare_parser.add_argument("--reviewed-by")
    compare_parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--overwrite", action="store_true")

    verify_parser = subparsers.add_parser("verify", help="Verify locked held-out run")
    verify_parser.add_argument("--results", required=True)
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--comparison", required=True)
    verify_parser.add_argument("--configuration-version", required=True)
    verify_parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    verify_parser.add_argument("--output", required=True)
    verify_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.command == "split":
        split = split_evaluation_dataset(
            load_jsonl(args.dataset),
            calibration_fraction=args.calibration_fraction,
            random_seed=args.seed,
            allow_drafts=args.allow_drafts,
        )
        save_dataset_split(
            split,
            calibration_path=args.calibration_output,
            heldout_path=args.heldout_output,
            manifest_path=args.manifest_output,
            overwrite=args.overwrite,
        )
        print(split.manifest.model_dump_json(indent=2))
        return

    manifest = load_split_manifest(args.manifest)
    if args.command == "compare":
        report = compare_calibration_runs(
            load_case_results(args.baseline_results),
            load_case_results(args.candidate_results),
            manifest,
            baseline_configuration=args.baseline_version,
            candidate_configuration=args.candidate_version,
            change_summary=args.change_summary,
            reviewed_by=args.reviewed_by,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    else:
        report = verify_heldout_run(
            load_case_results(args.results),
            manifest,
            CalibrationComparisonReport.model_validate_json(
                Path(args.comparison).read_text(encoding="utf-8")
            ),
            configuration_version=args.configuration_version,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    _write_report(args.output, report, args.overwrite)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
