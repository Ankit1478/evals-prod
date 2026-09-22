"""Step 11: classify judge errors and quantify statistical uncertainty."""

import argparse
import math
import random
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .contracts import Decision, EvaluationMode, PairwiseDecision
from .dataset import ReviewStatus
from .dataset_runner import CaseRunResult, CaseRunStatus
from .multi_judge import JudgeModel
from .reliability import load_case_results


class AnalysisTarget(str, Enum):
    """The individual or combined judge result being compared with humans."""

    TERRA = "terra"
    LUNA = "luna"
    AGGREGATE = "aggregate"


class MetricEstimate(BaseModel):
    """One metric, its denominator, and a percentile-bootstrap interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: Optional[float] = Field(default=None, ge=0, le=1)
    ci_lower: Optional[float] = Field(default=None, ge=0, le=1)
    ci_upper: Optional[float] = Field(default=None, ge=0, le=1)
    denominator: int = Field(ge=0)
    denominator_description: str
    bootstrap_samples: int = Field(ge=0)


class BinaryConfusionMatrix(BaseModel):
    """Counts where PASS is the positive class and FAIL is the negative class."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    true_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)


class BinaryClassificationAnalysis(BaseModel):
    """PASS/FAIL errors for one target and one group of evaluation modes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    available_cases: int = Field(ge=0)
    evaluated_cases: int = Field(ge=0)
    abstentions: int = Field(ge=0)
    confusion_matrix: BinaryConfusionMatrix
    accuracy: MetricEstimate
    precision: MetricEstimate
    recall: MetricEstimate
    f1: MetricEstimate
    false_pass_rate: MetricEstimate
    false_fail_rate: MetricEstimate
    false_pass_case_ids: List[str]
    false_fail_case_ids: List[str]


class PairwiseLabelAnalysis(BaseModel):
    """One pairwise label's one-vs-rest precision, recall, and F1."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    support: int = Field(ge=0)
    predicted: int = Field(ge=0)
    precision: MetricEstimate
    recall: MetricEstimate
    f1: MetricEstimate


class PairwiseClassificationAnalysis(BaseModel):
    """Three-class pairwise error analysis for one judge target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    available_cases: int = Field(ge=0)
    evaluated_cases: int = Field(ge=0)
    abstentions: int = Field(ge=0)
    # Rows are human labels; columns are predictions.
    confusion_matrix: Dict[str, Dict[str, int]]
    accuracy: MetricEstimate
    macro_f1: MetricEstimate
    by_label: Dict[str, PairwiseLabelAnalysis]
    mismatch_case_ids: List[str]


class JudgeErrorAnalysis(BaseModel):
    """All Step 11 views for Terra, Luna, or their aggregate decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: AnalysisTarget
    pass_fail_overall: BinaryClassificationAnalysis
    pass_fail_by_mode: Dict[EvaluationMode, BinaryClassificationAnalysis]
    pairwise: PairwiseClassificationAnalysis


class ErrorAnalysisReport(BaseModel):
    """Auditable error counts and uncertainty estimates for every judge target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_cases: int = Field(ge=1)
    completed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_iterations: int = Field(ge=1)
    random_seed: int
    analyses: Dict[AnalysisTarget, JudgeErrorAnalysis]
    warnings: List[str]


BinaryPair = Tuple[str, str, str]
PairwisePair = Tuple[str, str, str]


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Calculate an interpolated percentile without an extra dependency."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_values(
    pairs: Sequence[Tuple[str, str, str]],
    metric: Callable[[Sequence[Tuple[str, str, str]]], Optional[float]],
    *,
    iterations: int,
    confidence_level: float,
    rng: random.Random,
) -> Tuple[Optional[float], Optional[float], int]:
    if not pairs:
        return None, None, 0
    values = []
    for _ in range(iterations):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        value = metric(sample)
        # Some bootstrap samples contain only one class, making a rate undefined.
        if value is not None:
            values.append(value)
    if not values:
        return None, None, 0
    tail = (1 - confidence_level) / 2
    return (
        round(_percentile(values, tail), 4),
        round(_percentile(values, 1 - tail), 4),
        len(values),
    )


def _estimate(
    value: Optional[float],
    denominator: int,
    denominator_description: str,
    pairs: Sequence[Tuple[str, str, str]],
    metric: Callable[[Sequence[Tuple[str, str, str]]], Optional[float]],
    *,
    iterations: int,
    confidence_level: float,
    rng: random.Random,
) -> MetricEstimate:
    if value is None:
        return MetricEstimate(
            denominator=denominator,
            denominator_description=denominator_description,
            bootstrap_samples=0,
        )
    lower, upper, samples = _bootstrap_values(
        pairs,
        metric,
        iterations=iterations,
        confidence_level=confidence_level,
        rng=rng,
    )
    return MetricEstimate(
        value=round(value, 4),
        ci_lower=lower,
        ci_upper=upper,
        denominator=denominator,
        denominator_description=denominator_description,
        bootstrap_samples=samples,
    )


def _binary_counts(pairs: Sequence[BinaryPair]) -> BinaryConfusionMatrix:
    return BinaryConfusionMatrix(
        true_positive=sum(h == Decision.PASS.value and p == h for _, h, p in pairs),
        true_negative=sum(h == Decision.FAIL.value and p == h for _, h, p in pairs),
        false_positive=sum(
            h == Decision.FAIL.value and p == Decision.PASS.value
            for _, h, p in pairs
        ),
        false_negative=sum(
            h == Decision.PASS.value and p == Decision.FAIL.value
            for _, h, p in pairs
        ),
    )


def _binary_metric(name: str) -> Callable[[Sequence[BinaryPair]], Optional[float]]:
    def calculate(pairs: Sequence[BinaryPair]) -> Optional[float]:
        counts = _binary_counts(pairs)
        tp = counts.true_positive
        tn = counts.true_negative
        fp = counts.false_positive
        fn = counts.false_negative
        if name == "accuracy":
            return _safe_ratio(tp + tn, tp + tn + fp + fn)
        if name == "precision":
            return _safe_ratio(tp, tp + fp)
        if name == "recall":
            return _safe_ratio(tp, tp + fn)
        if name == "f1":
            return _safe_ratio(2 * tp, 2 * tp + fp + fn)
        if name == "false_pass_rate":
            return _safe_ratio(fp, fp + tn)
        if name == "false_fail_rate":
            return _safe_ratio(fn, fn + tp)
        raise ValueError(f"Unknown binary metric: {name}")

    return calculate


def _prediction(result: CaseRunResult, target: AnalysisTarget) -> Optional[str]:
    if target == AnalysisTarget.AGGREGATE:
        return result.judge_result.aggregate_decision
    wanted = JudgeModel.TERRA if target == AnalysisTarget.TERRA else JudgeModel.LUNA
    judgment = next(item for item in result.judge_result.judgments if item.model == wanted)
    return judgment.result.decision.value


def _binary_analysis(
    cases: Sequence[CaseRunResult],
    target: AnalysisTarget,
    *,
    iterations: int,
    confidence_level: float,
    rng: random.Random,
) -> BinaryClassificationAnalysis:
    pairs = []
    for case in cases:
        prediction = _prediction(case, target)
        if prediction is not None:
            pairs.append((case.case_id, case.human_decision, prediction))
    counts = _binary_counts(pairs)
    tp, tn = counts.true_positive, counts.true_negative
    fp, fn = counts.false_positive, counts.false_negative
    definitions = {
        "accuracy": (tp + tn + fp + fn, "all evaluated PASS/FAIL cases"),
        "precision": (tp + fp, "cases predicted PASS"),
        "recall": (tp + fn, "human PASS cases"),
        "f1": (len(pairs), "all evaluated PASS/FAIL cases"),
        "false_pass_rate": (fp + tn, "human FAIL cases"),
        "false_fail_rate": (fn + tp, "human PASS cases"),
    }
    estimates = {}
    for name, (denominator, description) in definitions.items():
        metric = _binary_metric(name)
        estimates[name] = _estimate(
            metric(pairs),
            denominator,
            description,
            pairs,
            metric,
            iterations=iterations,
            confidence_level=confidence_level,
            rng=rng,
        )
    return BinaryClassificationAnalysis(
        available_cases=len(cases),
        evaluated_cases=len(pairs),
        abstentions=len(cases) - len(pairs),
        confusion_matrix=counts,
        false_pass_case_ids=[
            case_id
            for case_id, human, prediction in pairs
            if human == Decision.FAIL.value and prediction == Decision.PASS.value
        ],
        false_fail_case_ids=[
            case_id
            for case_id, human, prediction in pairs
            if human == Decision.PASS.value and prediction == Decision.FAIL.value
        ],
        **estimates,
    )


def _pairwise_label_metric(
    label: str, name: str
) -> Callable[[Sequence[PairwisePair]], Optional[float]]:
    def calculate(pairs: Sequence[PairwisePair]) -> Optional[float]:
        tp = sum(h == label and p == label for _, h, p in pairs)
        fp = sum(h != label and p == label for _, h, p in pairs)
        fn = sum(h == label and p != label for _, h, p in pairs)
        if name == "precision":
            return _safe_ratio(tp, tp + fp)
        if name == "recall":
            return _safe_ratio(tp, tp + fn)
        if name == "f1":
            return _safe_ratio(2 * tp, 2 * tp + fp + fn)
        raise ValueError(f"Unknown pairwise label metric: {name}")

    return calculate


def _pairwise_accuracy(pairs: Sequence[PairwisePair]) -> Optional[float]:
    return _safe_ratio(sum(human == prediction for _, human, prediction in pairs), len(pairs))


def _pairwise_macro_f1(pairs: Sequence[PairwisePair]) -> Optional[float]:
    values = [
        _pairwise_label_metric(label.value, "f1")(pairs) for label in PairwiseDecision
    ]
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _pairwise_analysis(
    cases: Sequence[CaseRunResult],
    target: AnalysisTarget,
    *,
    iterations: int,
    confidence_level: float,
    rng: random.Random,
) -> PairwiseClassificationAnalysis:
    labels = [item.value for item in PairwiseDecision]
    pairs = []
    for case in cases:
        prediction = _prediction(case, target)
        if prediction is not None:
            pairs.append((case.case_id, case.human_decision, prediction))
    matrix = {
        human: {
            prediction: sum(h == human and p == prediction for _, h, p in pairs)
            for prediction in labels
        }
        for human in labels
    }
    by_label = {}
    for label in labels:
        support = sum(human == label for _, human, _ in pairs)
        predicted = sum(prediction == label for _, _, prediction in pairs)
        metrics = {}
        for name, denominator, description in (
            ("precision", predicted, f"cases predicted {label}"),
            ("recall", support, f"human {label} cases"),
            ("f1", len(pairs), "all evaluated pairwise cases"),
        ):
            metric = _pairwise_label_metric(label, name)
            metrics[name] = _estimate(
                metric(pairs),
                denominator,
                description,
                pairs,
                metric,
                iterations=iterations,
                confidence_level=confidence_level,
                rng=rng,
            )
        by_label[label] = PairwiseLabelAnalysis(
            support=support,
            predicted=predicted,
            **metrics,
        )
    return PairwiseClassificationAnalysis(
        available_cases=len(cases),
        evaluated_cases=len(pairs),
        abstentions=len(cases) - len(pairs),
        confusion_matrix=matrix,
        accuracy=_estimate(
            _pairwise_accuracy(pairs),
            len(pairs),
            "all evaluated pairwise cases",
            pairs,
            _pairwise_accuracy,
            iterations=iterations,
            confidence_level=confidence_level,
            rng=rng,
        ),
        macro_f1=_estimate(
            _pairwise_macro_f1(pairs),
            len(pairs),
            "all evaluated pairwise cases",
            pairs,
            _pairwise_macro_f1,
            iterations=iterations,
            confidence_level=confidence_level,
            rng=rng,
        ),
        by_label=by_label,
        mismatch_case_ids=[
            case_id for case_id, human, prediction in pairs if human != prediction
        ],
    )


def calculate_error_analysis(
    results: Sequence[CaseRunResult],
    *,
    bootstrap_iterations: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> ErrorAnalysisReport:
    """Compare all judge targets with humans and attach bootstrap intervals."""

    if not results:
        raise ValueError("At least one case result is required")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    case_ids = [result.case_id for result in results]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Error-analysis input contains duplicate case IDs")

    completed = [
        result for result in results if result.status == CaseRunStatus.COMPLETED
    ]
    pass_fail = [
        result
        for result in completed
        if result.mode in (EvaluationMode.BINARY, EvaluationMode.SCORE)
    ]
    pairwise = [result for result in completed if result.mode == EvaluationMode.PAIRWISE]
    rng = random.Random(random_seed)
    analyses = {}
    for target in AnalysisTarget:
        analyses[target] = JudgeErrorAnalysis(
            target=target,
            pass_fail_overall=_binary_analysis(
                pass_fail,
                target,
                iterations=bootstrap_iterations,
                confidence_level=confidence_level,
                rng=rng,
            ),
            pass_fail_by_mode={
                mode: _binary_analysis(
                    [result for result in pass_fail if result.mode == mode],
                    target,
                    iterations=bootstrap_iterations,
                    confidence_level=confidence_level,
                    rng=rng,
                )
                for mode in (EvaluationMode.BINARY, EvaluationMode.SCORE)
            },
            pairwise=_pairwise_analysis(
                pairwise,
                target,
                iterations=bootstrap_iterations,
                confidence_level=confidence_level,
                rng=rng,
            ),
        )

    warnings = []
    if any(result.review_status == ReviewStatus.DRAFT for result in results):
        warnings.append(
            "Draft human labels are present; this error analysis is learning-only."
        )
    if len(completed) < 30:
        warnings.append(
            "Fewer than 30 completed cases are available; estimates and confidence intervals may be unstable."
        )
    if not pass_fail:
        warnings.append("No completed binary or score cases are available.")
    else:
        human_labels = {result.human_decision for result in pass_fail}
        if human_labels != {Decision.PASS.value, Decision.FAIL.value}:
            warnings.append(
                "PASS/FAIL class coverage is incomplete; some classification rates are undefined."
            )
    pairwise_labels = {result.human_decision for result in pairwise}
    missing_pairwise = [
        label.value for label in PairwiseDecision if label.value not in pairwise_labels
    ]
    if missing_pairwise:
        warnings.append(
            "Pairwise human-label coverage is incomplete; missing: "
            + ", ".join(missing_pairwise)
            + "."
        )
    warnings.append(
        "Bootstrap intervals describe sampling uncertainty only; they do not by themselves approve a judge for production."
    )
    return ErrorAnalysisReport(
        total_cases=len(results),
        completed_cases=len(completed),
        failed_cases=len(results) - len(completed),
        confidence_level=confidence_level,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
        analyses=analyses,
        warnings=warnings,
    )


def main() -> None:
    """Command-line entry point for local Step 11 analysis."""

    parser = argparse.ArgumentParser(
        description="Calculate error analysis and bootstrap confidence intervals"
    )
    parser.add_argument("--input", required=True, help="Runner JSONL result file")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing report file"
    )
    args = parser.parse_args()

    report = calculate_error_analysis(
        load_case_results(args.input),
        bootstrap_iterations=args.bootstrap_iterations,
        confidence_level=args.confidence_level,
        random_seed=args.seed,
    )
    rendered = report.model_dump_json(indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
