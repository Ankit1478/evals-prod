"""Step 9: measure judge reliability against human-approved labels."""

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from .contracts import Criterion, EvaluationMode, EvaluationResult
from .dataset import ReviewStatus
from .dataset_runner import CaseRunResult, CaseRunStatus
from .multi_judge import JudgeModel


class AgreementMetrics(BaseModel):
    """Exact categorical agreement and chance-corrected agreement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    available_cases: int = Field(ge=0)
    evaluated_cases: int = Field(ge=0)
    abstentions: int = Field(ge=0)
    exact_matches: int = Field(ge=0)
    agreement_rate: Optional[float] = Field(default=None, ge=0, le=1)
    cohens_kappa: Optional[float] = Field(default=None, ge=-1, le=1)


class ScoreCorrelationMetrics(BaseModel):
    """Pearson correlation across human and judge rubric-dimension scores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score_cases: int = Field(ge=0)
    score_pairs: int = Field(ge=0)
    terra_pearson: Optional[float] = Field(default=None, ge=-1, le=1)
    luna_pearson: Optional[float] = Field(default=None, ge=-1, le=1)
    aggregate_pearson: Optional[float] = Field(default=None, ge=-1, le=1)


class ModeReliabilityMetrics(BaseModel):
    """Agreement metrics isolated to one evaluation mode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_cases: int = Field(ge=1)
    completed_cases: int = Field(ge=0)
    terra_vs_human: AgreementMetrics
    luna_vs_human: AgreementMetrics
    aggregate_vs_human: AgreementMetrics
    terra_vs_luna: AgreementMetrics


class ReliabilityReport(BaseModel):
    """Metrics plus warnings needed to interpret them responsibly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_cases: int = Field(ge=1)
    completed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    human_review_required: int = Field(ge=0)
    human_review_rate: float = Field(ge=0, le=1)
    model_disagreements: int = Field(ge=0)
    model_disagreement_rate: Optional[float] = Field(default=None, ge=0, le=1)
    terra_vs_human: AgreementMetrics
    luna_vs_human: AgreementMetrics
    aggregate_vs_human: AgreementMetrics
    terra_vs_luna: AgreementMetrics
    by_mode: Dict[EvaluationMode, ModeReliabilityMetrics]
    score_correlation: ScoreCorrelationMetrics
    warnings: List[str]


def cohens_kappa(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
) -> Optional[float]:
    """Return Cohen's Kappa, or None when it is mathematically undefined."""

    if len(labels_a) != len(labels_b):
        raise ValueError("Cohen's Kappa label lists must have equal length")
    if not labels_a:
        return None

    total = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / total
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    labels = set(counts_a) | set(counts_b)
    expected = sum(
        (counts_a[label] / total) * (counts_b[label] / total)
        for label in labels
    )
    denominator = 1 - expected
    if math.isclose(denominator, 0.0):
        # When both raters always use one label, chance agreement is already 1.
        return None
    return round((observed - expected) / denominator, 4)


def pearson_correlation(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> Optional[float]:
    """Return Pearson r, or None for insufficient or constant observations."""

    if len(values_a) != len(values_b):
        raise ValueError("Correlation value lists must have equal length")
    if len(values_a) < 2:
        return None

    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    denominator = math.sqrt(
        sum(value * value for value in centered_a)
        * sum(value * value for value in centered_b)
    )
    if math.isclose(denominator, 0.0):
        return None
    numerator = sum(a * b for a, b in zip(centered_a, centered_b))
    # Floating-point arithmetic can produce values microscopically outside [-1, 1].
    return round(max(-1.0, min(1.0, numerator / denominator)), 4)


def load_case_results(path: Union[str, Path]) -> List[CaseRunResult]:
    """Load and validate the dataset runner's one-result-per-line JSONL."""

    source = Path(path)
    results = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                results.append(CaseRunResult.model_validate_json(line))
            except ValueError as error:
                raise ValueError(
                    f"Invalid runner result at {source}:{line_number}"
                ) from error
    if not results:
        raise ValueError(f"Runner result file is empty: {source}")
    case_ids = [result.case_id for result in results]
    duplicates = sorted(
        case_id for case_id, count in Counter(case_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate case IDs in runner results: {duplicates}")
    return results


def _model_decision(result: CaseRunResult, model: JudgeModel) -> str:
    judgment = next(
        item for item in result.judge_result.judgments if item.model == model
    )
    return judgment.result.decision.value


def _agreement(
    pairs: Iterable[Tuple[str, Optional[str]]],
    *,
    available_cases: int,
) -> AgreementMetrics:
    comparable = [(left, right) for left, right in pairs if right is not None]
    labels_a = [left for left, _ in comparable]
    labels_b = [right for _, right in comparable]
    matches = sum(left == right for left, right in comparable)
    evaluated = len(comparable)
    return AgreementMetrics(
        available_cases=available_cases,
        evaluated_cases=evaluated,
        abstentions=available_cases - evaluated,
        exact_matches=matches,
        agreement_rate=round(matches / evaluated, 4) if evaluated else None,
        cohens_kappa=cohens_kappa(labels_a, labels_b),
    )


def _score_correlations(
    completed: Sequence[CaseRunResult],
) -> ScoreCorrelationMetrics:
    human_values: List[float] = []
    terra_values: List[float] = []
    luna_values: List[float] = []
    aggregate_values: List[float] = []
    score_cases = 0

    for case in completed:
        if case.mode != EvaluationMode.SCORE or not case.human_scores:
            continue
        score_cases += 1
        judgments = {
            item.model: item.result for item in case.judge_result.judgments
        }
        terra_result = judgments[JudgeModel.TERRA]
        luna_result = judgments[JudgeModel.LUNA]
        if not isinstance(terra_result, EvaluationResult) or not isinstance(
            luna_result, EvaluationResult
        ):
            raise ValueError("Score case contains a non-score model result")
        terra_scores = {item.criterion: item.score for item in terra_result.scores}
        luna_scores = {item.criterion: item.score for item in luna_result.scores}
        averages = case.judge_result.average_scores
        if averages is None:
            raise ValueError("Score case is missing aggregate average scores")

        for criterion in Criterion:
            human_values.append(float(case.human_scores[criterion]))
            terra_values.append(float(terra_scores[criterion]))
            luna_values.append(float(luna_scores[criterion]))
            aggregate_values.append(float(averages[criterion]))

    return ScoreCorrelationMetrics(
        score_cases=score_cases,
        score_pairs=len(human_values),
        terra_pearson=pearson_correlation(human_values, terra_values),
        luna_pearson=pearson_correlation(human_values, luna_values),
        aggregate_pearson=pearson_correlation(human_values, aggregate_values),
    )


def _agreement_group(
    completed: Sequence[CaseRunResult],
) -> Dict[str, AgreementMetrics]:
    """Calculate the four agreement views for one group of completed cases."""

    available = len(completed)
    return {
        "terra_vs_human": _agreement(
            (
                (result.human_decision, _model_decision(result, JudgeModel.TERRA))
                for result in completed
            ),
            available_cases=available,
        ),
        "luna_vs_human": _agreement(
            (
                (result.human_decision, _model_decision(result, JudgeModel.LUNA))
                for result in completed
            ),
            available_cases=available,
        ),
        "aggregate_vs_human": _agreement(
            (
                (result.human_decision, result.judge_result.aggregate_decision)
                for result in completed
            ),
            available_cases=available,
        ),
        "terra_vs_luna": _agreement(
            (
                (
                    _model_decision(result, JudgeModel.TERRA),
                    _model_decision(result, JudgeModel.LUNA),
                )
                for result in completed
            ),
            available_cases=available,
        ),
    }


def calculate_reliability(
    results: Sequence[CaseRunResult],
) -> ReliabilityReport:
    """Calculate reliability metrics from validated dataset-runner results."""

    if not results:
        raise ValueError("At least one case result is required")
    case_ids = [result.case_id for result in results]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Reliability input contains duplicate case IDs")

    completed = [
        result for result in results if result.status == CaseRunStatus.COMPLETED
    ]
    total = len(results)
    available = len(completed)
    overall_agreement = _agreement_group(completed)
    terra_vs_human = overall_agreement["terra_vs_human"]
    luna_vs_human = overall_agreement["luna_vs_human"]
    aggregate_vs_human = overall_agreement["aggregate_vs_human"]
    terra_vs_luna = overall_agreement["terra_vs_luna"]
    by_mode = {}
    for mode in EvaluationMode:
        all_for_mode = [result for result in results if result.mode == mode]
        if not all_for_mode:
            continue
        completed_for_mode = [
            result
            for result in all_for_mode
            if result.status == CaseRunStatus.COMPLETED
        ]
        mode_agreement = _agreement_group(completed_for_mode)
        by_mode[mode] = ModeReliabilityMetrics(
            total_cases=len(all_for_mode),
            completed_cases=len(completed_for_mode),
            **mode_agreement,
        )
    score_correlation = _score_correlations(completed)

    warnings = []
    if any(result.review_status == ReviewStatus.DRAFT for result in results):
        warnings.append(
            "Draft human labels are present; metrics are learning-only and cannot approve production."
        )
    if available < 30:
        warnings.append(
            "Fewer than 30 completed cases are available; reliability estimates are unstable."
        )
    named_agreements = {
        "Terra vs human": terra_vs_human,
        "Luna vs human": luna_vs_human,
        "Aggregate vs human": aggregate_vs_human,
        "Terra vs Luna": terra_vs_luna,
    }
    for name, metric in named_agreements.items():
        if metric.evaluated_cases and metric.cohens_kappa is None:
            warnings.append(
                f"{name} Cohen's Kappa is undefined because label variation is insufficient."
            )
    if score_correlation.score_cases and score_correlation.aggregate_pearson is None:
        warnings.append(
            "Score correlation is undefined because score variation is insufficient."
        )

    failed = total - available
    review_count = sum(result.requires_human_review for result in results)
    disagreement_count = sum(
        result.judge_result is not None and not result.judge_result.agreement
        for result in completed
    )
    return ReliabilityReport(
        total_cases=total,
        completed_cases=available,
        failed_cases=failed,
        failure_rate=round(failed / total, 4),
        human_review_required=review_count,
        human_review_rate=round(review_count / total, 4),
        model_disagreements=disagreement_count,
        model_disagreement_rate=(
            round(disagreement_count / available, 4) if available else None
        ),
        terra_vs_human=terra_vs_human,
        luna_vs_human=luna_vs_human,
        aggregate_vs_human=aggregate_vs_human,
        terra_vs_luna=terra_vs_luna,
        by_mode=by_mode,
        score_correlation=score_correlation,
        warnings=warnings,
    )


def main() -> None:
    """Command-line entry point for calculating a reliability report."""

    parser = argparse.ArgumentParser(
        description="Calculate reliability metrics from dataset-runner JSONL"
    )
    parser.add_argument("--input", required=True, help="Runner JSONL result file")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing report file",
    )
    args = parser.parse_args()

    report = calculate_reliability(load_case_results(args.input))
    rendered = report.model_dump_json(indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open(
            "w" if args.overwrite else "x",
            encoding="utf-8",
        ) as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
