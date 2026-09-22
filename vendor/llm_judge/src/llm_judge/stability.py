"""Step 10: measure repeat consistency and pairwise position sensitivity."""

import argparse
from collections import Counter
from enum import Enum
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Optional, Protocol, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import (
    Criterion,
    EvaluationInput,
    EvaluationMode,
    EvaluationResult,
    PairwiseDecision,
)
from .dataset import EvaluationCase, EvaluationDataset, ReviewStatus, load_jsonl
from .multi_judge import JudgeModel, ModelJudgment, TwoModelJudge
from .prompt_builder import JudgePrompt, build_judge_prompt
from .settings import AzureJudgeSettings


class SingleModelEvaluator(Protocol):
    """Interface shared by the real two-model judge and offline test fakes."""

    def evaluate_prompt(
        self,
        model: JudgeModel,
        prompt: JudgePrompt,
        evaluation_input: EvaluationInput,
    ) -> ModelJudgment:
        ...


class EvaluationOrder(str, Enum):
    """Whether candidates use their dataset order or the reversed order."""

    ORIGINAL = "original"
    SWAPPED = "swapped"


class ObservationStatus(str, Enum):
    """Whether one repeat produced a trusted judgment."""

    COMPLETED = "completed"
    ERROR = "error"


class StabilityObservation(BaseModel):
    """One auditable model call within a repeated stability experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_number: int = Field(ge=1)
    order: EvaluationOrder
    status: ObservationStatus
    judgment: Optional[ModelJudgment] = None
    raw_decision: Optional[str] = None
    # Swapped decisions are mapped back to the original candidate identities.
    canonical_decision: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    @model_validator(mode="after")
    def validate_status(self) -> "StabilityObservation":
        if self.status == ObservationStatus.COMPLETED:
            if (
                self.judgment is None
                or self.raw_decision is None
                or self.canonical_decision is None
                or self.error_type
                or self.error_message
            ):
                raise ValueError("Completed observation requires a judgment and decisions")
        elif self.judgment is not None or not self.error_type:
            raise ValueError("Failed observation requires an error and no judgment")
        return self


class CriterionVariation(BaseModel):
    """Distribution of one rubric score across successful repeated runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    criterion: Criterion
    observations: int = Field(ge=1)
    mean: float
    median: float
    population_stddev: float = Field(ge=0)
    minimum: int = Field(ge=1, le=5)
    maximum: int = Field(ge=1, le=5)
    score_range: int = Field(ge=0, le=4)


class StabilityCaseResult(BaseModel):
    """Repeat and optional position test results for one case and one model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    mode: EvaluationMode
    review_status: ReviewStatus
    model: JudgeModel
    repeat_count: int = Field(ge=2)
    original_observations: List[StabilityObservation] = Field(min_length=2)
    swapped_observations: List[StabilityObservation] = Field(default_factory=list)
    successful_original_runs: int = Field(ge=0)
    failed_original_runs: int = Field(ge=0)
    modal_original_decision: Optional[str] = None
    modal_decision_tied: bool
    repeat_consistency: Optional[float] = Field(default=None, ge=0, le=1)
    all_original_decisions_same: Optional[bool] = None
    score_variation: List[CriterionVariation] = Field(default_factory=list)
    position_comparisons: int = Field(ge=0)
    position_mismatches: int = Field(ge=0)
    position_flip_rate: Optional[float] = Field(default=None, ge=0, le=1)
    first_position_preference_pairs: int = Field(ge=0)
    second_position_preference_pairs: int = Field(ge=0)
    tie_relevant_pairs: int = Field(ge=0)
    tie_consistent_pairs: int = Field(ge=0)
    tie_consistency_rate: Optional[float] = Field(default=None, ge=0, le=1)
    requires_investigation: bool


class ModelStabilitySummary(BaseModel):
    """Aggregate stability measurements for one judge model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: JudgeModel
    evaluated_cases: int = Field(ge=1)
    total_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    repeat_comparable_cases: int = Field(ge=0)
    stable_cases: int = Field(ge=0)
    unstable_cases: int = Field(ge=0)
    insufficient_repeat_cases: int = Field(ge=0)
    mean_repeat_consistency: Optional[float] = Field(default=None, ge=0, le=1)
    median_repeat_consistency: Optional[float] = Field(default=None, ge=0, le=1)
    position_cases: int = Field(ge=0)
    position_comparisons: int = Field(ge=0)
    position_mismatches: int = Field(ge=0)
    position_flip_rate: Optional[float] = Field(default=None, ge=0, le=1)
    first_position_preference_pairs: int = Field(ge=0)
    second_position_preference_pairs: int = Field(ge=0)
    tie_relevant_pairs: int = Field(ge=0)
    tie_consistency_rate: Optional[float] = Field(default=None, ge=0, le=1)
    unstable_case_ids: List[str]
    position_biased_case_ids: List[str]
    insufficient_repeat_case_ids: List[str]


class StabilityReport(BaseModel):
    """Complete Step 10 report for Terra and Luna."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_cases: int = Field(ge=1)
    repeat_count: int = Field(ge=2)
    total_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    per_model: Dict[JudgeModel, ModelStabilitySummary]
    results: List[StabilityCaseResult]
    warnings: List[str]


def normalize_swapped_decision(decision: PairwiseDecision) -> PairwiseDecision:
    """Map a swapped-order winner back to the original candidate identities."""

    if decision == PairwiseDecision.A_WINS:
        return PairwiseDecision.B_WINS
    if decision == PairwiseDecision.B_WINS:
        return PairwiseDecision.A_WINS
    return PairwiseDecision.TIE


def make_swapped_input(case: EvaluationCase) -> EvaluationCase:
    """Create a pairwise case with A/B reversed and a distinct request case ID."""

    if case.mode != EvaluationMode.PAIRWISE or case.candidate_b is None:
        raise ValueError("Only pairwise cases can be swapped")
    return case.model_copy(
        update={
            "case_id": f"{case.case_id}::swapped",
            "candidate_answer": case.candidate_b,
            "candidate_b": case.candidate_answer,
        }
    )


def planned_call_count(dataset: EvaluationDataset, repeat_count: int) -> int:
    """Return paid calls: two models × repeats × original plus swapped cases."""

    if repeat_count < 2:
        raise ValueError("Repeat count must be at least 2")
    pairwise_cases = sum(
        case.mode == EvaluationMode.PAIRWISE for case in dataset.cases
    )
    return 2 * repeat_count * (len(dataset.cases) + pairwise_cases)


class StabilityRunner:
    """Run repeated original and swapped evaluations without hiding failures."""

    def __init__(self, evaluator: SingleModelEvaluator) -> None:
        self._evaluator = evaluator

    def run(
        self,
        dataset: EvaluationDataset,
        *,
        repeat_count: int = 3,
        output_path: Optional[Union[str, Path]] = None,
        allow_drafts: bool = False,
        overwrite: bool = False,
        include_examples: bool = True,
        example_limit: int = 3,
        max_calls: Optional[int] = None,
    ) -> StabilityReport:
        """Run stability experiments and optionally save each case/model as JSONL."""

        if repeat_count < 2:
            raise ValueError("Repeat count must be at least 2")
        if example_limit < 0:
            raise ValueError("Example limit cannot be negative")
        if not allow_drafts:
            dataset.ensure_ready_for_production()
        planned_calls = planned_call_count(dataset, repeat_count)
        if max_calls is not None:
            if max_calls < 1:
                raise ValueError("Maximum calls must be at least 1")
            if planned_calls > max_calls:
                raise ValueError(
                    f"Stability run requires {planned_calls} calls, exceeding max_calls={max_calls}"
                )

        output_handle = None
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            output_handle = destination.open(
                "w" if overwrite else "x",
                encoding="utf-8",
            )

        results = []
        try:
            for case in dataset.cases:
                original_prompt = build_judge_prompt(
                    case,
                    include_examples=include_examples,
                    example_limit=example_limit,
                )
                swapped_case = (
                    make_swapped_input(case)
                    if case.mode == EvaluationMode.PAIRWISE
                    else None
                )
                swapped_prompt = (
                    build_judge_prompt(
                        swapped_case,
                        include_examples=include_examples,
                        example_limit=example_limit,
                    )
                    if swapped_case is not None
                    else None
                )

                for model in (JudgeModel.TERRA, JudgeModel.LUNA):
                    original = self._run_repeats(
                        model,
                        case,
                        original_prompt,
                        EvaluationOrder.ORIGINAL,
                        repeat_count,
                    )
                    swapped = (
                        self._run_repeats(
                            model,
                            swapped_case,
                            swapped_prompt,
                            EvaluationOrder.SWAPPED,
                            repeat_count,
                        )
                        if swapped_case is not None and swapped_prompt is not None
                        else []
                    )
                    result = _analyze_case(case, model, repeat_count, original, swapped)
                    results.append(result)
                    if output_handle is not None:
                        output_handle.write(result.model_dump_json() + "\n")
                        output_handle.flush()
        finally:
            if output_handle is not None:
                output_handle.close()

        return _build_report(dataset, repeat_count, results)

    def _run_repeats(
        self,
        model: JudgeModel,
        evaluation_input: EvaluationCase,
        prompt: JudgePrompt,
        order: EvaluationOrder,
        repeat_count: int,
    ) -> List[StabilityObservation]:
        observations = []
        for run_number in range(1, repeat_count + 1):
            try:
                judgment = self._evaluator.evaluate_prompt(
                    model,
                    prompt,
                    evaluation_input,
                )
                raw_decision = judgment.result.decision
                canonical_decision = (
                    normalize_swapped_decision(raw_decision)
                    if order == EvaluationOrder.SWAPPED
                    else raw_decision
                )
                observations.append(
                    StabilityObservation(
                        run_number=run_number,
                        order=order,
                        status=ObservationStatus.COMPLETED,
                        judgment=judgment,
                        raw_decision=raw_decision.value,
                        canonical_decision=canonical_decision.value,
                    )
                )
            except Exception as error:
                # Never persist provider exception text: it may contain user data.
                observations.append(
                    StabilityObservation(
                        run_number=run_number,
                        order=order,
                        status=ObservationStatus.ERROR,
                        error_type=type(error).__name__,
                        error_message=(
                            "Stability evaluation failed; inspect controlled logs"
                        ),
                    )
                )
        return observations


def _decision_summary(
    observations: Sequence[StabilityObservation],
) -> tuple:
    decisions = [
        item.canonical_decision
        for item in observations
        if item.status == ObservationStatus.COMPLETED
    ]
    if len(decisions) < 2:
        return None, False, None, None
    counts = Counter(decisions)
    highest_count = max(counts.values())
    modes = sorted(
        decision for decision, count in counts.items() if count == highest_count
    )
    tied = len(modes) > 1
    modal_decision = None if tied else modes[0]
    consistency = round(highest_count / len(decisions), 4)
    return modal_decision, tied, consistency, len(counts) == 1


def _score_variation(
    observations: Sequence[StabilityObservation],
) -> List[CriterionVariation]:
    completed_results = [
        item.judgment.result
        for item in observations
        if item.status == ObservationStatus.COMPLETED
    ]
    if not completed_results:
        return []
    if not all(isinstance(result, EvaluationResult) for result in completed_results):
        return []

    variations = []
    for criterion in Criterion:
        values = [
            next(
                score.score
                for score in result.scores
                if score.criterion == criterion
            )
            for result in completed_results
        ]
        variations.append(
            CriterionVariation(
                criterion=criterion,
                observations=len(values),
                mean=round(mean(values), 4),
                median=round(median(values), 4),
                population_stddev=round(pstdev(values), 4),
                minimum=min(values),
                maximum=max(values),
                score_range=max(values) - min(values),
            )
        )
    return variations


def _analyze_case(
    case: EvaluationCase,
    model: JudgeModel,
    repeat_count: int,
    original: List[StabilityObservation],
    swapped: List[StabilityObservation],
) -> StabilityCaseResult:
    modal, modal_tied, consistency, all_same = _decision_summary(original)
    successful = sum(
        item.status == ObservationStatus.COMPLETED for item in original
    )

    original_by_run = {
        item.run_number: item
        for item in original
        if item.status == ObservationStatus.COMPLETED
    }
    swapped_by_run = {
        item.run_number: item
        for item in swapped
        if item.status == ObservationStatus.COMPLETED
    }
    paired = [
        (original_by_run[run_number], swapped_by_run[run_number])
        for run_number in sorted(set(original_by_run) & set(swapped_by_run))
    ]
    mismatches = sum(
        original_item.canonical_decision != swapped_item.canonical_decision
        for original_item, swapped_item in paired
    )
    first_preference = sum(
        original_item.raw_decision == PairwiseDecision.A_WINS.value
        and swapped_item.raw_decision == PairwiseDecision.A_WINS.value
        for original_item, swapped_item in paired
    )
    second_preference = sum(
        original_item.raw_decision == PairwiseDecision.B_WINS.value
        and swapped_item.raw_decision == PairwiseDecision.B_WINS.value
        for original_item, swapped_item in paired
    )
    tie_pairs = [
        pair
        for pair in paired
        if PairwiseDecision.TIE.value
        in (pair[0].raw_decision, pair[1].raw_decision)
    ]
    consistent_ties = sum(
        left.raw_decision == right.raw_decision == PairwiseDecision.TIE.value
        for left, right in tie_pairs
    )
    insufficient_repeat = successful < 2
    position_rate = round(mismatches / len(paired), 4) if paired else None
    return StabilityCaseResult(
        case_id=case.case_id,
        mode=case.mode,
        review_status=case.review_status,
        model=model,
        repeat_count=repeat_count,
        original_observations=original,
        swapped_observations=swapped,
        successful_original_runs=successful,
        failed_original_runs=repeat_count - successful,
        modal_original_decision=modal,
        modal_decision_tied=modal_tied,
        repeat_consistency=consistency,
        all_original_decisions_same=all_same,
        score_variation=(
            _score_variation(original)
            if case.mode == EvaluationMode.SCORE
            else []
        ),
        position_comparisons=len(paired),
        position_mismatches=mismatches,
        position_flip_rate=position_rate,
        first_position_preference_pairs=first_preference,
        second_position_preference_pairs=second_preference,
        tie_relevant_pairs=len(tie_pairs),
        tie_consistent_pairs=consistent_ties,
        tie_consistency_rate=(
            round(consistent_ties / len(tie_pairs), 4) if tie_pairs else None
        ),
        requires_investigation=(
            insufficient_repeat
            or all_same is False
            or mismatches > 0
            or any(item.status == ObservationStatus.ERROR for item in original)
            or any(item.status == ObservationStatus.ERROR for item in swapped)
        ),
    )


def _model_summary(
    model: JudgeModel,
    results: Sequence[StabilityCaseResult],
) -> ModelStabilitySummary:
    model_results = [result for result in results if result.model == model]
    observations = [
        observation
        for result in model_results
        for observation in result.original_observations + result.swapped_observations
    ]
    failed_calls = sum(
        item.status == ObservationStatus.ERROR for item in observations
    )
    comparable = [
        result for result in model_results if result.repeat_consistency is not None
    ]
    consistencies = [result.repeat_consistency for result in comparable]
    position_results = [
        result for result in model_results if result.mode == EvaluationMode.PAIRWISE
    ]
    comparisons = sum(result.position_comparisons for result in position_results)
    mismatches = sum(result.position_mismatches for result in position_results)
    tie_relevant = sum(result.tie_relevant_pairs for result in position_results)
    consistent_ties = sum(
        result.tie_consistent_pairs for result in position_results
    )
    total_calls = len(observations)
    insufficient = [
        result.case_id
        for result in model_results
        if result.repeat_consistency is None
    ]
    return ModelStabilitySummary(
        model=model,
        evaluated_cases=len(model_results),
        total_calls=total_calls,
        failed_calls=failed_calls,
        failure_rate=round(failed_calls / total_calls, 4) if total_calls else 0,
        repeat_comparable_cases=len(comparable),
        stable_cases=sum(result.all_original_decisions_same is True for result in comparable),
        unstable_cases=sum(result.all_original_decisions_same is False for result in comparable),
        insufficient_repeat_cases=len(insufficient),
        mean_repeat_consistency=(
            round(mean(consistencies), 4) if consistencies else None
        ),
        median_repeat_consistency=(
            round(median(consistencies), 4) if consistencies else None
        ),
        position_cases=len(position_results),
        position_comparisons=comparisons,
        position_mismatches=mismatches,
        position_flip_rate=(
            round(mismatches / comparisons, 4) if comparisons else None
        ),
        first_position_preference_pairs=sum(
            result.first_position_preference_pairs for result in position_results
        ),
        second_position_preference_pairs=sum(
            result.second_position_preference_pairs for result in position_results
        ),
        tie_relevant_pairs=tie_relevant,
        tie_consistency_rate=(
            round(consistent_ties / tie_relevant, 4) if tie_relevant else None
        ),
        unstable_case_ids=sorted(
            result.case_id
            for result in comparable
            if result.all_original_decisions_same is False
        ),
        position_biased_case_ids=sorted(
            result.case_id
            for result in position_results
            if result.position_mismatches > 0
        ),
        insufficient_repeat_case_ids=sorted(insufficient),
    )


def _build_report(
    dataset: EvaluationDataset,
    repeat_count: int,
    results: List[StabilityCaseResult],
) -> StabilityReport:
    """Compatibility wrapper used by the runner after collecting results."""

    return calculate_stability_report(results)


def load_stability_results(
    path: Union[str, Path],
) -> List[StabilityCaseResult]:
    """Load and validate the Step 10 one-case-per-model JSONL output."""

    source = Path(path)
    results = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                results.append(StabilityCaseResult.model_validate_json(line))
            except ValueError as error:
                raise ValueError(
                    f"Invalid stability result at {source}:{line_number}"
                ) from error
    if not results:
        raise ValueError(f"Stability result file is empty: {source}")
    identities = [(result.case_id, result.model) for result in results]
    duplicates = sorted(
        f"{case_id}/{model.value}"
        for (case_id, model), count in Counter(identities).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate stability case/model results: {duplicates}")
    return results


def calculate_stability_report(
    results: Sequence[StabilityCaseResult],
) -> StabilityReport:
    """Rebuild aggregate Step 10 metrics from saved stability results."""

    if not results:
        raise ValueError("At least one stability result is required")
    case_ids = {result.case_id for result in results}
    repeat_counts = {result.repeat_count for result in results}
    if len(repeat_counts) != 1:
        raise ValueError("Stability results contain different repeat counts")
    identities = [(result.case_id, result.model) for result in results]
    if len(set(identities)) != len(identities):
        raise ValueError("Stability results contain duplicate case/model pairs")
    expected = {
        (case_id, model) for case_id in case_ids for model in JudgeModel
    }
    missing = expected - set(identities)
    if missing:
        rendered = sorted(f"{case_id}/{model.value}" for case_id, model in missing)
        raise ValueError(f"Stability results are missing case/model pairs: {rendered}")

    per_model = {
        model: _model_summary(model, results)
        for model in (JudgeModel.TERRA, JudgeModel.LUNA)
    }
    total_calls = sum(item.total_calls for item in per_model.values())
    failed_calls = sum(item.failed_calls for item in per_model.values())
    representative_cases = {
        result.case_id: result for result in results
    }
    pairwise_cases = sum(
        case.mode == EvaluationMode.PAIRWISE
        for case in representative_cases.values()
    )
    warnings = []
    if any(case.review_status == ReviewStatus.DRAFT for case in representative_cases.values()):
        warnings.append(
            "Draft human labels are present; this stability run cannot approve production."
        )
    if len(case_ids) < 30:
        warnings.append(
            "Fewer than 30 cases were tested; repeat-consistency estimates are unstable."
        )
    if pairwise_cases < 30:
        warnings.append(
            "Fewer than 30 pairwise cases were tested; position-flip estimates are unstable."
        )
    return StabilityReport(
        dataset_cases=len(case_ids),
        repeat_count=next(iter(repeat_counts)),
        total_calls=total_calls,
        failed_calls=failed_calls,
        failure_rate=round(failed_calls / total_calls, 4) if total_calls else 0,
        per_model=per_model,
        results=results,
        warnings=warnings,
    )


def main() -> None:
    """Command-line entry point for a real Terra/Luna stability experiment."""

    parser = argparse.ArgumentParser(
        description="Measure repeat consistency and pairwise position bias"
    )
    parser.add_argument("--dataset", required=True, help="Input JSONL dataset")
    parser.add_argument("--output", required=True, help="Output JSONL observations")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--max-calls",
        type=int,
        default=200,
        help="Safety limit for paid Azure calls (default: 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned call count without contacting Azure",
    )
    parser.add_argument("--allow-drafts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-examples", action="store_true")
    parser.add_argument("--example-limit", type=int, default=3)
    args = parser.parse_args()

    dataset = load_jsonl(args.dataset)
    if args.dry_run:
        print(
            f"planned_calls={planned_call_count(dataset, args.repeats)} "
            f"cases={len(dataset.cases)} repeats={args.repeats}"
        )
        return

    evaluator = TwoModelJudge.from_settings(AzureJudgeSettings.from_env())
    report = StabilityRunner(evaluator).run(
        dataset,
        repeat_count=args.repeats,
        output_path=args.output,
        allow_drafts=args.allow_drafts,
        overwrite=args.overwrite,
        include_examples=not args.no_examples,
        example_limit=args.example_limit,
        max_calls=args.max_calls,
    )
    print(report.model_dump_json(indent=2, exclude={"results"}))


if __name__ == "__main__":
    main()
