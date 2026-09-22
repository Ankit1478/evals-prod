"""Step 8: run Terra and Luna across a human-labelled evaluation dataset."""

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import CRITERIA, TASK_DEFINITION, Criterion, Decision, EvaluationMode
from .dataset import EvaluationCase, EvaluationDataset, ReviewStatus, load_jsonl
from .multi_judge import TwoModelJudge, TwoModelJudgeResult
from .settings import AzureJudgeSettings


class DatasetJudge(Protocol):
    """Interface used by the real two-model judge and offline test fakes."""

    def evaluate(
        self,
        evaluation_input: EvaluationCase,
        *,
        include_examples: bool = True,
        example_limit: int = 3,
    ) -> TwoModelJudgeResult:
        ...


class CaseRunStatus(str, Enum):
    """Whether one dataset case completed or failed safely."""

    COMPLETED = "completed"
    ERROR = "error"


class CaseRunResult(BaseModel):
    """Auditable comparison between a human label and two model judgments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    mode: EvaluationMode
    review_status: ReviewStatus
    human_decision: str
    human_scores: Optional[Dict[Criterion, int]] = None
    status: CaseRunStatus
    judge_result: Optional[TwoModelJudgeResult] = None
    matches_human: Optional[bool] = None
    requires_human_review: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> "CaseRunResult":
        if self.status == CaseRunStatus.COMPLETED:
            if self.judge_result is None or self.error_type or self.error_message:
                raise ValueError("Completed case requires a result and no error")
        else:
            if self.judge_result is not None or not self.error_type:
                raise ValueError("Failed case requires an error and no judge result")
            if self.matches_human is not None:
                raise ValueError("Failed case cannot claim human agreement")
        return self


class DatasetRunReport(BaseModel):
    """In-memory summary returned after all cases have been attempted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_cases: int = Field(ge=1)
    completed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    matches_human: int = Field(ge=0)
    mismatches_human: int = Field(ge=0)
    model_disagreements: int = Field(ge=0)
    human_review_required: int = Field(ge=0)
    results: List[CaseRunResult]


def human_decision_for_case(case: EvaluationCase) -> str:
    """Convert the mode-specific human answer key into one comparable label."""

    if case.mode == EvaluationMode.BINARY:
        return case.expected_binary_decision.value
    if case.mode == EvaluationMode.PAIRWISE:
        return case.expected_pairwise_decision.value

    scores = case.expected_scores
    weighted_score = sum(
        scores[criterion] * definition.weight
        for criterion, definition in CRITERIA.items()
    )
    critical_scores_pass = all(
        scores[criterion] >= TASK_DEFINITION.minimum_critical_score
        for criterion in (Criterion.CORRECTNESS, Criterion.RELEVANCE)
    )
    if critical_scores_pass and weighted_score >= TASK_DEFINITION.pass_threshold:
        return Decision.PASS.value
    return Decision.FAIL.value


class DatasetRunner:
    """Run every case without allowing one provider failure to stop the batch."""

    def __init__(self, judge: DatasetJudge) -> None:
        self._judge = judge

    def run(
        self,
        dataset: EvaluationDataset,
        *,
        output_path: Optional[Union[str, Path]] = None,
        allow_drafts: bool = False,
        overwrite: bool = False,
        include_examples: bool = True,
        example_limit: int = 3,
    ) -> DatasetRunReport:
        """Evaluate a dataset and optionally append each completed attempt to JSONL.

        Production mode is the default and requires the complete dataset to pass
        its human-review and coverage gates. ``allow_drafts`` is only for learning
        and must be selected explicitly.
        """

        if not allow_drafts:
            dataset.ensure_ready_for_production()
        if example_limit < 0:
            raise ValueError("Example limit cannot be negative")

        output_handle = None
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Refuse accidental replacement. The CLI requires --overwrite when a
            # caller intentionally wants to replace a previous run.
            output_handle = destination.open(
                "w" if overwrite else "x",
                encoding="utf-8",
            )

        results = []
        try:
            for case in dataset.cases:
                result = self._run_case(
                    case,
                    include_examples=include_examples,
                    example_limit=example_limit,
                )
                results.append(result)
                if output_handle is not None:
                    output_handle.write(result.model_dump_json() + "\n")
                    # Flush each case so an interrupted batch keeps finished work.
                    output_handle.flush()
        finally:
            if output_handle is not None:
                output_handle.close()

        return _build_report(dataset, results)

    def _run_case(
        self,
        case: EvaluationCase,
        *,
        include_examples: bool,
        example_limit: int,
    ) -> CaseRunResult:
        human_decision = human_decision_for_case(case)
        try:
            judge_result = self._judge.evaluate(
                case,
                include_examples=include_examples,
                example_limit=example_limit,
            )
        except Exception as error:
            # Do not copy exception text into a report: an SDK exception could
            # include candidate data or provider details. Keep only its type.
            return CaseRunResult(
                case_id=case.case_id,
                mode=case.mode,
                review_status=case.review_status,
                human_decision=human_decision,
                human_scores=case.expected_scores,
                status=CaseRunStatus.ERROR,
                requires_human_review=True,
                error_type=type(error).__name__,
                error_message="Case evaluation failed; inspect controlled logs",
            )

        aggregate_decision = judge_result.aggregate_decision
        matches_human = (
            aggregate_decision == human_decision
            if aggregate_decision is not None
            else None
        )
        return CaseRunResult(
            case_id=case.case_id,
            mode=case.mode,
            review_status=case.review_status,
            human_decision=human_decision,
            human_scores=case.expected_scores,
            status=CaseRunStatus.COMPLETED,
            judge_result=judge_result,
            matches_human=matches_human,
            requires_human_review=judge_result.requires_human_review,
        )


def _build_report(
    dataset: EvaluationDataset,
    results: List[CaseRunResult],
) -> DatasetRunReport:
    completed = [item for item in results if item.status == CaseRunStatus.COMPLETED]
    return DatasetRunReport(
        dataset_cases=len(dataset.cases),
        completed_cases=len(completed),
        failed_cases=len(results) - len(completed),
        matches_human=sum(item.matches_human is True for item in completed),
        mismatches_human=sum(item.matches_human is False for item in completed),
        model_disagreements=sum(
            item.judge_result is not None and not item.judge_result.agreement
            for item in completed
        ),
        human_review_required=sum(item.requires_human_review for item in results),
        results=results,
    )


def main() -> None:
    """Command-line entry point for a real Azure dataset run."""

    parser = argparse.ArgumentParser(
        description="Evaluate a human-labelled dataset with Terra and Luna"
    )
    parser.add_argument("--dataset", required=True, help="Input JSONL dataset")
    parser.add_argument("--output", required=True, help="Output JSONL results")
    parser.add_argument(
        "--allow-drafts",
        action="store_true",
        help="Learning only: allow labels that have not been human-reviewed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Do not add rubric examples to judge prompts",
    )
    parser.add_argument("--example-limit", type=int, default=3)
    args = parser.parse_args()

    settings = AzureJudgeSettings.from_env()
    judge = TwoModelJudge.from_settings(settings)
    report = DatasetRunner(judge).run(
        load_jsonl(args.dataset),
        output_path=args.output,
        allow_drafts=args.allow_drafts,
        overwrite=args.overwrite,
        include_examples=not args.no_examples,
        example_limit=args.example_limit,
    )
    # Print only summary counts. Full case data is kept in the requested JSONL.
    print(report.model_dump_json(indent=2, exclude={"results"}))


if __name__ == "__main__":
    main()
