"""Step 3: load and validate human-labelled evaluation cases."""

import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import (
    RELIABLE_EXAMPLE_POLICY,
    Criterion,
    Decision,
    EvaluationInput,
    EvaluationMode,
    ExampleLabel,
    PairwiseDecision,
    ReliableExamplePolicy,
)
from .rubric import ExampleKind


class ReviewStatus(str, Enum):
    """Shows whether suggested labels have been confirmed by people."""

    DRAFT = "draft"
    HUMAN_REVIEWED = "human_reviewed"
    ADJUDICATED = "adjudicated"


class EvaluationCase(EvaluationInput):
    """One judge input plus its expected human-approved result.

    Draft cases may contain suggested labels for development, but only
    HUMAN_REVIEWED or ADJUDICATED cases are valid production gold data.
    """

    case_kind: ExampleKind
    expected_scores: Optional[Dict[Criterion, int]] = None
    expected_binary_decision: Optional[Decision] = None
    expected_pairwise_decision: Optional[PairwiseDecision] = None
    review_status: ReviewStatus = ReviewStatus.DRAFT
    reviewer_count: int = Field(default=0, ge=0)
    review_notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_expected_label(self) -> "EvaluationCase":
        """Make the expected label agree with the selected evaluation mode."""

        if self.mode == EvaluationMode.SCORE:
            if not self.expected_scores or set(self.expected_scores) != set(Criterion):
                raise ValueError("Score cases need one expected score per criterion")
            if any(score < 1 or score > 5 for score in self.expected_scores.values()):
                raise ValueError("Expected scores must be between 1 and 5")
            if self.expected_binary_decision or self.expected_pairwise_decision:
                raise ValueError("Score cases cannot include another mode's decision")
        elif self.mode == EvaluationMode.BINARY:
            if not self.expected_binary_decision:
                raise ValueError("Binary cases need an expected PASS or FAIL decision")
            if self.expected_scores or self.expected_pairwise_decision:
                raise ValueError("Binary cases cannot include another mode's labels")
        else:
            if not self.expected_pairwise_decision:
                raise ValueError("Pairwise cases need an expected A/B/TIE decision")
            if self.expected_scores or self.expected_binary_decision:
                raise ValueError("Pairwise cases cannot include another mode's labels")

        if self.case_kind == ExampleKind.PAIRWISE_A_WINS and (
            self.expected_pairwise_decision != PairwiseDecision.A_WINS
        ):
            raise ValueError("PAIRWISE_A_WINS cases must expect A_WINS")
        if self.case_kind == ExampleKind.PAIRWISE_B_WINS and (
            self.expected_pairwise_decision != PairwiseDecision.B_WINS
        ):
            raise ValueError("PAIRWISE_B_WINS cases must expect B_WINS")
        if self.case_kind == ExampleKind.PAIRWISE_TIE and (
            self.expected_pairwise_decision != PairwiseDecision.TIE
        ):
            raise ValueError("PAIRWISE_TIE cases must expect TIE")

        if self.review_status == ReviewStatus.DRAFT and self.reviewer_count != 0:
            raise ValueError("Draft cases cannot claim human reviewers")
        if self.review_status == ReviewStatus.HUMAN_REVIEWED and self.reviewer_count < 1:
            raise ValueError("Human-reviewed cases need at least one reviewer")
        if self.review_status == ReviewStatus.ADJUDICATED and self.reviewer_count < 2:
            raise ValueError("Adjudicated cases need at least two reviewers")
        return self


_POLICY_KIND_MAP = {
    ExampleLabel.GOOD: ExampleKind.GOOD,
    ExampleLabel.BAD: ExampleKind.BAD,
    ExampleLabel.BORDERLINE: ExampleKind.BORDERLINE,
    ExampleLabel.PAIRWISE_TIE: ExampleKind.PAIRWISE_TIE,
}


class EvaluationDataset(BaseModel):
    """A collection of unique, validated evaluation cases."""

    model_config = ConfigDict(frozen=True)

    cases: List[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_case_ids(self) -> "EvaluationDataset":
        case_ids = [case.case_id for case in self.cases]
        duplicates = sorted(
            case_id for case_id, count in Counter(case_ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate case IDs: {duplicates}")
        return self

    @property
    def kind_counts(self) -> Dict[ExampleKind, int]:
        """Count how many cases exist for each learning/test category."""

        counts = Counter(case.case_kind for case in self.cases)
        return {kind: counts.get(kind, 0) for kind in ExampleKind}

    def validate_example_policy(
        self,
        policy: ReliableExamplePolicy = RELIABLE_EXAMPLE_POLICY,
    ) -> None:
        """Check the Step 1 minimum-category policy against this dataset."""

        missing = {}
        for required_label in policy.required_labels:
            kind = _POLICY_KIND_MAP[required_label]
            actual = self.kind_counts[kind]
            if actual < policy.minimum_examples_per_label:
                missing[kind.value] = {
                    "required": policy.minimum_examples_per_label,
                    "actual": actual,
                }
        if missing:
            raise ValueError(f"Dataset does not satisfy example policy: {missing}")

    def ensure_ready_for_production(self) -> None:
        """Block deployment until coverage and human-review requirements pass."""

        self.validate_example_policy()
        unreviewed = [
            case.case_id
            for case in self.cases
            if case.review_status == ReviewStatus.DRAFT
        ]
        if unreviewed:
            raise ValueError(f"Draft cases require human review: {unreviewed}")


def load_jsonl(path: Union[str, Path]) -> EvaluationDataset:
    """Load one JSON object per line and return a validated dataset."""

    source = Path(path)
    cases = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                cases.append(EvaluationCase.model_validate_json(line))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"Invalid case at {source}:{line_number}: {error}") from error
    if not cases:
        raise ValueError(f"Dataset is empty: {source}")
    return EvaluationDataset(cases=cases)

