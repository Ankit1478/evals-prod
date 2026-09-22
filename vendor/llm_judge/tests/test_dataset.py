import unittest
from pathlib import Path

from pydantic import ValidationError

from llm_judge.contracts import EvaluationMode, PairwiseDecision, ReferencePolicy
from llm_judge.dataset import EvaluationCase, EvaluationDataset, ReviewStatus, load_jsonl
from llm_judge.rubric import ExampleKind


EXAMPLE_DATASET = (
    Path(__file__).resolve().parents[1] / "datasets" / "evaluation_cases.example.jsonl"
)


class DatasetTests(unittest.TestCase):
    def test_example_dataset_loads_and_has_all_case_kinds(self) -> None:
        dataset = load_jsonl(EXAMPLE_DATASET)

        self.assertEqual(len(dataset.cases), 12)
        for kind in ExampleKind:
            self.assertGreaterEqual(dataset.kind_counts[kind], 2)

    def test_example_dataset_satisfies_minimum_category_policy(self) -> None:
        dataset = load_jsonl(EXAMPLE_DATASET)

        dataset.validate_example_policy()

    def test_draft_dataset_is_blocked_from_production(self) -> None:
        dataset = load_jsonl(EXAMPLE_DATASET)

        with self.assertRaisesRegex(ValueError, "require human review"):
            dataset.ensure_ready_for_production()

    def test_duplicate_case_ids_are_rejected(self) -> None:
        case = load_jsonl(EXAMPLE_DATASET).cases[0]

        with self.assertRaises(ValidationError):
            EvaluationDataset(cases=[case, case])

    def test_pairwise_kind_must_match_expected_decision(self) -> None:
        with self.assertRaises(ValidationError):
            EvaluationCase(
                case_id="mismatched-pairwise-label",
                case_kind=ExampleKind.PAIRWISE_A_WINS,
                mode=EvaluationMode.PAIRWISE,
                reference_policy=ReferencePolicy.REQUIRED,
                question="Which candidate is correct?",
                reference_answer="Candidate A is correct.",
                candidate_answer="Candidate A",
                candidate_b="Candidate B",
                expected_pairwise_decision=PairwiseDecision.B_WINS,
                review_status=ReviewStatus.DRAFT,
                reviewer_count=0,
                review_notes="Invalid test case.",
            )

    def test_human_review_status_requires_a_reviewer(self) -> None:
        with self.assertRaises(ValidationError):
            EvaluationCase(
                case_id="false-human-review",
                case_kind=ExampleKind.PAIRWISE_TIE,
                mode=EvaluationMode.PAIRWISE,
                reference_policy=ReferencePolicy.REFERENCE_FREE,
                question="Which answer is better?",
                candidate_answer="Answer A",
                candidate_b="Answer B",
                expected_pairwise_decision=PairwiseDecision.TIE,
                review_status=ReviewStatus.HUMAN_REVIEWED,
                reviewer_count=0,
                review_notes="No reviewer was recorded.",
            )


if __name__ == "__main__":
    unittest.main()
