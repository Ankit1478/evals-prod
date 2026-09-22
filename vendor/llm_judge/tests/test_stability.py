import tempfile
import unittest
from pathlib import Path

from llm_judge.azure_client import TokenUsage
from llm_judge.contracts import (
    BinaryEvaluationResult,
    Criterion,
    CriterionScore,
    EvaluationMode,
    EvaluationResult,
    PairwiseDecision,
    PairwiseEvaluationResult,
    ReferencePolicy,
)
from llm_judge.dataset import EvaluationCase, EvaluationDataset
from llm_judge.multi_judge import JudgeModel, ModelJudgment
from llm_judge.rubric import ExampleKind
from llm_judge.stability import (
    EvaluationOrder,
    ObservationStatus,
    StabilityRunner,
    calculate_stability_report,
    load_stability_results,
    make_swapped_input,
    normalize_swapped_decision,
    planned_call_count,
)


def binary_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="binary-001",
        case_kind=ExampleKind.GOOD,
        mode=EvaluationMode.BINARY,
        reference_policy=ReferencePolicy.REQUIRED,
        question="Is 2 + 2 equal to 4?",
        reference_answer="Yes.",
        candidate_answer="Yes.",
        expected_binary_decision="PASS",
        review_notes="Draft test label.",
    )


def pairwise_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="pair-001",
        case_kind=ExampleKind.PAIRWISE_A_WINS,
        mode=EvaluationMode.PAIRWISE,
        reference_policy=ReferencePolicy.REQUIRED,
        question="Which answer is correct?",
        reference_answer="The answer is 4.",
        candidate_answer="4",
        candidate_b="5",
        expected_pairwise_decision="A_WINS",
        review_notes="Draft test label.",
    )


def score_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="score-001",
        case_kind=ExampleKind.BORDERLINE,
        mode=EvaluationMode.SCORE,
        reference_policy=ReferencePolicy.REQUIRED,
        question="What is 2 + 2?",
        reference_answer="4",
        candidate_answer="4",
        expected_scores={criterion: 4 for criterion in Criterion},
        review_notes="Draft test label.",
    )


class SequenceEvaluator:
    """Return configured decisions/scores while recording prompt identity."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.indices = {}
        self.prompt_ids = {}
        self.calls = 0

    def evaluate_prompt(self, model, prompt, evaluation_input):
        key = (model, evaluation_input.case_id)
        index = self.indices.get(key, 0)
        self.indices[key] = index + 1
        self.prompt_ids.setdefault(key, []).append(id(prompt))
        self.calls += 1

        configured = self.responses.get(key)
        if configured is None:
            if evaluation_input.mode == EvaluationMode.BINARY:
                value = "PASS"
            elif evaluation_input.mode == EvaluationMode.PAIRWISE:
                value = "A_WINS"
            else:
                value = [4, 4, 4, 4]
        else:
            value = configured[index]
        if isinstance(value, Exception):
            raise value

        if evaluation_input.mode == EvaluationMode.BINARY:
            result = BinaryEvaluationResult(
                case_id=evaluation_input.case_id,
                decision=value,
                evidence="Evidence.",
            )
        elif evaluation_input.mode == EvaluationMode.PAIRWISE:
            result = PairwiseEvaluationResult(
                case_id=evaluation_input.case_id,
                decision=value,
                evidence="Evidence.",
            )
        else:
            result = EvaluationResult(
                case_id=evaluation_input.case_id,
                scores=[
                    CriterionScore(
                        criterion=criterion,
                        score=score,
                        evidence="Evidence.",
                    )
                    for criterion, score in zip(Criterion, value)
                ],
                summary="Summary.",
            )
        return ModelJudgment(
            model=model,
            deployment=model.value,
            usage=TokenUsage(total_tokens=10),
            result=result,
        )


class StabilityTests(unittest.TestCase):
    def test_swapped_decisions_are_mapped_to_original_candidates(self) -> None:
        self.assertEqual(
            normalize_swapped_decision(PairwiseDecision.A_WINS),
            PairwiseDecision.B_WINS,
        )
        self.assertEqual(
            normalize_swapped_decision(PairwiseDecision.B_WINS),
            PairwiseDecision.A_WINS,
        )
        self.assertEqual(
            normalize_swapped_decision(PairwiseDecision.TIE),
            PairwiseDecision.TIE,
        )
        original = pairwise_case()
        swapped = make_swapped_input(original)
        self.assertEqual(swapped.candidate_answer, original.candidate_b)
        self.assertEqual(swapped.candidate_b, original.candidate_answer)

    def test_repeat_and_position_metrics_for_both_models(self) -> None:
        binary = binary_case()
        pairwise = pairwise_case()
        responses = {
            (JudgeModel.TERRA, binary.case_id): ["PASS", "PASS", "FAIL"],
            (JudgeModel.LUNA, binary.case_id): ["PASS", "PASS", "PASS"],
            (JudgeModel.TERRA, pairwise.case_id): ["A_WINS"] * 3,
            (JudgeModel.TERRA, f"{pairwise.case_id}::swapped"): ["B_WINS"] * 3,
            (JudgeModel.LUNA, pairwise.case_id): ["A_WINS"] * 3,
            # Luna always chooses displayed A, revealing first-position bias.
            (JudgeModel.LUNA, f"{pairwise.case_id}::swapped"): ["A_WINS"] * 3,
        }
        evaluator = SequenceEvaluator(responses)
        dataset = EvaluationDataset(cases=[binary, pairwise])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stability.jsonl"
            report = StabilityRunner(evaluator).run(
                dataset,
                repeat_count=3,
                allow_drafts=True,
                output_path=output,
            )
            lines = output.read_text(encoding="utf-8").splitlines()

        terra_binary = next(
            item
            for item in report.results
            if item.model == JudgeModel.TERRA and item.case_id == binary.case_id
        )
        terra_pair = next(
            item
            for item in report.results
            if item.model == JudgeModel.TERRA and item.case_id == pairwise.case_id
        )
        luna_pair = next(
            item
            for item in report.results
            if item.model == JudgeModel.LUNA and item.case_id == pairwise.case_id
        )

        self.assertEqual(terra_binary.repeat_consistency, 0.6667)
        self.assertFalse(terra_binary.all_original_decisions_same)
        self.assertEqual(terra_pair.position_flip_rate, 0.0)
        self.assertEqual(luna_pair.position_flip_rate, 1.0)
        self.assertEqual(luna_pair.first_position_preference_pairs, 3)
        self.assertTrue(luna_pair.requires_investigation)
        self.assertEqual(report.per_model[JudgeModel.TERRA].unstable_cases, 1)
        self.assertEqual(report.per_model[JudgeModel.LUNA].position_flip_rate, 1.0)
        self.assertEqual(report.total_calls, 18)
        self.assertEqual(len(lines), 4)
        self.assertTrue(
            all(len(set(ids)) == 1 for ids in evaluator.prompt_ids.values())
        )

    def test_score_runs_report_mean_median_spread_and_range(self) -> None:
        case = score_case()
        responses = {
            (JudgeModel.TERRA, case.case_id): [
                [3, 3, 3, 3],
                [4, 4, 4, 4],
                [5, 5, 5, 5],
            ],
            (JudgeModel.LUNA, case.case_id): [[4, 4, 4, 4]] * 3,
        }
        report = StabilityRunner(SequenceEvaluator(responses)).run(
            EvaluationDataset(cases=[case]),
            repeat_count=3,
            allow_drafts=True,
        )
        terra = next(
            item for item in report.results if item.model == JudgeModel.TERRA
        )
        correctness = next(
            item
            for item in terra.score_variation
            if item.criterion == Criterion.CORRECTNESS
        )

        self.assertEqual(correctness.mean, 4.0)
        self.assertEqual(correctness.median, 4.0)
        self.assertEqual(correctness.population_stddev, 0.8165)
        self.assertEqual(correctness.score_range, 2)

    def test_failed_repeat_is_recorded_and_remaining_calls_continue(self) -> None:
        case = binary_case()
        responses = {
            (JudgeModel.TERRA, case.case_id): [
                "PASS",
                RuntimeError("secret provider detail"),
                "PASS",
            ]
        }
        report = StabilityRunner(SequenceEvaluator(responses)).run(
            EvaluationDataset(cases=[case]),
            repeat_count=3,
            allow_drafts=True,
        )
        terra = next(
            item for item in report.results if item.model == JudgeModel.TERRA
        )
        failed = terra.original_observations[1]

        self.assertEqual(failed.status, ObservationStatus.ERROR)
        self.assertEqual(failed.error_type, "RuntimeError")
        self.assertNotIn("secret provider detail", failed.error_message)
        self.assertEqual(terra.successful_original_runs, 2)
        self.assertTrue(terra.requires_investigation)
        self.assertEqual(report.failed_calls, 1)
        self.assertEqual(report.total_calls, 6)

    def test_production_mode_rejects_drafts_before_calls(self) -> None:
        evaluator = SequenceEvaluator()

        with self.assertRaises(ValueError):
            StabilityRunner(evaluator).run(
                EvaluationDataset(cases=[binary_case()]),
                repeat_count=3,
            )

        self.assertEqual(evaluator.calls, 0)

    def test_call_estimate_and_safety_limit_block_expensive_run(self) -> None:
        dataset = EvaluationDataset(cases=[binary_case(), pairwise_case()])
        evaluator = SequenceEvaluator()

        self.assertEqual(planned_call_count(dataset, repeat_count=3), 18)
        with self.assertRaisesRegex(ValueError, "requires 18 calls"):
            StabilityRunner(evaluator).run(
                dataset,
                repeat_count=3,
                allow_drafts=True,
                max_calls=17,
            )

        self.assertEqual(evaluator.calls, 0)

    def test_saved_results_can_rebuild_the_same_summary(self) -> None:
        dataset = EvaluationDataset(cases=[binary_case(), pairwise_case()])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stability.jsonl"
            original = StabilityRunner(SequenceEvaluator()).run(
                dataset,
                repeat_count=2,
                allow_drafts=True,
                output_path=output,
            )
            loaded = load_stability_results(output)

        rebuilt = calculate_stability_report(loaded)

        self.assertEqual(rebuilt.dataset_cases, original.dataset_cases)
        self.assertEqual(rebuilt.total_calls, original.total_calls)
        self.assertEqual(rebuilt.per_model, original.per_model)


if __name__ == "__main__":
    unittest.main()
