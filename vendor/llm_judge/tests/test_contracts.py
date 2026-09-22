import unittest

from pydantic import ValidationError

from llm_judge.contracts import (
    INPUT_ORDER_POLICY,
    RELIABLE_EXAMPLE_POLICY,
    TASK_DEFINITION,
    BinaryEvaluationResult,
    Criterion,
    CriterionScore,
    Decision,
    EvaluationMode,
    EvaluationInput,
    EvaluationResult,
    ExampleLabel,
    PairwiseDecision,
    PairwiseEvaluationResult,
    ReferencePolicy,
)


def score(criterion: Criterion, value: int) -> CriterionScore:
    return CriterionScore(
        criterion=criterion,
        score=value,
        evidence="Evidence from the supplied reference.",
    )


class EvaluationContractTests(unittest.TestCase):
    def test_input_requires_reference_answer(self) -> None:
        with self.assertRaises(ValidationError):
            EvaluationInput(
                case_id="case-1",
                question="What is Python?",
                reference_answer="",
                candidate_answer="Python is a programming language.",
            )

    def test_reference_free_input_does_not_require_a_reference(self) -> None:
        evaluation = EvaluationInput(
            case_id="case-reference-free",
            mode=EvaluationMode.BINARY,
            reference_policy=ReferencePolicy.REFERENCE_FREE,
            question="Is this answer clear?",
            candidate_answer="Yes. It explains the result in two short steps.",
        )

        self.assertIsNone(evaluation.reference_answer)

    def test_pairwise_input_requires_second_candidate(self) -> None:
        with self.assertRaises(ValidationError):
            EvaluationInput(
                case_id="case-pairwise",
                mode=EvaluationMode.PAIRWISE,
                reference_policy=ReferencePolicy.REFERENCE_FREE,
                question="Which answer is better?",
                candidate_answer="Answer A",
            )

    def test_pairwise_input_accepts_two_blinded_candidates(self) -> None:
        evaluation = EvaluationInput(
            case_id="case-pairwise",
            mode=EvaluationMode.PAIRWISE,
            question="What is Python?",
            reference_answer="Python is a programming language.",
            candidate_answer="Python is a programming language.",
            candidate_b="Python is a snake.",
        )

        self.assertEqual(evaluation.mode, EvaluationMode.PAIRWISE)
        self.assertEqual(evaluation.candidate_b, "Python is a snake.")

    def test_step_one_prefers_pairwise_and_controls_position(self) -> None:
        self.assertEqual(TASK_DEFINITION.preferred_mode, EvaluationMode.PAIRWISE)
        self.assertTrue(INPUT_ORDER_POLICY.blind_candidate_identity)
        self.assertTrue(INPUT_ORDER_POLICY.evaluate_swapped_order)
        self.assertEqual(TASK_DEFINITION.reliability_thresholds, {})
        self.assertIn(ExampleLabel.BORDERLINE, RELIABLE_EXAMPLE_POLICY.required_labels)

    def test_result_passes_when_thresholds_are_met(self) -> None:
        result = EvaluationResult(
            case_id="case-1",
            scores=[
                score(Criterion.CORRECTNESS, 4),
                score(Criterion.RELEVANCE, 4),
                score(Criterion.COMPLETENESS, 3),
                score(Criterion.CLARITY, 4),
            ],
            summary="The answer is correct and directly addresses the question.",
        )

        self.assertEqual(result.weighted_score, 3.8)
        self.assertEqual(result.decision, Decision.PASS)

    def test_low_correctness_forces_failure(self) -> None:
        result = EvaluationResult(
            case_id="case-2",
            scores=[
                score(Criterion.CORRECTNESS, 2),
                score(Criterion.RELEVANCE, 5),
                score(Criterion.COMPLETENESS, 5),
                score(Criterion.CLARITY, 5),
            ],
            summary="The response is polished but contains a material factual error.",
        )

        self.assertEqual(result.weighted_score, 3.8)
        self.assertEqual(result.decision, Decision.FAIL)

    def test_all_criteria_are_required_exactly_once(self) -> None:
        with self.assertRaises(ValidationError):
            EvaluationResult(
                case_id="case-3",
                scores=[
                    score(Criterion.CORRECTNESS, 4),
                    score(Criterion.RELEVANCE, 4),
                    score(Criterion.COMPLETENESS, 4),
                    score(Criterion.COMPLETENESS, 4),
                ],
                summary="Invalid duplicate criterion.",
            )

    def test_binary_and_pairwise_results_are_explicit(self) -> None:
        binary = BinaryEvaluationResult(
            case_id="case-binary",
            decision=Decision.PASS,
            evidence="The central statement is supported by the reference.",
        )
        pairwise = PairwiseEvaluationResult(
            case_id="case-pairwise",
            decision=PairwiseDecision.A_WINS,
            evidence="Candidate A is correct while candidate B contradicts the reference.",
        )

        self.assertEqual(binary.decision, Decision.PASS)
        self.assertEqual(pairwise.decision, PairwiseDecision.A_WINS)


if __name__ == "__main__":
    unittest.main()
