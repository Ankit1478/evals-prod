import unittest

from pydantic import ValidationError

from llm_judge.contracts import (
    Criterion,
    EvaluationMode,
    PairwiseDecision,
    ReferencePolicy,
)
from llm_judge.rubric import (
    ACTIVE_RUBRIC,
    RUBRIC_V1,
    RUBRIC_V2,
    EvaluationRubric,
    ExampleKind,
    RubricCriterion,
    RubricExample,
)


class RubricTests(unittest.TestCase):
    def test_v1_contains_every_criterion_and_score_anchor(self) -> None:
        self.assertEqual(
            {item.criterion for item in RUBRIC_V1.criteria},
            set(Criterion),
        )
        for item in RUBRIC_V1.criteria:
            self.assertEqual(set(item.score_anchors), {1, 2, 3, 4, 5})

    def test_criterion_lookup(self) -> None:
        correctness = RUBRIC_V1.for_criterion(Criterion.CORRECTNESS)

        self.assertIn("reference", correctness.definition)
        self.assertIn("fabricated", correctness.score_anchors[1])

    def test_v2_is_active_and_preserves_v1(self) -> None:
        self.assertEqual(RUBRIC_V1.version, "1.0.0")
        self.assertEqual(RUBRIC_V2.version, "2.0.0")
        self.assertIs(ACTIVE_RUBRIC, RUBRIC_V2)
        self.assertEqual(RUBRIC_V2.criteria, RUBRIC_V1.criteria)

    def test_v2_covers_every_mode_and_reference_policy(self) -> None:
        self.assertEqual(set(RUBRIC_V2.mode_instructions), set(EvaluationMode))
        self.assertEqual(set(RUBRIC_V2.reference_instructions), set(ReferencePolicy))

    def test_v2_defines_every_pairwise_outcome(self) -> None:
        self.assertEqual(
            {guide.decision for guide in RUBRIC_V2.pairwise_outcomes},
            set(PairwiseDecision),
        )

    def test_v2_contains_all_required_example_kinds(self) -> None:
        self.assertEqual(
            {example.kind for example in RUBRIC_V2.examples},
            set(ExampleKind),
        )
        self.assertEqual(len(RUBRIC_V2.examples), 6)

    def test_v2_has_explicit_bias_controls(self) -> None:
        guidance = " ".join(RUBRIC_V2.bias_control_instructions).lower()

        for expected in ("order", "model", "length", "confident", "formatting"):
            self.assertIn(expected, guidance)

    def test_incomplete_score_anchors_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RubricCriterion(
                criterion=Criterion.CLARITY,
                definition="Is the answer clear?",
                score_anchors={1: "Unclear", 5: "Clear"},
            )

    def test_incomplete_rubric_is_rejected(self) -> None:
        one_criterion = RubricCriterion(
            criterion=Criterion.CLARITY,
            definition="Is the answer clear?",
            score_anchors={
                1: "Very unclear",
                2: "Unclear",
                3: "Adequate",
                4: "Clear",
                5: "Very clear",
            },
        )

        with self.assertRaises(ValidationError):
            EvaluationRubric(
                name="incomplete",
                version="1.0.0",
                criteria=[one_criterion],
                judge_instructions=["Use the supplied evidence."],
            )

    def test_pairwise_example_requires_candidate_b(self) -> None:
        with self.assertRaises(ValidationError):
            RubricExample(
                example_id="invalid-pairwise",
                kind=ExampleKind.PAIRWISE_A_WINS,
                mode=EvaluationMode.PAIRWISE,
                reference_policy=ReferencePolicy.REQUIRED,
                question="Which answer is better?",
                reference_answer="The trusted answer.",
                candidate_a="Only one candidate was supplied.",
                expected_pairwise_decision=PairwiseDecision.A_WINS,
                explanation="This example must be rejected.",
            )

    def test_reference_free_example_rejects_a_reference(self) -> None:
        with self.assertRaises(ValidationError):
            RubricExample(
                example_id="invalid-reference-free",
                kind=ExampleKind.PAIRWISE_TIE,
                mode=EvaluationMode.PAIRWISE,
                reference_policy=ReferencePolicy.REFERENCE_FREE,
                question="Which answer is clearer?",
                reference_answer="A reference is forbidden in this mode.",
                candidate_a="Answer A",
                candidate_b="Answer B",
                expected_pairwise_decision=PairwiseDecision.TIE,
                explanation="This example must be rejected.",
            )


if __name__ == "__main__":
    unittest.main()
