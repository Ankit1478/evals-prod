import json
import unittest

from llm_judge.azure_client import RawJudgeResponse, TokenUsage
from llm_judge.contracts import EvaluationInput, EvaluationMode, ReferencePolicy
from llm_judge.multi_judge import JudgeModel, TwoModelJudge
from llm_judge.settings import AzureJudgeSettings


def settings(deployment: str) -> AzureJudgeSettings:
    return AzureJudgeSettings(
        endpoint="https://judge-resource.openai.azure.com",
        api_key="test-secret",
        deployment=deployment,
        api_version="2025-01-01-preview",
    )


class FakeJudgeClient:
    def __init__(self, deployment: str, content: dict):
        self.settings = settings(deployment)
        self.content = json.dumps(content)
        self.prompts = []

    def evaluate(self, prompt):
        self.prompts.append(prompt)
        return RawJudgeResponse(
            deployment=self.settings.deployment,
            mode=prompt.mode,
            rubric_version=prompt.rubric_version,
            content=self.content,
            usage=TokenUsage(),
        )


def case(mode: EvaluationMode) -> EvaluationInput:
    values = {
        "case_id": "case-001",
        "mode": mode,
        "reference_policy": ReferencePolicy.REQUIRED,
        "question": "What is 2 + 2?",
        "reference_answer": "4",
        "candidate_answer": "4",
    }
    if mode == EvaluationMode.PAIRWISE:
        values["candidate_b"] = "5"
    return EvaluationInput(**values)


def binary_content(decision: str) -> dict:
    return {"case_id": "case-001", "decision": decision, "evidence": "Evidence."}


def score_content(scores: list) -> dict:
    criteria = ("correctness", "relevance", "completeness", "clarity")
    return {
        "case_id": "case-001",
        "scores": [
            {"criterion": criterion, "score": score, "evidence": "Evidence."}
            for criterion, score in zip(criteria, scores)
        ],
        "summary": "Summary.",
    }


class TwoModelJudgeTests(unittest.TestCase):
    def test_binary_agreement_produces_consensus(self) -> None:
        terra = FakeJudgeClient(JudgeModel.TERRA.value, binary_content("PASS"))
        luna = FakeJudgeClient(JudgeModel.LUNA.value, binary_content("PASS"))
        judge = TwoModelJudge(terra, luna)

        result = judge.evaluate(case(EvaluationMode.BINARY), include_examples=False)

        self.assertTrue(result.agreement)
        self.assertEqual(result.aggregate_decision, "PASS")
        self.assertFalse(result.requires_human_review)
        self.assertEqual(
            [item.model for item in result.judgments],
            [JudgeModel.TERRA, JudgeModel.LUNA],
        )
        self.assertEqual(terra.prompts[0], luna.prompts[0])

    def test_binary_disagreement_requires_human_review(self) -> None:
        judge = TwoModelJudge(
            FakeJudgeClient(JudgeModel.TERRA.value, binary_content("PASS")),
            FakeJudgeClient(JudgeModel.LUNA.value, binary_content("FAIL")),
        )

        result = judge.evaluate(case(EvaluationMode.BINARY))

        self.assertFalse(result.agreement)
        self.assertIsNone(result.aggregate_decision)
        self.assertTrue(result.requires_human_review)

    def test_pairwise_agreement_is_preserved(self) -> None:
        content = {
            "case_id": "case-001",
            "decision": "A_WINS",
            "evidence": "A matches the reference.",
        }
        judge = TwoModelJudge(
            FakeJudgeClient(JudgeModel.TERRA.value, content),
            FakeJudgeClient(JudgeModel.LUNA.value, content),
        )

        result = judge.evaluate(case(EvaluationMode.PAIRWISE))

        self.assertEqual(result.aggregate_decision, "A_WINS")
        self.assertFalse(result.requires_human_review)

    def test_score_mode_averages_dimensions_and_flags_decision_disagreement(self) -> None:
        judge = TwoModelJudge(
            FakeJudgeClient(JudgeModel.TERRA.value, score_content([5, 5, 5, 5])),
            FakeJudgeClient(JudgeModel.LUNA.value, score_content([2, 2, 4, 4])),
        )

        result = judge.evaluate(case(EvaluationMode.SCORE))

        self.assertEqual(result.average_scores["correctness"], 3.5)
        # The rubric weights correctness/relevance more heavily, producing 3.85.
        self.assertEqual(result.average_weighted_score, 3.85)
        self.assertEqual(result.aggregate_decision, "PASS")
        self.assertFalse(result.agreement)
        self.assertTrue(result.requires_human_review)

    def test_clients_must_use_the_expected_deployments(self) -> None:
        wrong_terra = FakeJudgeClient("another-model", binary_content("PASS"))
        luna = FakeJudgeClient(JudgeModel.LUNA.value, binary_content("PASS"))

        with self.assertRaisesRegex(ValueError, "gpt-5.6-terra"):
            TwoModelJudge(wrong_terra, luna)

    def test_combined_result_requires_one_judgment_per_model(self) -> None:
        terra = FakeJudgeClient(JudgeModel.TERRA.value, binary_content("PASS"))
        luna = FakeJudgeClient(JudgeModel.LUNA.value, binary_content("PASS"))
        result = TwoModelJudge(terra, luna).evaluate(case(EvaluationMode.BINARY))
        payload = result.model_dump()
        payload["judgments"][1]["model"] = JudgeModel.TERRA.value

        with self.assertRaisesRegex(ValueError, "one Terra and one Luna"):
            type(result).model_validate(payload)


if __name__ == "__main__":
    unittest.main()
