import json
import unittest

from llm_judge.azure_client import RawJudgeResponse, TokenUsage
from llm_judge.contracts import (
    BinaryEvaluationResult,
    EvaluationInput,
    EvaluationMode,
    EvaluationResult,
    PairwiseEvaluationResult,
    ReferencePolicy,
)
from llm_judge.response_parser import (
    JudgeRefusalError,
    JudgeResponseValidationError,
    parse_judge_response,
)


def evaluation_input(mode: EvaluationMode) -> EvaluationInput:
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


def raw_response(mode: EvaluationMode, *, content=None, refusal=None) -> RawJudgeResponse:
    return RawJudgeResponse(
        deployment="gpt-5.6-terra",
        mode=mode,
        rubric_version="2.0.0",
        content=content,
        refusal=refusal,
        usage=TokenUsage(),
    )


class ResponseParserTests(unittest.TestCase):
    def test_parses_binary_result(self) -> None:
        case = evaluation_input(EvaluationMode.BINARY)
        response = raw_response(
            EvaluationMode.BINARY,
            content=json.dumps(
                {"case_id": "case-001", "decision": "PASS", "evidence": "Correct."}
            ),
        )

        result = parse_judge_response(response, case)

        self.assertIsInstance(result, BinaryEvaluationResult)
        self.assertEqual(result.decision.value, "PASS")

    def test_parses_pairwise_result(self) -> None:
        case = evaluation_input(EvaluationMode.PAIRWISE)
        response = raw_response(
            EvaluationMode.PAIRWISE,
            content=json.dumps(
                {
                    "case_id": "case-001",
                    "decision": "A_WINS",
                    "evidence": "A matches the reference.",
                }
            ),
        )

        result = parse_judge_response(response, case)

        self.assertIsInstance(result, PairwiseEvaluationResult)

    def test_parses_score_result_and_computes_decision(self) -> None:
        case = evaluation_input(EvaluationMode.SCORE)
        scores = [
            {"criterion": name, "score": 5, "evidence": "Meets the criterion."}
            for name in ("correctness", "relevance", "completeness", "clarity")
        ]
        response = raw_response(
            EvaluationMode.SCORE,
            content=json.dumps(
                {"case_id": "case-001", "scores": scores, "summary": "Correct answer."}
            ),
        )

        result = parse_judge_response(response, case)

        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.decision.value, "PASS")
        self.assertEqual(result.weighted_score, 5.0)

    def test_rejects_refusal(self) -> None:
        case = evaluation_input(EvaluationMode.BINARY)
        response = raw_response(EvaluationMode.BINARY, refusal="Cannot evaluate.")

        with self.assertRaises(JudgeRefusalError):
            parse_judge_response(response, case)

    def test_rejects_invalid_json_or_fields(self) -> None:
        case = evaluation_input(EvaluationMode.BINARY)

        for content in (
            "not JSON",
            json.dumps({"case_id": "case-001", "decision": "MAYBE", "evidence": "x"}),
            json.dumps({"case_id": "case-001", "decision": "PASS"}),
            json.dumps(
                {
                    "case_id": "case-001",
                    "decision": "PASS",
                    "evidence": "x",
                    "unexpected": True,
                }
            ),
        ):
            with self.subTest(content=content):
                response = raw_response(EvaluationMode.BINARY, content=content)
                with self.assertRaises(JudgeResponseValidationError):
                    parse_judge_response(response, case)

    def test_rejects_case_id_and_mode_mismatches(self) -> None:
        case = evaluation_input(EvaluationMode.BINARY)
        wrong_id = raw_response(
            EvaluationMode.BINARY,
            content=json.dumps(
                {"case_id": "other-case", "decision": "PASS", "evidence": "x"}
            ),
        )
        wrong_mode = raw_response(
            EvaluationMode.PAIRWISE,
            content=json.dumps(
                {"case_id": "case-001", "decision": "A_WINS", "evidence": "x"}
            ),
        )

        with self.assertRaisesRegex(JudgeResponseValidationError, "case_id"):
            parse_judge_response(wrong_id, case)
        with self.assertRaisesRegex(JudgeResponseValidationError, "mode"):
            parse_judge_response(wrong_mode, case)


if __name__ == "__main__":
    unittest.main()
