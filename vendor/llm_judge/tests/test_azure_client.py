import json
import unittest
from types import SimpleNamespace

from llm_judge.azure_client import AzureJudgeClient, JudgeClientError
from llm_judge.contracts import EvaluationInput, EvaluationMode, ReferencePolicy
from llm_judge.prompt_builder import build_judge_prompt
from llm_judge.settings import AzureJudgeSettings


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.response


class FakeAzureClient:
    def __init__(self, response=None, error=None):
        self.completions = FakeCompletions(response=response, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def settings(**changes):
    values = {
        "endpoint": "https://judge-resource.openai.azure.com",
        "api_key": "test-secret-key",
        "deployment": "judge-deployment",
        "api_version": "2025-01-01-preview",
        "timeout_seconds": 30,
        "max_retries": 2,
        "max_output_tokens": 1200,
    }
    values.update(changes)
    return AzureJudgeSettings(**values)


def score_prompt():
    return build_judge_prompt(
        EvaluationInput(
            case_id="score-001",
            mode=EvaluationMode.SCORE,
            reference_policy=ReferencePolicy.REQUIRED,
            question="What is 2 + 2?",
            reference_answer="4",
            candidate_answer="4",
        )
    )


def successful_response(content=None, refusal=None):
    return SimpleNamespace(
        id="completion-001",
        _request_id="request-001",
        model="resolved-judge-model",
        created=1_788_200_000,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, refusal=refusal),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=500,
            completion_tokens=100,
            total_tokens=600,
        ),
    )


class AzureClientTests(unittest.TestCase):
    def test_build_request_uses_deployment_messages_and_strict_schema(self) -> None:
        client = AzureJudgeClient(settings(), client=FakeAzureClient())

        request = client.build_request(score_prompt())

        self.assertEqual(request["model"], "judge-deployment")
        self.assertEqual([item["role"] for item in request["messages"]], ["system", "user"])
        self.assertEqual(request["response_format"]["type"], "json_schema")
        json_schema = request["response_format"]["json_schema"]
        self.assertTrue(json_schema["strict"])
        self.assertFalse(json_schema["schema"]["additionalProperties"])
        self.assertEqual(request["max_completion_tokens"], 1200)
        self.assertNotIn("temperature", request)

    def test_optional_temperature_is_sent_only_when_configured(self) -> None:
        client = AzureJudgeClient(settings(temperature=0), client=FakeAzureClient())

        request = client.build_request(score_prompt())

        self.assertEqual(request["temperature"], 0)

    def test_evaluate_returns_raw_content_and_usage_metadata(self) -> None:
        content = json.dumps(
            {
                "case_id": "score-001",
                "scores": [],
                "summary": "Raw content is parsed in Step 6.",
            }
        )
        fake = FakeAzureClient(response=successful_response(content=content))
        client = AzureJudgeClient(settings(), client=fake)

        result = client.evaluate(score_prompt())

        self.assertEqual(result.content, content)
        self.assertEqual(result.request_id, "request-001")
        self.assertEqual(result.deployment, "judge-deployment")
        self.assertEqual(result.rubric_version, "2.0.0")
        self.assertEqual(result.usage.total_tokens, 600)
        self.assertEqual(len(fake.completions.requests), 1)

    def test_refusal_is_preserved_without_fake_content(self) -> None:
        fake = FakeAzureClient(response=successful_response(refusal="Cannot evaluate."))
        client = AzureJudgeClient(settings(), client=fake)

        result = client.evaluate(score_prompt())

        self.assertIsNone(result.content)
        self.assertEqual(result.refusal, "Cannot evaluate.")

    def test_missing_choices_raise_safe_error(self) -> None:
        response = SimpleNamespace(choices=[])
        client = AzureJudgeClient(settings(), client=FakeAzureClient(response=response))

        with self.assertRaisesRegex(JudgeClientError, "no choices"):
            client.evaluate(score_prompt())

    def test_sdk_error_is_wrapped_without_request_or_secret(self) -> None:
        fake = FakeAzureClient(error=RuntimeError("provider failed: test-secret-key"))
        client = AzureJudgeClient(settings(), client=fake)

        with self.assertRaises(JudgeClientError) as raised:
            client.evaluate(score_prompt())

        self.assertNotIn("test-secret-key", str(raised.exception))
        self.assertNotIn("What is 2 + 2?", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
