import unittest

from pydantic import ValidationError

from llm_judge.settings import AzureJudgeSettings, SettingsError


REQUIRED_ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://judge-resource.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "test-secret-key",
    "AZURE_OPENAI_DEPLOYMENT": "judge-deployment",
    "AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
}


class SettingsTests(unittest.TestCase):
    def test_loads_required_and_optional_environment_values(self) -> None:
        environment = {
            **REQUIRED_ENV,
            "AZURE_OPENAI_TIMEOUT_SECONDS": "45",
            "AZURE_OPENAI_MAX_RETRIES": "3",
            "AZURE_OPENAI_MAX_OUTPUT_TOKENS": "1600",
            "AZURE_OPENAI_TEMPERATURE": "0",
        }

        settings = AzureJudgeSettings.from_env(environment)

        self.assertEqual(settings.endpoint, "https://judge-resource.openai.azure.com")
        self.assertEqual(settings.timeout_seconds, 45)
        self.assertEqual(settings.max_retries, 3)
        self.assertEqual(settings.max_output_tokens, 1600)
        self.assertEqual(settings.temperature, 0)

    def test_missing_required_environment_values_are_reported_together(self) -> None:
        with self.assertRaisesRegex(SettingsError, "AZURE_OPENAI_API_KEY"):
            AzureJudgeSettings.from_env({"AZURE_OPENAI_ENDPOINT": "https://example.com"})

    def test_api_key_is_masked_in_normal_representation(self) -> None:
        settings = AzureJudgeSettings.from_env(REQUIRED_ENV)

        self.assertNotIn("test-secret-key", repr(settings))
        self.assertEqual(settings.api_key.get_secret_value(), "test-secret-key")

    def test_terra_is_the_default_deployment(self) -> None:
        environment = {
            key: value
            for key, value in REQUIRED_ENV.items()
            if key != "AZURE_OPENAI_DEPLOYMENT"
        }

        settings = AzureJudgeSettings.from_env(environment)

        self.assertEqual(settings.deployment, "gpt-5.6-terra")

    def test_invalid_endpoint_is_rejected(self) -> None:
        environment = {**REQUIRED_ENV, "AZURE_OPENAI_ENDPOINT": "not-a-url"}

        with self.assertRaises(ValidationError):
            AzureJudgeSettings.from_env(environment)


if __name__ == "__main__":
    unittest.main()
