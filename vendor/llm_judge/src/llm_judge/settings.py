"""Step 5 configuration loaded from environment variables."""

import os
from typing import Mapping, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class SettingsError(ValueError):
    """Raised when required Azure judge configuration is missing or invalid."""


class AzureJudgeSettings(BaseModel):
    """Validated Azure OpenAI connection and request settings.

    API keys use ``SecretStr`` so normal repr/logging does not expose the value.
    This class reads process environment variables but never reads or writes a
    local `.env` file automatically.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint: str = Field(min_length=1)
    api_key: SecretStr
    # Azure still needs a model/deployment name on every request. Terra is the
    # project default, so it does not have to be repeated in the local `.env`.
    # AZURE_OPENAI_DEPLOYMENT can override this when testing another model.
    deployment: str = Field(default="gpt-5.6-terra", min_length=1)
    api_version: str = Field(min_length=1)
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_output_tokens: int = Field(default=1200, ge=100, le=100_000)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        """Require an HTTP(S) endpoint and normalize its trailing slash."""

        normalized = value.strip().rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Azure endpoint must be a valid HTTP(S) URL")
        return normalized

    @field_validator("deployment", "api_version")
    @classmethod
    def strip_non_secret_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Configuration value cannot be blank")
        return normalized

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "AzureJudgeSettings":
        """Build settings from environment variables with clear missing-key errors."""

        source = os.environ if environ is None else environ
        required_names = {
            "endpoint": "AZURE_OPENAI_ENDPOINT",
            "api_key": "AZURE_OPENAI_API_KEY",
            "api_version": "AZURE_OPENAI_API_VERSION",
        }
        missing = [name for name in required_names.values() if not source.get(name)]
        if missing:
            raise SettingsError(
                "Missing required environment variables: " + ", ".join(sorted(missing))
            )

        values = {
            field: source[environment_name]
            for field, environment_name in required_names.items()
        }
        optional_names = {
            "deployment": "AZURE_OPENAI_DEPLOYMENT",
            "timeout_seconds": "AZURE_OPENAI_TIMEOUT_SECONDS",
            "max_retries": "AZURE_OPENAI_MAX_RETRIES",
            "max_output_tokens": "AZURE_OPENAI_MAX_OUTPUT_TOKENS",
            "temperature": "AZURE_OPENAI_TEMPERATURE",
        }
        for field, environment_name in optional_names.items():
            value = source.get(environment_name)
            if value not in (None, ""):
                values[field] = value
        return cls.model_validate(values)
