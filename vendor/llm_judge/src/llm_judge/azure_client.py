"""Step 5 Azure OpenAI transport for judge prompts.

This module deliberately returns raw model content. Step 6 will parse that content
into the mode-specific Pydantic result and handle refusals or malformed JSON.
"""

from copy import deepcopy
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import EvaluationMode
from .prompt_builder import JudgePrompt
from .settings import AzureJudgeSettings


class JudgeClientError(RuntimeError):
    """A safe, provider-independent error raised by the judge transport."""


class TokenUsage(BaseModel):
    """Token counts returned by Azure when usage metadata is available."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)


class RawJudgeResponse(BaseModel):
    """Auditable provider response before Step 6 output validation."""

    model_config = ConfigDict(frozen=True)

    response_id: Optional[str] = None
    request_id: Optional[str] = None
    deployment: str
    model: Optional[str] = None
    created: Optional[int] = None
    mode: EvaluationMode
    rubric_version: str
    content: Optional[str] = None
    refusal: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: TokenUsage

    @model_validator(mode="after")
    def require_content_or_refusal(self) -> "RawJudgeResponse":
        if not self.content and not self.refusal:
            raise ValueError("Azure response contained neither content nor refusal")
        return self


def _make_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add strict-object constraints required by Structured Outputs."""

    strict_schema = deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(strict_schema)
    return strict_schema


class AzureJudgeClient:
    """Send a ``JudgePrompt`` through an Azure OpenAI-compatible SDK client.

    ``client`` is injectable so tests can use a fake object without network calls,
    credentials, or token cost.
    """

    def __init__(
        self,
        settings: AzureJudgeSettings,
        client: Optional[Any] = None,
    ) -> None:
        self.settings = settings
        self._client = client if client is not None else self._create_sdk_client()

    def _create_sdk_client(self) -> Any:
        try:
            from openai import AzureOpenAI
        except ImportError as error:
            raise JudgeClientError(
                "The 'openai' package is required; install the project dependencies"
            ) from error

        return AzureOpenAI(
            api_key=self.settings.api_key.get_secret_value(),
            api_version=self.settings.api_version,
            azure_endpoint=self.settings.endpoint,
            timeout=self.settings.timeout_seconds,
            max_retries=self.settings.max_retries,
        )

    def build_request(self, prompt: JudgePrompt) -> Dict[str, Any]:
        """Create the exact SDK request without performing network I/O."""

        schema_name = (
            f"judge_{prompt.mode.value}_v{prompt.rubric_version.replace('.', '_')}"
        )
        request: Dict[str, Any] = {
            "model": self.settings.deployment,
            "messages": prompt.as_api_messages(),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": _make_strict_json_schema(prompt.response_schema),
                },
            },
            "max_completion_tokens": self.settings.max_output_tokens,
        }
        if self.settings.temperature is not None:
            request["temperature"] = self.settings.temperature
        return request

    def evaluate(self, prompt: JudgePrompt) -> RawJudgeResponse:
        """Call Azure and return raw content plus reproducibility metadata."""

        try:
            completion = self._client.chat.completions.create(
                **self.build_request(prompt)
            )
        except Exception as error:
            # Avoid including credentials or a potentially sensitive request body
            # in the public exception message. The chained exception remains
            # available to controlled application logging.
            raise JudgeClientError(
                f"Azure judge request failed ({type(error).__name__})"
            ) from error

        choices = getattr(completion, "choices", None)
        if not choices:
            raise JudgeClientError("Azure judge response contained no choices")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise JudgeClientError("Azure judge response contained no message")

        usage = getattr(completion, "usage", None)
        return RawJudgeResponse(
            response_id=getattr(completion, "id", None),
            request_id=getattr(completion, "_request_id", None),
            deployment=self.settings.deployment,
            model=getattr(completion, "model", None),
            created=getattr(completion, "created", None),
            mode=prompt.mode,
            rubric_version=prompt.rubric_version,
            content=getattr(message, "content", None),
            refusal=getattr(message, "refusal", None),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            ),
        )
