"""Let the vendored LLM_AS_JUDGE run on Claude as well as OpenAI.

The vendored AzureJudgeClient makes exactly one call -
`client.chat.completions.create(**request)` - and reads the OpenAI response
shape back. ClaudeChatClient answers that one call through the Claude
Messages API (Claude API, Bedrock or Foundry), so the vendored code runs
unchanged on any of them.

The judge's JSON schema becomes the input schema of one tool the model is
forced to call (`tool_choice`), and the tool's input is handed back as the
reply text - JSON the vendored parser can read. Forced tool use works on
the Claude API, Bedrock and Foundry alike; `output_config.format` does not
(Bedrock rejects it).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

VERDICT_TOOL = "submit_verdict"


def _sync_client(platform: str) -> Any:
    """Sync, because the vendored judge is sync (it runs in a worker thread).
    Credentials come from the environment, same as providers/anthropic.py."""
    import anthropic

    if platform == "anthropic":
        return anthropic.Anthropic(max_retries=4)
    if platform == "bedrock":
        return anthropic.AnthropicBedrockMantle(max_retries=4)
    if platform == "foundry":
        return anthropic.AnthropicFoundry(max_retries=4)
    raise ValueError(f"not a Claude platform: {platform}")


class ClaudeChatClient:
    """Duck-types `.chat.completions.create` over the Claude Messages API."""

    def __init__(self, platform: str, client: Any = None):
        self.platform = platform
        self._client = client          # built on first call: no credentials needed to construct
        self.chat = self
        self.completions = self

    def create(self, *, model: str, messages: list[dict],
               response_format: dict | None = None,
               max_completion_tokens: int | None = None,
               temperature: float | None = None, **_ignored: Any) -> Any:
        if self._client is None:
            self._client = _sync_client(self.platform)

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_completion_tokens or 4096,
            "messages": [{"role": m["role"], "content": m["content"]}
                         for m in messages if m["role"] != "system"],
        }
        if system:
            kwargs["system"] = system
        if response_format and response_format.get("type") == "json_schema":
            kwargs["tools"] = [{"name": VERDICT_TOOL,
                                "description": "Submit the evaluation result.",
                                "input_schema": response_format["json_schema"]["schema"]}]
            kwargs["tool_choice"] = {"type": "tool", "name": VERDICT_TOOL}
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as e:
            # Some Claude models do not take `temperature`. Drop it and retry
            # once rather than hardcoding which ones.
            if "temperature" not in kwargs or "temperature" not in str(e):
                raise
            kwargs.pop("temperature")
            resp = self._client.messages.create(**kwargs)
        return _as_chat_completion(resp)


def _as_chat_completion(resp: Any) -> Any:
    """The OpenAI response shape AzureJudgeClient reads."""
    verdict = next((b for b in resp.content if getattr(b, "type", None) == "tool_use"
                    and b.name == VERDICT_TOOL), None)
    text = (json.dumps(verdict.input) if verdict is not None else
            "".join(b.text for b in resp.content if getattr(b, "type", None) == "text"))
    refused = resp.stop_reason == "refusal"
    usage = resp.usage
    tokens_in = getattr(usage, "input_tokens", 0) or 0
    tokens_out = getattr(usage, "output_tokens", 0) or 0
    return SimpleNamespace(
        id=resp.id, model=resp.model, created=None,
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None if refused else text,
                                    refusal="the model refused" if refused else None),
            finish_reason=resp.stop_reason)],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out,
                              total_tokens=tokens_in + tokens_out),
    )
