"""Claude, reached three ways: the Claude API, Amazon Bedrock, Microsoft Foundry.

All three serve the same Messages API, so one class covers them; only the
client differs. Each client reads its own settings from the environment:

    anthropic  ANTHROPIC_API_KEY (or an `ant auth login` profile)
    bedrock    AWS_REGION + the usual AWS credentials (AWS_PROFILE, access
               keys, or AWS_BEARER_TOKEN_BEDROCK). Model ids carry an
               `anthropic.` prefix: anthropic.claude-opus-5
    foundry    ANTHROPIC_FOUNDRY_API_KEY + ANTHROPIC_FOUNDRY_RESOURCE (or
               ANTHROPIC_FOUNDRY_BASE_URL)

Deliberately NOT enabled: refusal fallbacks. A fallback quietly answers with
a different model, and an eval that swaps the model under test mid-run is
measuring something else. A refusal is recorded as stop_reason="refusal".
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from evalkit.providers.base import (Completion, ProviderError, ProviderToolCall,
                                    ToolDef, clean_schema)
from evalkit.schema.trajectory import Usage

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "bedrock": "anthropic.claude-opus-5",
    "foundry": "claude-opus-5",
}

EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def _make_client(kind: str) -> Any:
    import anthropic

    if kind == "anthropic":
        return anthropic.AsyncAnthropic(max_retries=4)
    if kind == "bedrock":
        return anthropic.AsyncAnthropicBedrockMantle(max_retries=4)
    if kind == "foundry":
        return anthropic.AsyncAnthropicFoundry(max_retries=4)
    raise ProviderError(f"unknown Claude platform: {kind}")


def to_anthropic_messages(messages: list[dict], provider: str) -> list[dict]:
    """Shared format -> Messages API format.

    Tool results that arrive back to back go into ONE user message. Splitting
    them teaches the model to stop calling tools in parallel.
    """
    out: list[dict] = []
    pending: list[dict] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": list(pending)})
            pending.clear()

    for m in messages:
        role = m["role"]
        if role == "tool":
            pending.append({"type": "tool_result",
                            "tool_use_id": m["tool_call_id"],
                            "content": m.get("content") or "",
                            "is_error": bool(m.get("is_error"))})
            continue
        flush()
        if role == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            if m.get("raw") is not None and m.get("provider") == provider:
                # Replay exactly what the model sent, thinking blocks included.
                out.append({"role": "assistant", "content": m["raw"]})
                continue
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": c["id"],
                               "name": c["name"], "input": c.get("arguments") or {}})
            out.append({"role": "assistant", "content": blocks or ""})
        elif role == "system":
            raise ProviderError("pass system text via `system=`, not as a message")
    flush()
    return out


def to_anthropic_tools(tools: Sequence[ToolDef]) -> list[dict]:
    return [{"name": t.name, "description": t.description,
             "input_schema": clean_schema(t.input_schema)} for t in tools]


def parse_response(resp: Any, provider: str) -> Completion:
    text = "".join(b.text for b in resp.content if b.type == "text")
    calls = [ProviderToolCall(id=b.id, name=b.name, arguments=dict(b.input or {}),
                              raw_arguments=json.dumps(b.input or {}))
             for b in resp.content if b.type == "tool_use"]
    u = resp.usage
    return Completion(
        text=text,
        tool_calls=calls,
        stop_reason=resp.stop_reason or "end_turn",
        model=resp.model,
        usage=Usage(input_tokens=u.input_tokens or 0,
                    output_tokens=u.output_tokens or 0,
                    cached_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0),
        raw=[b.model_dump(mode="json", exclude_none=True) for b in resp.content],
        provider=provider,
    )


class AnthropicProvider:
    """name is the platform: anthropic, bedrock or foundry."""

    def __init__(self, name: str = "anthropic", model: str | None = None,
                 effort: str | None = None, client: Any = None) -> None:
        if name not in DEFAULT_MODELS:
            raise ProviderError(f"unknown Claude platform: {name}")
        if effort is not None and effort not in EFFORTS:
            raise ProviderError(f"effort must be one of {sorted(EFFORTS)}, got {effort!r}")
        self.name = name
        self.model = model or DEFAULT_MODELS[name]
        self.effort = effort
        self._client = client         # built on first use, so specs parse without credentials

    async def complete(self, messages: list[dict], *, system: str = "",
                       tools: Sequence[ToolDef] | None = None,
                       max_tokens: int = 16000) -> Completion:
        if self._client is None:
            self._client = _make_client(self.name)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": to_anthropic_messages(messages, self.name),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = await self._client.messages.create(**kwargs)
        return parse_response(resp, self.name)

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model,
                "effort": self.effort or "default"}
