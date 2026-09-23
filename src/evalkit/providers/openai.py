"""OpenAI models, reached directly or through Azure OpenAI.

    openai  OPENAI_API_KEY
    azure   AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_VERSION.
            The model is the DEPLOYMENT name (AZURE_OPENAI_DEPLOYMENT by default).

These are the same variables the vendored judge already reads, so one .env
serves both.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from evalkit.providers.base import (Completion, ProviderError, ProviderToolCall,
                                    ToolDef, clean_schema)
from evalkit.schema.trajectory import Usage


def _make_client(kind: str) -> Any:
    import openai

    if kind == "openai":
        return openai.AsyncOpenAI(max_retries=4)
    if kind == "azure":
        missing = [v for v in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                               "AZURE_OPENAI_API_VERSION") if not os.environ.get(v)]
        if missing:
            raise ProviderError(f"azure provider needs {', '.join(missing)}")
        return openai.AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            max_retries=4,
        )
    raise ProviderError(f"unknown OpenAI platform: {kind}")


def to_openai_messages(messages: list[dict], system: str, provider: str) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            if m.get("raw") is not None and m.get("provider") == provider:
                out.append(m["raw"])
                continue
            msg: dict = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"],
                                  "arguments": json.dumps(c.get("arguments") or {})}}
                    for c in m["tool_calls"]]
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                        "content": m.get("content") or ""})
        elif role == "system":
            raise ProviderError("pass system text via `system=`, not as a message")
    return out


def to_openai_tools(tools: Sequence[ToolDef]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description,
                          "parameters": clean_schema(t.input_schema)}}
            for t in tools]


def parse_response(resp: Any, provider: str) -> Completion:
    choice = resp.choices[0]
    msg = choice.message
    calls: list[ProviderToolCall] = []
    for tc in msg.tool_calls or []:
        raw_args = tc.function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}                 # kept as raw_arguments, so a grader can see it
        calls.append(ProviderToolCall(id=tc.id, name=tc.function.name,
                                      arguments=args if isinstance(args, dict) else {},
                                      raw_arguments=raw_args))
    reason = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens",
              "content_filter": "refusal"}.get(choice.finish_reason or "stop",
                                               choice.finish_reason or "end_turn")
    u = resp.usage
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) if u else 0
    return Completion(
        text=msg.content or "",
        tool_calls=calls,
        stop_reason=reason,
        model=resp.model or "",
        usage=Usage(input_tokens=(u.prompt_tokens if u else 0) or 0,
                    output_tokens=(u.completion_tokens if u else 0) or 0,
                    cached_input_tokens=cached or 0),
        raw=msg.model_dump(exclude_none=True),
        provider=provider,
    )


class OpenAIProvider:
    """name is the platform: openai or azure."""

    def __init__(self, name: str = "openai", model: str | None = None,
                 client: Any = None) -> None:
        if name not in ("openai", "azure"):
            raise ProviderError(f"unknown OpenAI platform: {name}")
        if model is None:
            # A bare "openai:" spec. Which model each ROLE uses is decided
            # in .env (AGENT_MODEL, USER_MODEL, ...), not here.
            model = ("gpt-4o-mini" if name == "openai"
                     else os.environ.get("AZURE_OPENAI_DEPLOYMENT"))
        if not model:
            raise ProviderError("azure provider needs a deployment: azure:<deployment> "
                                "or AZURE_OPENAI_DEPLOYMENT")
        self.name = name
        self.model = model
        self._client = client
        # Set only after the model itself demands it - see complete().
        self._reasoning_effort: str | None = None

    async def complete(self, messages: list[dict], *, system: str = "",
                       tools: Sequence[ToolDef] | None = None,
                       max_tokens: int = 16000) -> Completion:
        import openai

        if self._client is None:
            self._client = _make_client(self.name)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages, system, self.name),
            "max_completion_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            # Reasoning models (gpt-5.x) refuse function tools on
            # chat.completions unless reasoning_effort is "none", and say so
            # in the error. Retry once with it, and keep it for this run.
            # Sending it up front would break models that reject the
            # parameter altogether (gpt-4o-mini).
            if not tools or self._reasoning_effort or "reasoning_effort" not in str(e):
                raise
            self._reasoning_effort = "none"
            kwargs["reasoning_effort"] = "none"
            resp = await self._client.chat.completions.create(**kwargs)
        return parse_response(resp, self.name)

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model,
                "reasoning_effort": self._reasoning_effort or "default"}
