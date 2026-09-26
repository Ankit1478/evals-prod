"""A Provider is one way of reaching a model: OpenAI, Azure, Claude, Bedrock.

Everything above this layer - the simulated user, a generic agent, a judge -
speaks ONE message format and never knows which vendor answered. Swapping
Bedrock for Azure must be a flag, not a code change.

The shared message format is a list of plain dicts:

    {"role": "user",      "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [...], "raw": ...}
    {"role": "tool",      "tool_call_id": "...", "name": "...",
                          "content": "...", "is_error": False}

`raw` is the vendor's own copy of an assistant turn. A provider replays it
unchanged when the same provider sees that turn again - Claude, for one,
rejects a conversation whose earlier turns were rebuilt differently from what
it actually sent.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from evalkit.schema.case import Frozen
from evalkit.schema.trajectory import Usage


class ProviderError(RuntimeError):
    """A provider was asked for something it cannot do (bad spec, missing
    setting). Never raised for a model's answer - a refusal is a result."""


class ToolDef(Protocol):
    """Anything with these three attributes can be offered as a tool.
    env.base.ToolSpec is the usual one."""
    name: str
    description: str
    input_schema: dict


class ProviderToolCall(Frozen):
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""       # the text before parsing, for malformed-args grading


class Completion(Frozen):
    """One model turn, in the shared shape."""
    text: str = ""
    tool_calls: list[ProviderToolCall] = []
    stop_reason: str = "end_turn"   # end_turn | tool_use | max_tokens | refusal | ...
    model: str = ""
    usage: Usage = Usage()
    raw: Any = None                 # vendor payload, replayed on the next turn
    provider: str = ""

    def as_message(self) -> dict:
        """This turn as an assistant message for the next request."""
        return {
            "role": "assistant",
            "content": self.text,
            "tool_calls": [c.model_dump() for c in self.tool_calls],
            "raw": self.raw,
            "provider": self.provider,
        }


@runtime_checkable
class Provider(Protocol):
    name: str      # "anthropic", "bedrock", "foundry", "openai", "azure"
    model: str

    async def complete(self, messages: list[dict], *, system: str = "",
                       tools: Sequence[ToolDef] | None = None,
                       max_tokens: int = 16000) -> Completion: ...

    def describe(self) -> dict:
        """Goes into the run manifest. Never includes a credential."""
        ...

    async def aclose(self) -> None:
        """Release the HTTP client. Safe to call more than once."""
        ...


def clean_schema(schema: dict) -> dict:
    """Drop keys some vendors reject on a tool's input schema.

    JSON Schema exported by zod-to-json-schema carries a top-level `$schema`
    URL. It says nothing about the arguments, and not every API accepts it.
    """
    return {k: v for k, v in schema.items() if k != "$schema"}


def add_usage(total: Usage, more: Usage) -> Usage:
    return Usage(
        input_tokens=total.input_tokens + more.input_tokens,
        output_tokens=total.output_tokens + more.output_tokens,
        cached_input_tokens=total.cached_input_tokens + more.cached_input_tokens,
        cost_usd=total.cost_usd + more.cost_usd,
    )
