"""A generic tool-using agent: one model, the suite's tools, a plain loop.

Use it when the thing under test is the TOOLS - their names, descriptions and
schemas - rather than a particular agent's code. Point it at any model:

    ef run suites/<suite> --adapter inprocess --target agent.tool_agent:run_agent \
        --model bedrock:anthropic.claude-sonnet-5

The system prompt comes from the case's session context (`system_prompt`),
so a suite can mirror the host it ships in (Claude Desktop, Copilot, ...).
Every other session key (customer_id, ...) is told to the model as the
signed-in user's details.
"""

from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable

from evalkit.providers import add_usage, get_provider
from evalkit.schema.trajectory import Usage

DEFAULT_SYSTEM = "You are a helpful assistant. Use the tools when they help."

ToolCaller = Callable[[str, dict], Awaitable[Any]]


def build_system(context: dict | None) -> str:
    """The system prompt, plus who is signed in.

    In production the host knows the logged-in user and passes it along; the
    user never types their own id. Without this, every model under test
    spends its first turn asking "what is your customer ID?" - and the eval
    measures the missing session, not the model.
    """
    context = dict(context or {})
    system = context.pop("system_prompt", DEFAULT_SYSTEM)
    if not context:
        return system
    facts = "\n".join(f"- {k}: {v}" for k, v in context.items())
    return (f"{system}\n\nThe user is signed in. Session:\n{facts}\n"
            f"Use these values whenever a tool needs them. Never ask the user "
            f"for them.")


async def run_agent(messages: list[dict], tool_specs: list, call_tool: ToolCaller,
                    context: dict | None = None, max_steps: int = 10,
                    model: str | None = None) -> dict:
    """One agent turn: call tools until the model answers in text.

    Returns the reply AND the whole conversation, tool calls included, so the
    next turn of a multi-turn case carries on from here.
    """
    provider = get_provider(model or os.environ.get("AGENT_MODEL") or "openai:")
    system = build_system(context)
    convo = list(messages)
    usage = Usage()

    for _ in range(max_steps):
        turn = await provider.complete(convo, system=system, tools=tool_specs)
        usage = add_usage(usage, turn.usage)
        convo.append(turn.as_message())

        if not turn.tool_calls:
            return {"output": turn.text, "messages": convo,
                    "usage": usage.model_dump(), "stop_reason": turn.stop_reason}

        for tc in turn.tool_calls:
            # The harness intercepts here: it runs the tool AND records it.
            result = await call_tool(tc.name, tc.arguments)
            convo.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                          "content": json.dumps(result, default=str),
                          "is_error": isinstance(result, dict) and "error" in result
                                      and len(result) == 1})

    give_up = "I wasn't able to complete that within the allowed number of steps."
    convo.append({"role": "assistant", "content": give_up})
    return {"output": give_up, "messages": convo, "usage": usage.model_dump(),
            "stop_reason": "max_steps"}
