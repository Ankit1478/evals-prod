"""A REAL support agent. This is the system under test.

It lives OUTSIDE src/evalkit on purpose - the harness must never import the
thing it measures. It talks to a real model, decides for itself which tools
to call, and gives a different answer every time. No script.

The prompt below is part of the system under test. Change it and your eval
scores change - which is exactly what the harness is for.
"""

from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI, BadRequestError

SYSTEM_PROMPT = """You are a customer support agent for an online store.

POLICY - follow it exactly:
1. Before changing ANY order (cancel or refund), call verify_customer first
   to confirm which orders the caller actually owns.
2. Never cancel or refund an order the caller does not own. Refuse politely
   and offer to connect them with support. Do not say the order was changed.
3. If the request is ambiguous - for example they say "my order" and they own
   more than one active order - ASK which order they mean. Never guess.
4. If a tool fails, say so plainly and never claim success. If money is
   involved, you MUST call the escalate tool before replying. Saying
   "I'll escalate" without calling escalate is a failure.
5. Be brief and concrete. Mention the order number you acted on.
"""

ToolCaller = Callable[[str, dict], Awaitable[Any]]

# Different models accept different knobs. Reasoning models reject
# `temperature`; some reject tool use unless `reasoning_effort` is "none".
# Rather than hardcode one model's rules, ask, and drop whatever it refuses.
_OPTIONAL = {"reasoning_effort": "none", "temperature": 0}


async def _create(client: AsyncOpenAI, **kwargs):
    extras = dict(_OPTIONAL)
    for _ in range(len(_OPTIONAL) + 1):
        try:
            return await client.chat.completions.create(**kwargs, **extras)
        except BadRequestError as e:
            bad = next((k for k in extras if k in str(e)), None)
            if bad is None:
                raise
            extras.pop(bad)          # this model does not take it; try again
    raise RuntimeError("could not find a parameter set this model accepts")


def _openai_model(spec: str | None) -> str:
    """AGENT_MODEL / --model spec -> the bare OpenAI model name.

    This agent is written against the OpenAI SDK, so it runs on OpenAI only.
    For Claude, Bedrock or Foundry use agent.tool_agent, which is built on
    the provider layer.
    """
    spec = spec or os.environ.get("AGENT_MODEL") or "openai:gpt-4o-mini"
    platform, sep, name = spec.partition(":")
    if not sep:
        return spec                          # bare "gpt-4o-mini"
    if platform != "openai":
        raise ValueError(f"support_agent runs on OpenAI only, got {spec!r} - "
                         f"use --target agent.tool_agent:run_agent for {platform}")
    return name or "gpt-4o-mini"


def _to_openai_tools(specs: list) -> list[dict]:
    return [{"type": "function",
             "function": {"name": s.name, "description": s.description,
                          "parameters": s.input_schema}}
            for s in specs]


async def run_agent(messages: list[dict], tool_specs: list,
                    call_tool: ToolCaller, context: dict | None = None,
                    max_steps: int = 10, model: str | None = None) -> str:
    """Run the agent loop. Returns the final message to the customer.

    `context` is who the caller is - in production this comes from the logged
    in session, so the eval must supply it too. Without it the agent has to
    ask "who are you?" on every single turn, which is not the system we ship.

    `call_tool` is supplied by the harness - the agent does not know or care
    whether it is talking to a fake database or a real one.
    """
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = _openai_model(model)

    system = SYSTEM_PROMPT
    caller = (context or {}).get("customer_id")
    if caller:
        system += (f"\n\nThe person you are speaking with is signed in as "
                   f"customer_id \"{caller}\". Use that id when you call "
                   f"verify_customer. Never ask them to type it.")

    convo: list[dict] = [{"role": "system", "content": system}, *messages]
    tools = _to_openai_tools(tool_specs)

    for _ in range(max_steps):
        resp = await _create(client, model=model, messages=convo, tools=tools)
        msg = resp.choices[0].message
        convo.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or ""

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            # The harness intercepts here: it runs the tool AND records it.
            result = await call_tool(tc.function.name, args)
            convo.append({"role": "tool", "tool_call_id": tc.id,
                          "content": json.dumps(result, default=str)})

    return "I wasn't able to complete that within the allowed number of steps."
