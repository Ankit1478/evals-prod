"""A REAL support agent. This is the system under test.

It lives OUTSIDE src/evalkit on purpose - the harness must never import the
thing it measures. It talks to a real model, decides for itself which tools
to call, and gives a different answer every time. No script.

The prompt below is part of the system under test. Change it and your eval
scores change - which is exactly what the harness is for.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

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


def build_system(context: dict | None) -> str:
    """The policy, plus who is signed in.

    `context` is who the caller is - in production this comes from the logged
    in session, so the eval must supply it too. Without it the agent has to
    ask "who are you?" on every single turn, which is not the system we ship.
    """
    system = SYSTEM_PROMPT
    caller = (context or {}).get("customer_id")
    if caller:
        system += (f"\n\nThe person you are speaking with is signed in as "
                   f"customer_id \"{caller}\". Use that id when you call "
                   f"verify_customer. Never ask them to type it.")
    return system


async def run_agent(messages: list[dict], tool_specs: list,
                    call_tool: ToolCaller, context: dict | None = None,
                    max_steps: int = 10, model: str | None = None) -> dict:
    """Run the agent. Returns the final message and the whole conversation.

    The model is any provider spec - openai:, anthropic:, bedrock:, foundry:
    or azure: - taken from --model, else AGENT_MODEL in .env. The policy
    above is this agent; the loop is the shared one in tool_agent.

    `call_tool` is supplied by the harness - the agent does not know or care
    whether it is talking to a fake database or a real one.
    """
    from agent.tool_agent import run_agent as tool_loop

    return await tool_loop(messages, tool_specs, call_tool,
                           context={"system_prompt": build_system(context)},
                           max_steps=max_steps, model=model)
