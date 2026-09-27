"""What an LLM judge is shown: the whole exchange, and the facts behind it.

A judge that sees only the question and the final reply has to GUESS
whether a claim in the reply is true. "It will arrive before Friday" reads
as helpful unless you know the tools never returned a delivery date. So the
judge is also shown every tool call the agent made and what came back -
the only facts the agent had - and can tell grounded from invented.
"""

from __future__ import annotations

import json

from evalkit.schema.case import Case
from evalkit.schema.trajectory import StepType, Trajectory

# A judge prompt has a budget. Tool results beyond this are cut, with a
# marker, rather than silently dropped or allowed to crowd out the rubric.
MAX_CONTEXT_CHARS = 8000


def conversation(case: Case, traj: Trajectory) -> str:
    """Everything said before the reply being judged.

    One customer message -> that message, as before. A multi-turn exchange
    -> the whole transcript, because the final reply only makes sense
    against what the customer said along the way.
    """
    turns = [("Customer" if s.type is StepType.USER else "Agent", s.content)
             for s in traj.steps
             if s.type in (StepType.USER, StepType.ASSISTANT) and s.content]
    if turns and turns[-1][0] == "Agent":
        turns = turns[:-1]                 # the reply under judgement
    if not turns:                          # an adapter that records no user steps
        turns = [("Customer", m.content) for m in case.input.messages if m.role == "user"]
    if len(turns) == 1:
        return turns[0][1].strip()
    return "\n".join(f"{who}: {text}" for who, text in turns).strip()


def tool_context(traj: Trajectory) -> str | None:
    """Every tool call and its result, or None when no tool was called."""
    if not traj.tool_calls:
        return None
    lines = []
    for tc in traj.tool_calls:
        outcome = (f"FAILED: {tc.error}" if tc.error
                   else json.dumps(tc.result, default=str, ensure_ascii=False))
        lines.append(f"- {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)}) "
                     f"-> {outcome}")
    text = ("Tool calls the agent made and what each returned. These are the "
            "only facts the agent had; a claim in the reply that none of them "
            "supports was invented.\n" + "\n".join(lines))
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS] + "\n[... tool output truncated]"
    return text
