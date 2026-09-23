"""The simulated user, and the multi-turn loop in the in-process adapter."""

from datetime import date

import pytest

from evalkit.adapters.inprocess import InProcessAdapter
from evalkit.env.memory import MemoryEnv
from evalkit.providers import Completion
from evalkit.schema.case import Case, CaseInput, Expected, Message, UserSim
from evalkit.schema.trajectory import StepType
from evalkit.simulation import DONE, SimulatedUser, get_persona
from evalkit.simulation.user import render_transcript
from tests.conftest import WORLD

SECRET = "USER_ONLY_FACT_7c1"


def _case(**sim) -> Case:
    return Case(id="mt_01", suite="test", kind="regression",
                input=CaseInput(messages=[Message(role="user", content="cancel my order")],
                                max_turns=4, persona="terse"),
                initial_state=WORLD, expected=Expected(),
                user_sim=UserSim(goal="cancel order 123", facts={"order": SECRET}, **sim),
                added_at=date(2026, 1, 1))


def test_multi_turn_case_without_a_user_is_rejected():
    with pytest.raises(ValueError, match="needs user_sim"):
        Case(id="x", suite="t", kind="regression",
             input=CaseInput(messages=[Message(role="user", content="hi")], max_turns=3),
             initial_state={}, expected=Expected(), added_at=date(2026, 1, 1))


async def test_script_is_replayed_in_order_then_ends():
    u = SimulatedUser(UserSim(goal="g", script=["one", "two"]), get_persona(None))
    assert [await u.reply([]), await u.reply([]), await u.reply([])] == ["one", "two", None]


async def test_done_in_script_ends_the_conversation():
    u = SimulatedUser(UserSim(goal="g", script=[DONE, "never"]), get_persona(None))
    assert await u.reply([]) is None


def test_model_user_needs_a_provider():
    with pytest.raises(ValueError, match="--user-model"):
        SimulatedUser(UserSim(goal="g"), get_persona(None))


def test_unknown_persona_fails_loudly():
    with pytest.raises(ValueError):
        get_persona("grumpy")


def test_transcript_shows_typed_text_only():
    text = render_transcript([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "SECRET_TOOL_OUTPUT"},
        {"role": "assistant", "content": "which order?"},
    ])
    assert "SECRET_TOOL_OUTPUT" not in text
    assert text == "USER (you): hi\n\nASSISTANT: which order?"


class FakeUserModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    async def complete(self, messages, *, system="", tools=None, max_tokens=16000):
        self.seen.append((system, messages))
        return Completion(text=self.replies.pop(0))

    def describe(self):
        return {"provider": "fake", "model": "fake"}


async def test_model_user_gets_goal_facts_and_persona():
    fake = FakeUserModel(["order 123", DONE])
    u = SimulatedUser(UserSim(goal="cancel 123", facts={"order": "123"}),
                      get_persona("terse"), fake)
    assert await u.reply([{"role": "assistant", "content": "which one?"}]) == "order 123"
    assert await u.reply([]) is None
    system, msgs = fake.seen[0]
    assert "cancel 123" in system and "- order: 123" in system and "very short" in system
    assert "which one?" in msgs[0]["content"]


# ---- the adapter loop -------------------------------------------------------

SEEN_BY_AGENT: list[list[dict]] = []


async def history_agent(messages, tools, call_tool, context=None, max_steps=10, model=None):
    """Returns its full history, like agent.tool_agent does."""
    SEEN_BY_AGENT.append(list(messages))
    await call_tool("lookup_order", {"order_id": "123"})
    reply = f"turn {len(SEEN_BY_AGENT)}"
    return {"output": reply, "usage": {"input_tokens": 7, "output_tokens": 1},
            "messages": [*messages, {"role": "tool", "tool_call_id": "x", "content": "{}"},
                         {"role": "assistant", "content": reply}]}


async def string_agent(messages, tools, call_tool, context=None, max_steps=10, model=None):
    return f"seen {len(messages)}"


async def test_adapter_runs_turns_until_the_user_is_done():
    SEEN_BY_AGENT.clear()
    case = _case(script=["it's 123", DONE])
    env = MemoryEnv()
    await env.setup(case.initial_state)
    traj = await InProcessAdapter("tests.simulation.test_multiturn:history_agent").run(
        case, env, "r1", 0)

    assert len(SEEN_BY_AGENT) == 2                      # 2 agent turns, then DONE
    second = SEEN_BY_AGENT[1]
    assert [m["role"] for m in second] == ["user", "tool", "assistant", "user"]
    assert second[-1]["content"] == "it's 123"          # tool history carried over
    kinds = [s.type for s in traj.steps if s.type in (StepType.USER, StepType.ASSISTANT)]
    assert kinds == [StepType.USER, StepType.ASSISTANT, StepType.USER, StepType.ASSISTANT]
    assert traj.final_output == "turn 2"
    assert traj.usage.input_tokens == 14 and len(traj.tool_calls) == 2


async def test_plain_string_agents_still_work_multi_turn():
    case = _case(script=["a", "b", "c"])
    env = MemoryEnv()
    await env.setup(case.initial_state)
    traj = await InProcessAdapter("tests.simulation.test_multiturn:string_agent").run(
        case, env, "r1", 0)
    # max_turns=4: user, agent, a, agent, b, agent, c, agent
    assert traj.final_output == "seen 7"


async def test_user_side_never_reaches_the_agent():
    SEEN_BY_AGENT.clear()
    case = _case(script=["ok", DONE])
    env = MemoryEnv()
    await env.setup(case.initial_state)
    await InProcessAdapter("tests.simulation.test_multiturn:history_agent").run(
        case, env, "r1", 0)
    assert SECRET not in repr(SEEN_BY_AGENT)
