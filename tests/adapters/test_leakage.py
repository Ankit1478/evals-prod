"""The answer key must never reach the agent."""

import json

from evalkit.adapters.echo import EchoAdapter
from evalkit.env.memory import MemoryEnv
from tests.conftest import make_case


async def test_expected_never_reaches_the_adapter(tmp_path):
    """Poison `expected` with a unique string, then assert it appears nowhere
    in what the adapter produced or could have seen."""
    poison = "SECRET_ANSWER_KEY_9f3a"
    case = make_case({"answer_contains": [poison],
                      "final_state": {"orders": {"123": {"status": poison}}}})

    script = tmp_path / "s.json"
    script.write_text(json.dumps({case.id: {"actions": [], "say": "hello"}}))

    env = MemoryEnv()
    await env.setup(case.initial_state)
    traj = await EchoAdapter(script).run(case, env, "r1", 0)

    assert poison not in traj.model_dump_json()
    assert poison not in json.dumps(await env.snapshot())


async def test_adapter_sees_the_question_but_not_the_answer(tmp_path):
    case = make_case({"answer_contains": ["TOP_SECRET"]}, ask="Cancel order 123.")
    script = tmp_path / "s.json"
    script.write_text(json.dumps({case.id: {"actions": [], "say": "ok"}}))

    env = MemoryEnv()
    await env.setup(case.initial_state)
    traj = await EchoAdapter(script).run(case, env, "r1", 0)

    dump = traj.model_dump_json()
    assert "Cancel order 123." in dump      # the question is allowed through
    assert "TOP_SECRET" not in dump         # the answer key is not


async def test_echo_records_the_scripted_conversation_but_never_reads_user_sim(tmp_path):
    """The transcript comes from the script; the case's private user side stays private."""
    from evalkit.schema.case import UserSim

    case = make_case(ask="Cancel my order.").model_copy(update={
        "user_sim": UserSim(goal="PRIVATE_GOAL_77", facts={"x": "PRIVATE_FACT_77"})})
    script = tmp_path / "s.json"
    script.write_text(json.dumps({case.id: {"actions": [], "say": "Cancelled 789.",
                                            "conversation": [
                                                {"role": "assistant", "content": "Which one?"},
                                                {"role": "user", "content": "789."}]}}))
    env = MemoryEnv()
    await env.setup(case.initial_state)
    traj = await EchoAdapter(script).run(case, env, "r1", 0)
    assert [s.content for s in traj.steps] == ["Cancel my order.", "Which one?", "789.", "Cancelled 789."]
    assert "PRIVATE_" not in traj.model_dump_json()
