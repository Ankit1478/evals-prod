"""The judge must see the facts behind a reply, not just the reply."""

import json

from evalkit.graders.judge_input import MAX_CONTEXT_CHARS, conversation, tool_context
from evalkit.graders.llm_as_judge import LLMAsJudge
from evalkit.schema.trajectory import Step, StepType, Trajectory

from tests.conftest import call, make_case, make_traj
from tests.graders.test_llm_as_judge import RecordingFake, good_payload, write_rubric


def test_single_turn_question_is_the_customer_message():
    assert conversation(make_case(ask="Where is 123?"), make_traj()) == "Where is 123?"


def test_multi_turn_question_is_the_whole_exchange_minus_the_judged_reply():
    steps = [Step(index=0, type=StepType.USER, content="Cancel my order."),
             Step(index=1, type=StepType.ASSISTANT, content="Which one, 123 or 789?"),
             Step(index=2, type=StepType.USER, content="789."),
             Step(index=3, type=StepType.ASSISTANT, content="789 is cancelled.")]
    t = Trajectory(run_id="r", case_id="c", steps=steps, final_output="789 is cancelled.")
    assert conversation(make_case(), t) == ("Customer: Cancel my order.\n"
                                            "Agent: Which one, 123 or 789?\n"
                                            "Customer: 789.")


def test_tool_context_lists_every_call_and_result_including_failures():
    t = make_traj(calls=[call("lookup_order", {"order_id": "123"},
                              result={"status": "active"}),
                         call("cancel_order", {"order_id": "123"}, error="timeout")])
    ctx = tool_context(t)
    assert 'lookup_order({"order_id": "123"}) -> {"status": "active"}' in ctx
    assert "cancel_order" in ctx and "FAILED: timeout" in ctx


def test_no_tools_means_no_context():
    assert tool_context(make_traj()) is None


def test_huge_tool_output_is_truncated_with_a_marker():
    t = make_traj(calls=[call("dump", result={"x": "y" * (MAX_CONTEXT_CHARS * 2)})])
    ctx = tool_context(t)
    assert len(ctx) < MAX_CONTEXT_CHARS + 100
    assert ctx.endswith("[... tool output truncated]")


async def test_the_judge_request_carries_tool_facts_and_the_reference(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    rubric_dir = write_rubric(tmp_path / "rubrics", "answer_quality", judge="llm_as_judge")
    case = make_case(expected={"rubric_id": "answer_quality",
                               "reference_answer": "123 is active; no delivery date."})
    fake = RecordingFake(good_payload(case.id))
    judge = LLMAsJudge(rubric_dir=rubric_dir, client=fake)
    traj = make_traj(calls=[call("lookup_order", {"order_id": "123"},
                                 result={"status": "active"})],
                     say="Order 123 arrives Friday.")
    s = await judge.grade(case, traj)
    sent = json.dumps(fake.sent["messages"])
    assert "lookup_order" in sent and "active" in sent          # grounded
    assert "no delivery date" in sent                           # referenced
    assert s.evidence["grounded_in_tools"] is True
    assert s.evidence["reference"] is True


async def test_without_a_reference_the_judge_stays_reference_free(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    rubric_dir = write_rubric(tmp_path / "rubrics", "answer_quality", judge="llm_as_judge")
    case = make_case(expected={"rubric_id": "answer_quality"})
    fake = RecordingFake(good_payload(case.id))
    s = await LLMAsJudge(rubric_dir=rubric_dir, client=fake).grade(case, make_traj(say="ok"))
    assert s.passed is True
    assert s.evidence["reference"] is False
    assert "reference_free" in json.dumps(fake.sent["messages"])


async def test_binary_rubric_asks_for_pass_or_fail(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    d = tmp_path / "rubrics"
    d.mkdir()
    (d / "honest.json").write_text(json.dumps({"judge": "llm_as_judge", "mode": "binary"}))
    case = make_case(expected={"rubric_id": "honest"})
    fake = RecordingFake(json.dumps({"case_id": case.id, "decision": "FAIL",
                                     "evidence": "claims a date no tool returned"}))
    s = await LLMAsJudge(rubric_dir=d, client=fake).grade(case, make_traj(say="Friday."))
    assert s.passed is False and s.value == 0.0
    assert s.evidence["mode"] == "binary"
    assert "binary" in json.dumps(fake.sent["messages"])


async def test_binary_two_model_agreement_is_a_vote(tmp_path, monkeypatch):
    from tests.graders.test_two_model import AZURE_ENV, FakeClient
    from llm_judge.multi_judge import TwoModelJudge

    class BinaryClient(FakeClient):
        def evaluate(self, prompt):
            from llm_judge.azure_client import RawJudgeResponse, TokenUsage
            return RawJudgeResponse(deployment=self.settings.deployment, mode=prompt.mode,
                                    rubric_version=prompt.rubric_version, usage=TokenUsage(),
                                    content=json.dumps({"case_id": "t_case_01",
                                                        "decision": "PASS",
                                                        "evidence": "grounded"}))
    for k, v in AZURE_ENV.items():
        monkeypatch.setenv(k, v)
    d = tmp_path / "rubrics"
    d.mkdir()
    (d / "q.json").write_text(json.dumps({"judge": "llm_as_judge", "two_model": True,
                                          "mode": "binary"}))
    panel = TwoModelJudge(terra_client=BinaryClient("gpt-5.6-terra", {}),
                          luna_client=BinaryClient("gpt-5.6-luna", {}))
    s = await LLMAsJudge(rubric_dir=d, two_model_judge=panel).grade(
        make_case(expected={"rubric_id": "q"}), make_traj(say="ok"))
    assert s.passed is True and s.evidence["mode"] == "binary"


async def test_judge_token_usage_is_recorded(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("LLM_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    rubric_dir = write_rubric(tmp_path / "rubrics", "answer_quality", judge="llm_as_judge")
    case = make_case(expected={"rubric_id": "answer_quality"})

    class WithUsage(RecordingFake):
        def create(self, **kw):
            resp = super().create(**kw)
            resp.usage = SimpleNamespace(prompt_tokens=900, completion_tokens=120,
                                         total_tokens=1020)
            return resp

    s = await LLMAsJudge(rubric_dir=rubric_dir, client=WithUsage(good_payload(case.id))
                         ).grade(case, make_traj(say="ok"))
    assert s.evidence["usage"] == {"input_tokens": 900, "output_tokens": 120}
