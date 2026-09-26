"""JUDGE_MODEL / JUDGE_MODEL_2 take any provider spec, and a Claude judge
answers the vendored judge's one OpenAI-shaped call. No network here."""

import json
from types import SimpleNamespace

import pytest

from evalkit.graders.judge_clients import VERDICT_TOOL, ClaudeChatClient
from evalkit.graders.llm_as_judge import (build_two_model_judge, judge_backend,
                                          model_key, self_judging)


class FakeMessages:
    """Records the Messages API call; replies with a forced tool call."""

    def __init__(self, reject_temperature=False):
        self.calls = []
        self.messages = self
        self._reject = reject_temperature

    def create(self, **kw):
        self.calls.append(kw)
        if self._reject and "temperature" in kw:
            raise ValueError("temperature is not supported for this model")
        verdict = SimpleNamespace(type="tool_use", name=VERDICT_TOOL,
                                  input={"case_id": "c", "scores": [], "summary": "ok"})
        return SimpleNamespace(id="msg_1", model=kw["model"], stop_reason="tool_use",
                               content=[verdict],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def openai_request(**extra):
    return {"model": "claude-x",
            "messages": [{"role": "system", "content": "be a judge"},
                         {"role": "user", "content": "judge this"}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "j", "strict": True,
                                                "schema": {"type": "object"}}},
            "max_completion_tokens": 1200, **extra}


def test_claude_client_translates_the_openai_call():
    fake = FakeMessages()
    resp = ClaudeChatClient("bedrock", client=fake).chat.completions.create(**openai_request())
    sent = fake.calls[0]
    assert sent["system"] == "be a judge"
    assert sent["messages"] == [{"role": "user", "content": "judge this"}]
    assert sent["tool_choice"] == {"type": "tool", "name": VERDICT_TOOL}
    assert sent["tools"][0]["input_schema"] == {"type": "object"}
    assert sent["max_tokens"] == 1200
    # ... and the forced tool call comes back as the JSON reply text.
    assert json.loads(resp.choices[0].message.content)["summary"] == "ok"
    assert resp.usage.total_tokens == 15


def test_claude_client_drops_temperature_when_the_model_refuses_it():
    fake = FakeMessages(reject_temperature=True)
    ClaudeChatClient("anthropic", client=fake).chat.completions.create(
        **openai_request(temperature=0))
    assert "temperature" not in fake.calls[-1]


def test_a_claude_refusal_reaches_the_judge_as_a_refusal():
    class Refuses(FakeMessages):
        def create(self, **kw):
            return SimpleNamespace(id="m", model="x", stop_reason="refusal", content=[],
                                   usage=SimpleNamespace(input_tokens=1, output_tokens=0))
    resp = ClaudeChatClient("anthropic", client=Refuses()).chat.completions.create(
        **openai_request())
    assert resp.choices[0].message.refusal
    assert resp.choices[0].message.content is None


@pytest.mark.parametrize("spec,platform,model", [
    ("gpt-5.6-terra", "OpenAI", "gpt-5.6-terra"),                  # bare = OpenAI, as before
    ("openai:gpt-4.1", "OpenAI", "gpt-4.1"),
    ("bedrock:anthropic.claude-sonnet-5", "ClaudeChatClient", "anthropic.claude-sonnet-5"),
    ("anthropic:claude-opus-5", "ClaudeChatClient", "claude-opus-5"),
    ("foundry:", "ClaudeChatClient", "claude-opus-5"),              # platform default
])
def test_judge_model_takes_any_provider_spec(monkeypatch, spec, platform, model):
    monkeypatch.delenv("LLM_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("JUDGE_MODEL", spec)
    settings, client = judge_backend()
    assert settings.deployment == model
    assert type(client).__name__ == platform


def test_the_same_model_on_two_platforms_is_one_model():
    assert (model_key("bedrock:us.anthropic.claude-opus-5-v1:0")
            == model_key("anthropic:claude-opus-5")
            == model_key("foundry:claude-opus-5"))
    assert model_key("anthropic:claude-opus-5") != model_key("anthropic:claude-sonnet-5")


def test_self_judging_is_caught_across_platforms(monkeypatch):
    monkeypatch.delenv("LLM_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("JUDGE_MODEL", "openai:gpt-5.6-terra")
    monkeypatch.setenv("JUDGE_MODEL_2", "anthropic:claude-opus-5")
    assert self_judging("bedrock:anthropic.claude-opus-5") == ["claude-opus-5"]
    assert self_judging("bedrock:anthropic.claude-sonnet-5") == []


def test_mixed_platform_panel_gives_each_judge_its_own_client(monkeypatch):
    """OpenAI first judge, Claude-on-Bedrock second: two different clients."""
    monkeypatch.delenv("LLM_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("JUDGE_MODEL", "openai:gpt-5.6-terra")
    monkeypatch.setenv("JUDGE_MODEL_2", "bedrock:anthropic.claude-sonnet-5")
    from llm_judge.multi_judge import JudgeModel

    panel = build_two_model_judge(*judge_backend())
    terra, luna = panel._clients[JudgeModel.TERRA], panel._clients[JudgeModel.LUNA]
    assert type(terra._client).__name__ == "OpenAI"
    assert type(luna._client).__name__ == "ClaudeChatClient"
