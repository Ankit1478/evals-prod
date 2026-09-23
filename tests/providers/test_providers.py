"""Providers: spec parsing and message conversion. No network - every client
here is a fake that records what it was asked."""

from types import SimpleNamespace as NS

import pytest

from evalkit.env.base import ToolSpec
from evalkit.providers import (AnthropicProvider, OpenAIProvider, ProviderError,
                               get_provider, parse_spec)
from evalkit.providers.anthropic import to_anthropic_messages
from evalkit.providers.openai import to_openai_messages

TOOL = ToolSpec(name="lookup", description="Look one up.",
                input_schema={"$schema": "http://json-schema.org/draft-07/schema#",
                              "type": "object", "properties": {"id": {"type": "string"}}})

CONVO = [
    {"role": "user", "content": "find 1 and 2"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "name": "lookup", "arguments": {"id": "1"}},
        {"id": "b", "name": "lookup", "arguments": {"id": "2"}}]},
    {"role": "tool", "tool_call_id": "a", "name": "lookup", "content": "{}"},
    {"role": "tool", "tool_call_id": "b", "name": "lookup", "content": "{}",
     "is_error": True},
]


@pytest.mark.parametrize("spec,want", [
    ("anthropic:claude-opus-5", ("anthropic", "claude-opus-5", {})),
    ("bedrock:anthropic.claude-sonnet-5", ("bedrock", "anthropic.claude-sonnet-5", {})),
    ("foundry:claude-opus-5?effort=low", ("foundry", "claude-opus-5", {"effort": "low"})),
    ("azure:my-deployment", ("azure", "my-deployment", {})),
    ("gpt-4o-mini", ("openai", "gpt-4o-mini", {})),
    ("claude-sonnet-5", ("anthropic", "claude-sonnet-5", {})),
    ("anthropic:", ("anthropic", None, {})),
])
def test_parse_spec(spec, want):
    assert parse_spec(spec) == want


def test_get_provider_needs_no_credentials_until_first_call():
    p = get_provider("bedrock:anthropic.claude-sonnet-5")
    assert isinstance(p, AnthropicProvider)
    assert p.describe() == {"provider": "bedrock", "model": "anthropic.claude-sonnet-5",
                            "effort": "default"}
    assert isinstance(get_provider("azure:dep"), OpenAIProvider)


def test_default_models():
    assert get_provider("anthropic:").model == "claude-opus-5"
    assert get_provider("bedrock:").model == "anthropic.claude-opus-5"


@pytest.mark.parametrize("bad", ["vertex:x", "anthropic:claude-opus-5?effort=huge",
                                 "anthropic:claude-opus-5?temp=0", "openai:gpt?effort=low"])
def test_bad_specs_fail_loudly(bad):
    with pytest.raises(ProviderError):
        get_provider(bad)


def test_anthropic_groups_back_to_back_tool_results_into_one_message():
    out = to_anthropic_messages(CONVO, "anthropic")
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    results = out[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[1]["is_error"] is True
    assert [b["type"] for b in out[1]["content"]] == ["tool_use", "tool_use"]


def test_raw_turn_is_replayed_only_to_the_provider_that_made_it():
    raw = [{"type": "thinking", "thinking": "", "signature": "sig"},
           {"type": "text", "text": "hi"}]
    turn = {"role": "assistant", "content": "hi", "raw": raw, "provider": "bedrock"}
    msgs = [{"role": "user", "content": "x"}, turn]
    assert to_anthropic_messages(msgs, "bedrock")[1]["content"] == raw
    rebuilt = to_anthropic_messages(msgs, "anthropic")[1]["content"]
    assert rebuilt == [{"type": "text", "text": "hi"}]


def test_openai_conversion():
    out = to_openai_messages(CONVO, "be brief", "openai")
    assert out[0] == {"role": "system", "content": "be brief"}
    assert out[2]["tool_calls"][0]["function"] == {"name": "lookup", "arguments": '{"id": "1"}'}
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "tool"]


class FakeAnthropic:
    def __init__(self):
        self.sent = None
        self.messages = self

    async def create(self, **kwargs):
        self.sent = kwargs
        block = NS(type="tool_use", id="t1", name="lookup", input={"id": "9"},
                   model_dump=lambda **_: {"type": "tool_use", "id": "t1",
                                           "name": "lookup", "input": {"id": "9"}})
        return NS(content=[block], stop_reason="tool_use", model="claude-opus-5",
                  usage=NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=2))


async def test_anthropic_complete_round_trip():
    client = FakeAnthropic()
    p = AnthropicProvider("anthropic", "claude-opus-5", effort="low", client=client)
    out = await p.complete([{"role": "user", "content": "go"}], system="sys", tools=[TOOL])

    assert client.sent["output_config"] == {"effort": "low"}
    assert client.sent["system"] == "sys"
    assert "$schema" not in client.sent["tools"][0]["input_schema"]
    assert out.tool_calls[0].arguments == {"id": "9"}
    assert out.usage.input_tokens == 10 and out.usage.cached_input_tokens == 2
    msg = out.as_message()
    assert msg["provider"] == "anthropic" and msg["raw"][0]["id"] == "t1"


class FakeOpenAI:
    def __init__(self):
        self.sent = None
        self.chat = NS(completions=self)

    async def create(self, **kwargs):
        self.sent = kwargs
        tc = NS(id="c1", function=NS(name="lookup", arguments="{not json"))
        msg = NS(content=None, tool_calls=[tc],
                 model_dump=lambda **_: {"role": "assistant", "tool_calls": []})
        return NS(choices=[NS(message=msg, finish_reason="tool_calls")], model="gpt",
                  usage=NS(prompt_tokens=3, completion_tokens=1, prompt_tokens_details=None))


async def test_openai_keeps_malformed_arguments_for_grading():
    p = OpenAIProvider("openai", "gpt", client=FakeOpenAI())
    out = await p.complete([{"role": "user", "content": "go"}], tools=[TOOL])
    assert out.stop_reason == "tool_use"
    assert out.tool_calls[0].arguments == {}
    assert out.tool_calls[0].raw_arguments == "{not json"


# --- reasoning models refuse function tools unless reasoning is off -----------

def bad_request(message: str):
    import httpx2
    import openai
    resp = httpx2.Response(400, request=httpx2.Request("POST", "https://api.openai.com"))
    return openai.BadRequestError(message, response=resp, body=None)


class ReasoningModelOpenAI(FakeOpenAI):
    """Rejects tools unless reasoning_effort='none', like gpt-5.6-luna."""

    def __init__(self):
        super().__init__()
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("tools") and kwargs.get("reasoning_effort") != "none":
            raise bad_request("Function tools with reasoning_effort are not supported "
                              "for gpt-5.6-luna. set reasoning_effort to 'none'.")
        return await super().create(**kwargs)


async def test_openai_retries_with_reasoning_off_when_the_model_demands_it():
    fake = ReasoningModelOpenAI()
    p = OpenAIProvider("openai", "gpt-5.6-luna", client=fake)
    out = await p.complete([{"role": "user", "content": "go"}], tools=[TOOL])
    assert out.stop_reason == "tool_use"
    assert [c.get("reasoning_effort") for c in fake.calls] == [None, "none"]


async def test_the_setting_is_kept_so_later_turns_do_not_fail_first():
    fake = ReasoningModelOpenAI()
    p = OpenAIProvider("openai", "gpt-5.6-luna", client=fake)
    await p.complete([{"role": "user", "content": "go"}], tools=[TOOL])
    await p.complete([{"role": "user", "content": "again"}], tools=[TOOL])
    assert len(fake.calls) == 3                  # one failed call, ever
    assert p.describe()["reasoning_effort"] == "none"


async def test_a_model_that_never_asks_never_gets_the_parameter():
    """gpt-4o-mini rejects reasoning_effort outright - never send it unasked."""
    fake = FakeOpenAI()
    await OpenAIProvider("openai", "gpt-4o-mini", client=fake).complete(
        [{"role": "user", "content": "go"}], tools=[TOOL])
    assert "reasoning_effort" not in fake.sent


async def test_an_unrelated_bad_request_is_not_swallowed():
    import openai
    import pytest

    class Broken(FakeOpenAI):
        async def create(self, **kwargs):
            raise bad_request("context length exceeded")

    with pytest.raises(openai.BadRequestError):
        await OpenAIProvider("openai", "gpt", client=Broken()).complete(
            [{"role": "user", "content": "go"}], tools=[TOOL])
