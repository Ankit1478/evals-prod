"""The generic agent must be told who is signed in.

Without it every model asks "what is your customer ID?" on every case, and
the eval measures the missing session instead of the model.
"""

from agent.tool_agent import DEFAULT_SYSTEM, build_system


def test_the_signed_in_user_reaches_the_model():
    system = build_system({"customer_id": "u1"})
    assert "customer_id: u1" in system
    assert "Never ask the user" in system


def test_a_suite_system_prompt_is_used_and_not_repeated_as_a_session_fact():
    system = build_system({"system_prompt": "You are ShopBot.", "customer_id": "u1"})
    assert system.startswith("You are ShopBot.")
    assert "system_prompt" not in system


def test_no_session_leaves_the_prompt_alone():
    assert build_system(None) == DEFAULT_SYSTEM
    assert build_system({"system_prompt": "Custom."}) == "Custom."


def test_support_agent_keeps_its_policy_and_the_signed_in_caller():
    from agent.support_agent import SYSTEM_PROMPT, build_system

    system = build_system({"customer_id": "u1"})
    assert system.startswith(SYSTEM_PROMPT)
    assert 'customer_id "u1"' in system


async def test_support_agent_runs_on_whatever_provider_the_spec_names(monkeypatch):
    """Any platform - the spec goes to the shared provider layer, and the
    policy prompt goes with it."""
    import agent.tool_agent as tool_agent
    from agent.support_agent import SYSTEM_PROMPT, run_agent
    from evalkit.providers.base import Completion

    seen = {}

    class FakeProvider:
        async def complete(self, messages, *, system="", tools=None):
            seen["system"] = system
            return Completion(text="Order 123 is on its way.", tool_calls=[],
                              stop_reason="end_turn", raw=None)

    def fake_get_provider(spec):
        seen["spec"] = spec
        return FakeProvider()

    monkeypatch.setattr(tool_agent, "get_provider", fake_get_provider)
    out = await run_agent([{"role": "user", "content": "where is 123?"}], [],
                          call_tool=None, context={"customer_id": "u1"},
                          model="bedrock:anthropic.claude-opus-5")
    assert seen["spec"] == "bedrock:anthropic.claude-opus-5"
    assert seen["system"].startswith(SYSTEM_PROMPT)
    assert out["output"] == "Order 123 is on its way."


async def test_support_agent_falls_back_to_agent_model(monkeypatch):
    import agent.tool_agent as tool_agent
    from agent.support_agent import run_agent
    from evalkit.providers.base import Completion

    seen = {}

    class FakeProvider:
        async def complete(self, messages, *, system="", tools=None):
            return Completion(text="ok", tool_calls=[], stop_reason="end_turn", raw=None)

    monkeypatch.setenv("AGENT_MODEL", "anthropic:claude-from-env")
    monkeypatch.setattr(tool_agent, "get_provider",
                        lambda spec: seen.setdefault("spec", spec) and FakeProvider())
    await run_agent([{"role": "user", "content": "hi"}], [], call_tool=None)
    assert seen["spec"] == "anthropic:claude-from-env"
