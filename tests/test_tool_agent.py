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


def test_support_agent_takes_the_same_spec_format():
    import pytest

    from agent.support_agent import _openai_model

    assert _openai_model("openai:gpt-5.6-luna") == "gpt-5.6-luna"
    assert _openai_model("gpt-4o-mini") == "gpt-4o-mini"
    with pytest.raises(ValueError, match="tool_agent"):
        _openai_model("bedrock:anthropic.claude-opus-5")


def test_support_agent_falls_back_to_agent_model(monkeypatch):
    from agent.support_agent import _openai_model

    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-from-env")
    assert _openai_model(None) == "gpt-from-env"
