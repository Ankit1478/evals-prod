"""grading.json holds a suite's domain facts, so the graders stay generic."""

import json

import pytest
from pydantic import ValidationError

from evalkit.graders.composite import build_graders
from evalkit.graders.config import GradingConfig, dig, load_grading_config


def test_a_suite_without_grading_json_gets_the_defaults(tmp_path):
    assert load_grading_config(tmp_path) == GradingConfig()


def test_notes_are_ignored_and_typos_are_errors(tmp_path):
    (tmp_path / "grading.json").write_text(json.dumps(
        {"_doc": "a note", "success_words": ["synced"]}))
    assert load_grading_config(tmp_path).success_words == ["synced"]

    (tmp_path / "grading.json").write_text(json.dumps({"sucess_words": ["x"]}))
    with pytest.raises(ValidationError):
        load_grading_config(tmp_path)


def test_the_support_suite_config_loads_and_reaches_the_graders():
    config = load_grading_config("suites/support-agent")
    assert config.ownership.caller == "session.customer_id"
    graders = {g.name: g for g in build_graders(config)}
    assert graders["safety.no_unauthorized_disclosure"].ownership is config.ownership
    assert "refunded" in graders["output.honesty"].success_words


def test_dig():
    assert dig({"a": {"b": "x"}}, "a.b") == "x"
    assert dig({"a": "flat"}, "a.b") is None
    assert dig({}, "a") is None
