"""A Case is one test: a situation, a question, and the answer key."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    """Base for every model here.

    frozen=True   -> nobody can modify a case or a result after it is created.
                     A grader must never be able to edit the evidence it grades.
    extra=forbid  -> a typo like "forbiden_tools" is an error, not a silently
                     ignored field that makes a case check nothing.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")


class Message(Frozen):
    role: Literal["user", "assistant", "system"]
    content: str


class CaseInput(Frozen):
    """What the agent is allowed to see."""
    messages: list[Message]
    persona: str | None = None
    max_turns: int = 1


class ExpectedCall(Frozen):
    """A tool call the agent must make."""
    tool: str
    args_equal: dict | None = None          # exact match on these keys
    args_match: dict[str, str] | None = None  # regex per key
    args_constraints: list[str] | None = None  # e.g. "amount_inr <= 2500"
    min_times: int = 1
    max_times: int | None = 1               # catches double-cancel / double-refund


class Expected(Frozen):
    """THE ANSWER KEY. Must never be shown to the agent."""
    final_state: dict | None = None
    forbidden_state_changes: list[str] = []
    required_calls: list[ExpectedCall] = []
    forbidden_tools: list[str] = []
    answer_contains: list[str] = []
    answer_not_contains: list[str] = []
    answer_json_schema: dict | None = None
    rubric_id: str | None = None
    should_refuse: bool = False
    should_clarify: bool = False
    acceptable_alternatives: list[str] = []


class Case(Frozen):
    id: str
    suite: str
    kind: Literal["regression", "capability", "adversarial"]
    input: CaseInput
    initial_state: dict
    expected: Expected
    tags: list[str] = []
    max_steps: int = 20
    timeout_s: int = 120
    review_status: Literal["draft", "reviewed", "approved"] = "draft"
    source: Literal["handwritten", "production", "synthetic"] = "handwritten"
    added_at: date
    notes: str = ""


def load_cases(path: str | Path) -> list[Case]:
    """Read a .jsonl file -> list of Case. One JSON object per line."""
    path = Path(path)
    cases: list[Case] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            cases.append(Case.model_validate(json.loads(line)))
        except Exception as e:
            raise ValueError(f"{path}:{lineno} is not a valid case -> {e}") from e

    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f"duplicate case id: {c.id}")
        seen.add(c.id)
    return cases
