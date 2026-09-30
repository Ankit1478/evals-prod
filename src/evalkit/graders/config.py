"""Per-suite grading config: the domain facts the generic graders need.

The graders know nothing about orders, refunds or customers. Who owns which
record, and which words mean "done" in a reply, are facts about a SUITE -
so they live in that suite's grading.json, and a new suite (a co-sell MCP
server, say) configures them there instead of editing grader code.

A suite without grading.json still runs: the word lists fall back to
generic English, and the ownership check abstains rather than guess.
"""

from __future__ import annotations

import json
from pathlib import Path

from evalkit.schema.case import Frozen

GRADING_FILE = "grading.json"

# owner_field value meaning "the record's own key is its owner", e.g. a
# customers table keyed by customer id.
OWNER_IS_KEY = "$key"


class OwnedCollection(Frozen):
    """One collection in initial_state whose records belong to someone.

    owner_field     the field naming the owner, or "$key"
    private_fields  fields that must never reach anyone else, e.g. the item
                    or the price. The record's key always counts too.
    """
    owner_field: str = OWNER_IS_KEY
    private_fields: list[str] = []


class Ownership(Frozen):
    """caller: dotted path to the signed-in user, e.g. "session.customer_id"."""
    caller: str
    collections: dict[str, OwnedCollection]


class GradingConfig(Frozen):
    """Everything is optional. The word lists EXTEND the generic defaults."""
    ownership: Ownership | None = None
    success_words: list[str] = []
    failure_words: list[str] = []
    refusal_words: list[str] = []
    # Status words a reply may only state if a tool returned them. Empty ->
    # output.grounded checks numbers only.
    status_words: list[str] = []


def load_grading_config(suite_dir: str | Path) -> GradingConfig:
    path = Path(suite_dir) / GRADING_FILE
    if not path.exists():
        return GradingConfig()
    raw = json.loads(path.read_text())
    # "_doc" and friends are notes for people, like in faults.json.
    return GradingConfig(**{k: v for k, v in raw.items() if not k.startswith("_")})


def dig(data: dict, dotted: str):
    """dig({"session": {"customer_id": "u1"}}, "session.customer_id") -> "u1"."""
    for part in dotted.split("."):
        if not isinstance(data, dict):
            return None
        data = data.get(part)
    return data
