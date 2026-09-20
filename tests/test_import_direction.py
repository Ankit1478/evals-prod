"""The single rule that decides whether this harness generalises.

If `evalkit` ever imports from `suites/`, the engine has learned about one
specific domain and can never be handed to another team.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "evalkit"


def test_engine_never_imports_content():
    offenders = []
    for py in SRC.rglob("*.py"):
        text = py.read_text()
        if re.search(r"^\s*(from|import)\s+suites", text, re.MULTILINE):
            offenders.append(str(py))
    assert not offenders, f"engine imports content: {offenders}"


def test_schema_imports_nothing_internal():
    """schema is the bottom of the stack - everything imports it, it imports
    nothing back. That is what keeps the dependency graph one-way."""
    allowed = {"evalkit.schema"}
    for py in (SRC / "schema").rglob("*.py"):
        for m in re.findall(r"^\s*from\s+(evalkit[\w.]*)", py.read_text(), re.MULTILINE):
            assert any(m.startswith(a) for a in allowed), f"{py.name} imports {m}"
