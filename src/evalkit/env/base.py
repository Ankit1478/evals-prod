"""The Environment is the fake backend the agent's tools act on.

This is the piece most teams skip, and skipping it is why their evals only
grade text. Without a real before/after you cannot prove anything actually
happened.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from evalkit.schema.case import Frozen
from evalkit.schema.trajectory import StateChange


class ToolSpec(Frozen):
    """The contract for one tool, as shown to the agent."""
    name: str
    description: str
    input_schema: dict


class ToolResult(Frozen):
    ok: bool
    data: Any = None
    error: str | None = None
    mutated: bool = False        # did this call actually CHANGE the data?


@runtime_checkable
class Environment(Protocol):
    version: str

    async def setup(self, initial_state: dict) -> None: ...
    async def snapshot(self) -> dict: ...
    async def reset(self) -> None: ...
    def tools(self) -> list[ToolSpec]: ...
    async def call(self, name: str, args: dict) -> ToolResult: ...
    def diff(self, before: dict, after: dict) -> list[StateChange]: ...


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """{"orders": {"123": {"status": "active"}}} -> {"orders/123/status": "active"}"""
    out: dict[str, Any] = {}
    for k, v in d.items():
        path = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, path))
        else:
            out[path] = v
    return out


def compute_diff(before: dict, after: dict) -> list[StateChange]:
    """Every field that changed, as entity + field + before + after."""
    fb, fa = flatten(before), flatten(after)
    changes: list[StateChange] = []
    for path in sorted(set(fb) | set(fa)):
        old, new = fb.get(path), fa.get(path)
        if old != new:
            entity, _, field = path.rpartition("/")
            changes.append(StateChange(entity=entity or path, field=field,
                                       before=old, after=new))
    return changes
