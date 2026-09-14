"""A Trajectory is the recording of ONE run: everything the agent said and did.

This is the single most important shape in the system. Every adapter produces
one. Every grader consumes one. Nothing else crosses that boundary.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from evalkit.schema.case import Frozen


class ToolStatus(str, Enum):
    """How far a tool call actually got. The distinction that matters.

    attempted    -> the agent emitted the call
    acknowledged -> the tool returned OK
    committed    -> the data ACTUALLY changed

    An API can return 200 while writing the wrong field: acknowledged, but
    never committed. Transport success is not task success.
    """
    ATTEMPTED = "attempted"
    ACKNOWLEDGED = "acknowledged"
    COMMITTED = "committed"
    FAILED = "failed"


class StepType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"


class StopReason(str, Enum):
    """Why the run ended. NEVER optional - a timed-out run is a recorded
    result, not a missing row. This is what protects the denominator."""
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    ERROR = "error"
    REFUSED = "refused"
    BUDGET_EXCEEDED = "budget_exceeded"


class ToolCall(Frozen):
    id: str
    name: str
    arguments: dict                 # parsed
    raw_arguments: str = ""         # text before parsing - needed to grade malformed args
    status: ToolStatus = ToolStatus.ATTEMPTED
    result: Any = None
    error: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class Step(Frozen):
    index: int
    type: StepType
    content: str | None = None
    tool_call: ToolCall | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


class StateChange(Frozen):
    """One thing that changed in the fake database."""
    entity: str                     # "orders/123"
    field: str                      # "status"
    before: Any
    after: Any


class Usage(Frozen):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0


class Trajectory(Frozen):
    run_id: str
    case_id: str
    trial_index: int = 0
    seed: int = 0

    steps: list[Step] = []
    tool_calls: list[ToolCall] = []   # flattened, ordered, INCLUDES failures
    final_output: str | None = None

    state_before: dict = {}
    state_after: dict = {}
    state_diff: list[StateChange] = []   # computed by the env, not the adapter

    usage: Usage = Usage()
    latency_ms: int = 0
    stop_reason: StopReason = StopReason.COMPLETED
    error: str | None = None
