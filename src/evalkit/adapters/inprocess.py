"""InProcessAdapter - run a real agent that is an importable Python callable.

The agent runs ITS OWN loop, exactly as it would in production. The harness
does not drive it. It only hands the agent a `call_tool` function and quietly
records every call that passes through - including the ones that fail.

That is the difference between measuring your agent and measuring a copy of
your agent that only exists in tests.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from evalkit.env.base import Environment
from evalkit.schema.case import Case
from evalkit.schema.trajectory import (Step, StepType, StopReason, ToolCall,
                                       ToolStatus, Trajectory, Usage)


class InProcessAdapter:
    """target looks like 'agent.support_agent:run_agent'."""

    name = "inprocess"

    def __init__(self, target: str, model: str | None = None):
        self.target = target
        self.model = model

        # The agent under test lives in the USER's project, not inside the
        # installed evalkit package. Make the working directory importable so
        # `agent.support_agent` resolves without anyone having to package it.
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)

        module_name, _, func_name = target.partition(":")
        self._fn = getattr(importlib.import_module(module_name), func_name)
        self.version = getattr(importlib.import_module(module_name),
                               "__version__", "0.1.0")

    async def run(self, case: Case, env: Environment, run_id: str,
                  trial_index: int) -> Trajectory:
        calls: list[ToolCall] = []
        steps: list[Step] = []

        # The agent only ever sees the conversation. case.expected - the
        # answer key - is not reachable from here.
        messages = [{"role": m.role, "content": m.content}
                    for m in case.input.messages]
        steps.append(Step(index=0, type=StepType.USER,
                          content=messages[-1]["content"]))

        async def call_tool(name: str, args: dict) -> Any:
            """Run the tool against the fake backend, and RECORD it.

            Every attempt is recorded, including rejected and malformed ones.
            A correct call does not cancel out an incorrect extra call.
            """
            started = time.perf_counter()
            out = await env.call(name, args)
            if not out.ok:
                status, result = ToolStatus.FAILED, {"error": out.error}
            elif out.mutated:
                status, result = ToolStatus.COMMITTED, out.data
            else:
                status, result = ToolStatus.ACKNOWLEDGED, out.data

            call = ToolCall(id=f"{case.id}-{len(calls)}", name=name,
                            arguments=args, raw_arguments=json.dumps(args),
                            status=status, result=out.data, error=out.error)
            calls.append(call)
            steps.append(Step(index=len(steps), type=StepType.TOOL_CALL,
                              tool_call=call,
                              latency_ms=int((time.perf_counter() - started) * 1000)))
            steps.append(Step(index=len(steps), type=StepType.TOOL_RESULT,
                              content=json.dumps(result, default=str)))
            return result

        t0 = time.perf_counter()
        stop = StopReason.COMPLETED
        try:
            final = await self._fn(
                messages, env.tools(), call_tool,
                # Session context the agent would also have in production.
                # This is NOT the answer key - it is who is logged in.
                context=case.initial_state.get("session", {}),
                max_steps=case.max_steps, model=self.model)
        except Exception as e:
            final, stop = f"[agent error] {type(e).__name__}: {e}", StopReason.ERROR

        if len(calls) >= case.max_steps:
            stop = StopReason.MAX_STEPS

        steps.append(Step(index=len(steps), type=StepType.ASSISTANT, content=final))

        return Trajectory(
            run_id=run_id, case_id=case.id, trial_index=trial_index,
            steps=steps, tool_calls=calls, final_output=final,
            usage=Usage(), latency_ms=int((time.perf_counter() - t0) * 1000),
            stop_reason=stop,
            error=final if stop is StopReason.ERROR else None,
        )

    def describe(self) -> dict:
        return {"adapter": self.name, "target": self.target,
                "model": self.model or "default", "version": self.version}
