"""EchoAdapter - a FAKE agent that replays a script.

Why a fake agent first? So the whole harness can be built and tested before a
real agent exists: no API keys, no cost, no network, same answer every time.
It is also how you test the harness itself - if a scripted agent that does the
wrong thing still PASSES, your graders are broken, not the agent.
"""

from __future__ import annotations

import json
from pathlib import Path

from evalkit.schema.case import Case
from evalkit.schema.trajectory import (
    Step, StepType, StopReason, ToolCall, ToolStatus, Trajectory, Usage,
)


class EchoAdapter:
    name = "echo"
    version = "0.1.0"

    def __init__(self, script_path: str | Path):
        self.script_path = Path(script_path)
        self.script: dict[str, dict] = json.loads(self.script_path.read_text())

    async def run(self, case: Case, run_id: str, trial_index: int) -> Trajectory:
        # NOTE: we read case.input only. case.expected is never touched.
        plan = self.script.get(case.id)

        if plan is None:
            # No script for this case -> the agent does nothing. This is a
            # real, recorded outcome, not a skipped test.
            return Trajectory(
                run_id=run_id, case_id=case.id, trial_index=trial_index,
                final_output="I don't know how to help with that.",
                stop_reason=StopReason.COMPLETED,
            )

        steps: list[Step] = []
        tool_calls: list[ToolCall] = []
        i = 0

        # step 0 is always what the user said
        steps.append(Step(index=i, type=StepType.USER,
                          content=case.input.messages[-1].content))
        i += 1

        for n, action in enumerate(plan.get("actions", [])):
            call = ToolCall(
                id=f"{case.id}-call-{n}",
                name=action["tool"],
                arguments=action.get("args", {}),
                raw_arguments=json.dumps(action.get("args", {})),
                status=ToolStatus(action.get("status", "committed")),
                result=action.get("result"),
                error=action.get("error"),
            )
            tool_calls.append(call)
            steps.append(Step(index=i, type=StepType.TOOL_CALL, tool_call=call))
            i += 1
            steps.append(Step(index=i, type=StepType.TOOL_RESULT,
                              content=json.dumps(action.get("result"))))
            i += 1

        say = plan.get("say", "")
        steps.append(Step(index=i, type=StepType.ASSISTANT, content=say,
                          tokens_out=len(say.split())))

        return Trajectory(
            run_id=run_id, case_id=case.id, trial_index=trial_index,
            steps=steps,
            tool_calls=tool_calls,
            final_output=say,
            usage=Usage(input_tokens=50, output_tokens=len(say.split())),
            latency_ms=10,
            stop_reason=StopReason(plan.get("stop_reason", "completed")),
        )

    def describe(self) -> dict:
        return {"adapter": self.name, "version": self.version,
                "script": str(self.script_path)}
