"""judge.answer_quality - wraps the vendored LLM_AS_JUDGE subsystem as ONE
grader behind the same Score shape as everything else.

That subsystem (vendor/llm_judge/) is ~4,900 lines with its own rubric,
prompt builder, response parser, two-model disagreement path and
calibration tooling. It is WRAPPED, never rewritten (README section 16):
everything here is adaptation, and all the judging logic stays upstream.

Two judges now coexist, so a rubric file names the one it is written for:
    {"judge": "llm_as_judge", ...}   -> this grader
    {"judge": "jev", ...}            -> graders/judge.py
A grader abstains on any rubric that is not addressed to it, so adding a
second judge cannot change what the first one already scored.

Like the jev judge, this ABSTAINS rather than guesses - no rubric, missing
Azure settings, a judge refusal, a malformed response. An abstained score
is excluded from the pass average entirely (composite.decide()).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from evalkit.graders.base import score
from evalkit.schema.case import Case
from evalkit.schema.score import Score, Severity
from evalkit.schema.trajectory import Trajectory

JUDGE_KEY = "llm_as_judge"

# Their criterion scores run 1..5, so 1 is the floor, not zero. Mapping onto
# our 0..1 `value` has to subtract that floor, or a worst-possible 1.0 would
# read as 20% instead of 0%.
_MIN_SCORE, _MAX_SCORE = 1.0, 5.0


def _normalise(weighted_score: float) -> float:
    return (weighted_score - _MIN_SCORE) / (_MAX_SCORE - _MIN_SCORE)


def judge_backend() -> tuple[Any, Any | None]:
    """(settings, sdk_client) for the vendored judge.

    LLM_JUDGE_PROVIDER=openai (default): OPENAI_API_KEY + JUDGE_MODEL. The
    vendored AzureJudgeClient builds a plain chat.completions request, so a
    regular OpenAI SDK client is injected into it - no vendored code changes.
    LLM_JUDGE_PROVIDER=azure: AZURE_OPENAI_* exactly as upstream intended;
    returning no client lets AzureJudgeClient build its own AzureOpenAI one.
    Raises SettingsError when credentials are missing.
    """
    from llm_judge.settings import AzureJudgeSettings, SettingsError

    if os.environ.get("LLM_JUDGE_PROVIDER", "openai").lower() == "azure":
        return AzureJudgeSettings.from_env(), None

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SettingsError("OPENAI_API_KEY is not set")
    settings = AzureJudgeSettings(
        # Unused by the injected client; only satisfies the settings schema.
        endpoint="https://api.openai.com/v1",
        api_key=key,
        # Its own setting, never the agent's: a judge that shares the
        # agent's model is grading its own answers.
        deployment=os.environ.get("JUDGE_MODEL") or "gpt-5.6-terra",
        api_version="openai",
    )
    from openai import OpenAI
    return settings, OpenAI(api_key=key, timeout=settings.timeout_seconds,
                            max_retries=settings.max_retries)


def build_two_model_judge(settings: Any, sdk_client: Any | None) -> Any:
    """Two independent judges on whichever backend judge_backend() chose.

    On OpenAI: terra slot = JUDGE_MODEL (default gpt-5.6-terra),
    luna slot = JUDGE_MODEL_2 (default gpt-5.6-luna). Upstream pins the slot names and rejects a
    client reporting any other deployment, so each client reports its slot
    name while the request carries the configured model.
    """
    from llm_judge.azure_client import AzureJudgeClient
    from llm_judge.multi_judge import JudgeModel, TwoModelJudge

    if sdk_client is None:
        return TwoModelJudge.from_settings(settings)

    terra = settings.deployment
    luna = os.environ.get("JUDGE_MODEL_2") or JudgeModel.LUNA.value
    if terra == luna:
        raise ValueError(f"both judges would be {terra} - a second opinion must "
                         f"come from a different model (set JUDGE_MODEL_2)")

    def client_for(slot: Any, model: str) -> Any:
        class _SlotClient(AzureJudgeClient):
            def build_request(self, prompt: Any) -> dict:
                return {**super().build_request(prompt), "model": model}

        return _SlotClient(settings.model_copy(update={"deployment": slot.value}),
                           client=sdk_client)

    return TwoModelJudge(terra_client=client_for(JudgeModel.TERRA, terra),
                         luna_client=client_for(JudgeModel.LUNA, luna))


class LLMAsJudge:
    """Scores answer quality on the vendored 4-criterion rubric
    (correctness, relevance, completeness, clarity).

    PASS/FAIL comes from the subsystem's own `decision`, which encodes
    thresholds it was calibrated with - this wrapper does not invent a bar
    of its own. The weighted score is carried as evidence for a human.
    """
    name = "judge.answer_quality"
    version = "1"
    severity = Severity.MAJOR

    def __init__(self, rubric_dir: Path, client: Any | None = None,
                 two_model_judge: Any | None = None):
        self.rubric_dir = Path(rubric_dir)
        # `client` is any object exposing .chat.completions.create(**kwargs).
        # Injected by tests so the suite never makes a real Azure call.
        self._client = client
        # A pre-built TwoModelJudge, injected by tests. In normal use the
        # rubric asks for two-model mode and it is built from settings.
        self._two_model_judge = two_model_judge

    def _abstain(self, evidence: dict, explanation: str) -> Score:
        return Score(grader=self.name, grader_version=self.version, value=0.0,
                    passed=None, severity=self.severity, abstained=True,
                    evidence=evidence, explanation=explanation)

    def _applies_to(self, case: Case) -> tuple[bool, dict | None, str]:
        """Is this case addressed to THIS judge? Returns (applies, rubric, why)."""
        rubric_id = case.expected.rubric_id
        if rubric_id is None:
            return False, None, "no rubric on this case - judge does not apply"

        path = self.rubric_dir / f"{rubric_id}.json"
        if not path.exists():
            return False, None, f"no rubric file at {path}"
        try:
            rubric = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            return False, None, f"rubric file is not valid JSON: {e}"

        if rubric.get("judge") != JUDGE_KEY:
            return False, None, (f"rubric '{rubric_id}' is addressed to "
                                 f"judge '{rubric.get('judge')}', not {JUDGE_KEY}")
        return True, rubric, ""

    async def grade(self, case: Case, traj: Trajectory) -> Score:
        applies, _rubric, why = self._applies_to(case)
        if not applies:
            return self._abstain({"rubric_id": case.expected.rubric_id}, why)

        from llm_judge.azure_client import AzureJudgeClient, JudgeClientError
        from llm_judge.contracts import (Decision, EvaluationInput,
                                         EvaluationMode, ReferencePolicy)
        from llm_judge.prompt_builder import build_judge_prompt
        from llm_judge.response_parser import (JudgeRefusalError,
                                               JudgeResponseValidationError,
                                               parse_judge_response)
        from llm_judge.settings import SettingsError

        question = "\n".join(m.content for m in case.input.messages
                             if m.role == "user").strip()
        answer = (traj.final_output or "").strip()
        if not question or not answer:
            return self._abstain(
                {"rubric_id": case.expected.rubric_id},
                "nothing to judge - the case has no user message, or the "
                "agent produced no final answer")

        try:
            eval_input = EvaluationInput(
                case_id=case.id,
                mode=EvaluationMode.SCORE,
                # We have no gold answer per case, so the judge must grade
                # the answer on its own merits rather than against one.
                reference_policy=ReferencePolicy.REFERENCE_FREE,
                question=question,
                candidate_answer=answer,
            )
            prompt = build_judge_prompt(eval_input)
            settings, sdk_client = judge_backend()
        except SettingsError as e:
            return self._abstain({"rubric_id": case.expected.rubric_id},
                                 f"judge credentials missing ({e}) - set "
                                 f"OPENAI_API_KEY (or AZURE_OPENAI_* with "
                                 f"LLM_JUDGE_PROVIDER=azure) in .env")
        except Exception as e:
            return self._abstain({"error": f"{type(e).__name__}: {e}"},
                                 "could not build the judge request - abstaining")

        # Injection aimed at the judge is recorded as evidence on every
        # judged case. The detector is a signal, not proof, so it annotates
        # here rather than deciding - graders/injection.py is what gates.
        from llm_judge.guardrails import detect_prompt_injection
        injection = [{"location": f.location.value, "category": f.category.value}
                     for f in detect_prompt_injection(eval_input)]

        if _rubric.get("two_model"):
            return await self._grade_two_model(case, eval_input, answer,
                                               settings, sdk_client, injection)

        client = AzureJudgeClient(settings, client=self._client or sdk_client)
        try:
            # The subsystem is entirely synchronous; this is the one place
            # that matters, so it runs off the event loop rather than
            # blocking every other case waiting to be graded.
            raw = await asyncio.to_thread(client.evaluate, prompt)
            result = await asyncio.to_thread(parse_judge_response, raw, eval_input)
        except JudgeRefusalError as e:
            return self._abstain({"refusal": str(e)},
                                 "the judge refused to answer - abstaining, "
                                 "which is exactly what a refusal should mean")
        except (JudgeResponseValidationError, JudgeClientError) as e:
            return self._abstain({"error": f"{type(e).__name__}: {e}"},
                                 "judge call failed or returned an unexpected "
                                 "shape - abstaining, not guessing")

        passed = result.decision is Decision.PASS
        weighted = float(result.weighted_score)
        return score(self, _normalise(weighted), passed,
                    {"rubric_id": case.expected.rubric_id,
                     "judges": 1,
                     "model": settings.deployment,
                     "weighted_score": weighted,
                     "scores": {s.criterion.value: s.score for s in result.scores},
                     "summary": result.summary,
                     "injection_findings": injection,
                     "reply": answer},
                    f"answer quality {weighted:.2f}/5 "
                    f"({result.decision.value.lower()})")

    async def _grade_two_model(self, case: Case, eval_input: Any, answer: str,
                               settings: Any, sdk_client: Any, injection: list) -> Score:
        """Two models (terra + luna) judge independently.

        When they disagree the case goes to a human - it is NOT averaged
        into a confident-looking number. Upstream still computes an
        aggregate decision in SCORE mode even on disagreement, so this
        checks `agreement` rather than trusting that decision.
        """
        from llm_judge.azure_client import JudgeClientError
        from llm_judge.response_parser import JudgeResponseValidationError

        try:
            judge = self._two_model_judge or build_two_model_judge(settings, sdk_client)
            result = await asyncio.to_thread(judge.evaluate, eval_input)
        except (JudgeResponseValidationError, JudgeClientError, ValueError, TypeError) as e:
            return self._abstain({"error": f"{type(e).__name__}: {e}"},
                                 "two-model judge call failed - abstaining, "
                                 "not guessing")

        ev = {"rubric_id": case.expected.rubric_id,
              "judges": 2,
              "agreement": result.agreement,
              "requires_human_review": result.requires_human_review,
              "per_model": {j.model.value: (j.result.decision.value
                                            if hasattr(j.result, "decision") else None)
                            for j in result.judgments},
              "average_weighted_score": result.average_weighted_score,
              "injection_findings": injection,
              "reply": answer}

        if result.requires_human_review:
            # Two judges disagreeing is not a score. Averaging it would
            # manufacture confidence that neither model actually has.
            return self._abstain(ev, "terra and luna disagreed - routing to "
                                     "human review instead of averaging")

        weighted = float(result.average_weighted_score or 0.0)
        passed = str(result.aggregate_decision).upper().endswith("PASS")
        return score(self, _normalise(weighted), passed, ev,
                    f"answer quality {weighted:.2f}/5 "
                    f"(terra and luna agreed)")
