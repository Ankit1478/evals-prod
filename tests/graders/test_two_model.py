"""Two judges that disagree must NOT be averaged into a confident number.

Upstream still computes an aggregate decision in SCORE mode even when the
two models disagree, so the wrapper has to check `agreement` - trusting
that decision would manufacture confidence neither model actually has.

No real Azure call here: both judge clients are injected fakes.
"""

import json
from pathlib import Path

from evalkit.graders.llm_as_judge import LLMAsJudge

from tests.conftest import make_case, make_traj

AZURE_ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
    "AZURE_OPENAI_API_KEY": "not-a-real-key",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "LLM_JUDGE_PROVIDER": "azure",
}


def two_model_rubric(tmp_path: Path) -> Path:
    d = tmp_path / "rubrics"
    d.mkdir(parents=True, exist_ok=True)
    (d / "answer_quality.json").write_text(json.dumps({
        "judge": "llm_as_judge", "two_model": True,
        "instructions": "quality", "criteria": {},
    }))
    return d


class FakeClient:
    """Satisfies the vendored JudgeTransport: .settings + .evaluate(prompt)."""

    def __init__(self, deployment: str, scores: dict):
        from llm_judge.settings import AzureJudgeSettings
        self.settings = AzureJudgeSettings(
            endpoint="https://example.openai.azure.com",
            api_key="k", deployment=deployment, api_version="2024-10-21")
        self._scores = scores

    def evaluate(self, prompt):
        from llm_judge.azure_client import RawJudgeResponse, TokenUsage
        content = json.dumps({
            "case_id": "t_case_01",
            "scores": [{"criterion": c, "score": n, "evidence": "ev"}
                       for c, n in self._scores.items()],
            "summary": "summary",
        })
        return RawJudgeResponse(deployment=self.settings.deployment,
                                mode=prompt.mode,
                                rubric_version=prompt.rubric_version,
                                content=content, usage=TokenUsage())


def build_judge(tmp_path, terra_scores, luna_scores):
    from llm_judge.multi_judge import TwoModelJudge
    return TwoModelJudge(
        terra_client=FakeClient("gpt-5.6-terra", terra_scores),
        luna_client=FakeClient("gpt-5.6-luna", luna_scores),
    )


HIGH = {"correctness": 5, "relevance": 5, "completeness": 5, "clarity": 5}
LOW = {"correctness": 1, "relevance": 1, "completeness": 1, "clarity": 1}


async def test_two_models_that_agree_produce_a_score(tmp_path, monkeypatch):
    for k, v in AZURE_ENV.items():
        monkeypatch.setenv(k, v)
    judge = LLMAsJudge(rubric_dir=two_model_rubric(tmp_path),
                       two_model_judge=build_judge(tmp_path, HIGH, HIGH))
    case = make_case(expected={"rubric_id": "answer_quality"})
    s = await judge.grade(case, make_traj(say="Order 123 is active and in transit."))
    assert s.abstained is False
    assert s.passed is True
    assert s.evidence["judges"] == 2
    assert s.evidence["agreement"] is True


async def test_two_models_that_disagree_abstain_for_human_review(tmp_path, monkeypatch):
    """The important one: disagreement is routed to a human, NOT averaged."""
    for k, v in AZURE_ENV.items():
        monkeypatch.setenv(k, v)
    judge = LLMAsJudge(rubric_dir=two_model_rubric(tmp_path),
                       two_model_judge=build_judge(tmp_path, HIGH, LOW))
    case = make_case(expected={"rubric_id": "answer_quality"})
    s = await judge.grade(case, make_traj(say="Order 123 is active."))
    assert s.abstained is True
    assert s.passed is None
    assert s.evidence["requires_human_review"] is True
    assert "human review" in s.explanation


async def test_a_single_model_rubric_still_uses_one_judge(tmp_path, monkeypatch):
    """two_model is opt-in per rubric - absent means one judge, as before."""
    for k, v in AZURE_ENV.items():
        monkeypatch.setenv(k, v)
    d = tmp_path / "rubrics"
    d.mkdir(parents=True)
    (d / "answer_quality.json").write_text(json.dumps({
        "judge": "llm_as_judge", "instructions": "quality", "criteria": {}}))

    class FakeAzure:
        def __init__(s):
            s.chat = s
            s.completions = s

        def create(s, **kw):
            payload = json.dumps({"case_id": "t_case_01", "scores": [
                {"criterion": c, "score": 5, "evidence": "ev"} for c in
                ("correctness", "relevance", "completeness", "clarity")],
                "summary": "ok"})
            return type("R", (), {"choices": [type("C", (), {"message": type(
                "M", (), {"content": payload})()})()], "usage": None})()

    judge = LLMAsJudge(rubric_dir=d, client=FakeAzure())
    case = make_case(expected={"rubric_id": "answer_quality"})
    s = await judge.grade(case, make_traj(say="Order 123 is active."))
    assert s.evidence["judges"] == 1
    assert s.passed is True


# --- OpenAI backend: which model each judge actually calls -------------------

class RecordingSDK:
    """Fake OpenAI SDK client: records every request's model."""

    def __init__(self):
        self.models = []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.models.append(kw["model"])
        payload = json.dumps({"case_id": "t_case_01", "scores": [
            {"criterion": c, "score": 5, "evidence": "ev"} for c in
            ("correctness", "relevance", "completeness", "clarity")],
            "summary": "ok"})
        return type("R", (), {"choices": [type("C", (), {"message": type(
            "M", (), {"content": payload})()})()], "usage": None})()


def openai_env(monkeypatch, judge_model="gpt-5.6-terra", model="gpt-5.6-luna"):
    monkeypatch.delenv("LLM_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("JUDGE_MODEL", judge_model)
    monkeypatch.setenv("JUDGE_MODEL_2", model)


async def test_openai_multi_judge_calls_both_configured_models(tmp_path, monkeypatch):
    from evalkit.graders.llm_as_judge import build_two_model_judge, judge_backend

    openai_env(monkeypatch)
    settings, _ = judge_backend()
    sdk = RecordingSDK()
    judge = LLMAsJudge(rubric_dir=two_model_rubric(tmp_path),
                       two_model_judge=build_two_model_judge(settings, sdk))
    case = make_case(expected={"rubric_id": "answer_quality"})
    s = await judge.grade(case, make_traj(say="Order 123 is active."))
    assert sorted(sdk.models) == ["gpt-5.6-luna", "gpt-5.6-terra"]
    assert s.evidence["agreement"] is True


async def test_evidence_names_the_models_actually_called(tmp_path, monkeypatch):
    """Not upstream's slot names - the report must say gpt-4.1 if gpt-4.1 judged."""
    from evalkit.graders.llm_as_judge import build_two_model_judge, judge_backend

    openai_env(monkeypatch, judge_model="gpt-judge-a", model="gpt-judge-b")
    settings, _ = judge_backend()
    judge = LLMAsJudge(rubric_dir=two_model_rubric(tmp_path),
                       two_model_judge=build_two_model_judge(settings, RecordingSDK()))
    case = make_case(expected={"rubric_id": "answer_quality"})
    s = await judge.grade(case, make_traj(say="Order 123 is active."))
    assert sorted(s.evidence["per_model"]) == ["gpt-judge-a", "gpt-judge-b"]


def test_the_env_model_is_what_is_actually_sent(monkeypatch):
    """JUDGE_MODEL is a real knob, not documentation - the request
    carries it even though upstream pins the slot name."""
    from evalkit.graders.llm_as_judge import build_two_model_judge, judge_backend

    openai_env(monkeypatch, judge_model="gpt-other-judge")
    settings, _ = judge_backend()
    sdk = RecordingSDK()
    two = build_two_model_judge(settings, sdk)
    from llm_judge.contracts import EvaluationInput, EvaluationMode, ReferencePolicy
    two.evaluate(EvaluationInput(case_id="t_case_01", mode=EvaluationMode.SCORE,
                                 reference_policy=ReferencePolicy.REFERENCE_FREE,
                                 question="q", candidate_answer="a"))
    assert "gpt-other-judge" in sdk.models


def test_two_identical_judges_are_refused(monkeypatch):
    """The same model twice is not a second opinion."""
    import pytest

    from evalkit.graders.llm_as_judge import build_two_model_judge, judge_backend

    openai_env(monkeypatch, judge_model="gpt-5.6-luna", model="gpt-5.6-luna")
    settings, _ = judge_backend()
    with pytest.raises(ValueError, match="different model"):
        build_two_model_judge(settings, RecordingSDK())
