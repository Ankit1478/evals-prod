"""The wrapped LLM_AS_JUDGE subsystem must behave like every other grader:
one Score, abstain rather than guess, and never touch a case addressed to a
different judge.

No test here makes a real Azure call - the transport is injected.
"""

import json
from pathlib import Path

from evalkit.graders.llm_as_judge import LLMAsJudge, _normalise
from evalkit.schema.score import Severity

from tests.conftest import make_case, make_traj

AZURE_ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
    "AZURE_OPENAI_API_KEY": "not-a-real-key",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "LLM_JUDGE_PROVIDER": "azure",
}


def set_azure_env(monkeypatch):
    for k, v in AZURE_ENV.items():
        monkeypatch.setenv(k, v)


def write_rubric(rubric_dir: Path, rubric_id: str, judge: str) -> Path:
    rubric_dir.mkdir(parents=True, exist_ok=True)
    (rubric_dir / f"{rubric_id}.json").write_text(json.dumps({
        "judge": judge, "instructions": "judge it", "criteria": {},
    }))
    return rubric_dir


class FakeAzure:
    """Mimics the one call AzureJudgeClient makes: .chat.completions.create."""

    def __init__(self, payload: str):
        self._payload = payload
        self.chat = self                      # .chat.completions.create(...)
        self.completions = self

    def create(self, **kwargs):
        class _Msg:
            content = self._payload

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = None

        return _Resp()


def good_payload(case_id: str, score: int = 5) -> str:
    return json.dumps({
        "case_id": case_id,
        "scores": [
            {"criterion": c, "score": score, "evidence": "looks fine to me"}
            for c in ("correctness", "relevance", "completeness", "clarity")
        ],
        "summary": "A clear and correct answer.",
    })


def test_normalise_maps_the_1_to_5_scale_onto_0_to_1():
    """1 is the floor of their scale, not zero - a worst-possible 1.0 must
    read as 0%, never as 20%."""
    assert _normalise(1.0) == 0.0
    assert _normalise(5.0) == 1.0
    assert _normalise(3.0) == 0.5


async def test_abstains_when_the_case_has_no_rubric(tmp_path):
    judge = LLMAsJudge(rubric_dir=tmp_path)
    s = await judge.grade(make_case(), make_traj())
    assert s.abstained is True
    assert s.passed is None


async def test_abstains_on_a_rubric_addressed_to_the_other_judge(tmp_path):
    """The jev rubric is not this judge's to score."""
    rubric_dir = write_rubric(tmp_path / "rubrics", "empathetic_refusal", judge="jev")
    judge = LLMAsJudge(rubric_dir=rubric_dir)
    case = make_case(expected={"rubric_id": "empathetic_refusal"})
    s = await judge.grade(case, make_traj())
    assert s.abstained is True
    assert "jev" in s.explanation


async def test_abstains_when_credentials_are_missing(tmp_path, monkeypatch):
    for k in [*AZURE_ENV, "OPENAI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    rubric_dir = write_rubric(tmp_path / "rubrics", "answer_quality", judge="llm_as_judge")
    judge = LLMAsJudge(rubric_dir=rubric_dir)
    case = make_case(expected={"rubric_id": "answer_quality"})
    s = await judge.grade(case, make_traj(say="Order 123 is on its way."))
    assert s.abstained is True
    assert "OPENAI_API_KEY" in s.explanation


class RecordingFake(FakeAzure):
    """FakeAzure that also records the request it was sent."""

    def create(self, **kwargs):
        self.sent = kwargs
        return super().create(**kwargs)


async def test_default_provider_is_openai_with_the_openai_model(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-model")
    rubric_dir = write_rubric(tmp_path / "rubrics", "answer_quality", judge="llm_as_judge")
    case = make_case(expected={"rubric_id": "answer_quality"})
    fake = RecordingFake(good_payload(case.id, score=5))
    judge = LLMAsJudge(rubric_dir=rubric_dir, client=fake)
    s = await judge.grade(case, make_traj(say="Order 123 is active and in transit."))
    assert s.passed is True
    assert fake.sent["model"] == "gpt-test-model"
    assert s.evidence["model"] == "gpt-test-model"


def test_openai_backend_builds_a_plain_openai_client(monkeypatch):
    """Not AzureOpenAI - the OpenAI key must go to api.openai.com."""
    from openai import AzureOpenAI, OpenAI

    from evalkit.graders.llm_as_judge import judge_backend

    monkeypatch.delenv("LLM_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-model")
    settings, client = judge_backend()
    assert settings.deployment == "gpt-test-model"
    assert isinstance(client, OpenAI) and not isinstance(client, AzureOpenAI)


def test_azure_backend_is_still_available_on_request(monkeypatch):
    from evalkit.graders.llm_as_judge import judge_backend

    set_azure_env(monkeypatch)
    settings, client = judge_backend()
    assert client is None            # AzureJudgeClient builds its own AzureOpenAI
    assert settings.endpoint == "https://example.openai.azure.com"


async def test_abstains_when_there_is_no_answer_to_judge(tmp_path, monkeypatch):
    set_azure_env(monkeypatch)
    rubric_dir = write_rubric(tmp_path / "rubrics", "answer_quality", judge="llm_as_judge")
    judge = LLMAsJudge(rubric_dir=rubric_dir)
    case = make_case(expected={"rubric_id": "answer_quality"})
    s = await judge.grade(case, make_traj(say="   "))
    assert s.abstained is True


async def test_scores_a_good_answer_and_passes(tmp_path, monkeypatch):
    set_azure_env(monkeypatch)
    rubric_dir = write_rubric(tmp_path / "rubrics", "answer_quality", judge="llm_as_judge")
    case = make_case(expected={"rubric_id": "answer_quality"})
    judge = LLMAsJudge(rubric_dir=rubric_dir,
                       client=FakeAzure(good_payload(case.id, score=5)))
    s = await judge.grade(case, make_traj(say="Order 123 is active and in transit."))
    assert s.abstained is False
    assert s.passed is True
    assert s.value == 1.0
    assert s.severity is Severity.MAJOR
    assert s.evidence["weighted_score"] == 5.0


async def test_a_weak_answer_fails_on_the_subsystems_own_decision(tmp_path, monkeypatch):
    """Pass/fail comes from upstream's calibrated `decision`, not a bar this
    wrapper invents."""
    set_azure_env(monkeypatch)
    rubric_dir = write_rubric(tmp_path / "rubrics", "answer_quality", judge="llm_as_judge")
    case = make_case(expected={"rubric_id": "answer_quality"})
    judge = LLMAsJudge(rubric_dir=rubric_dir,
                       client=FakeAzure(good_payload(case.id, score=1)))
    s = await judge.grade(case, make_traj(say="dunno"))
    assert s.passed is False
    assert s.value == 0.0


async def test_abstains_on_a_malformed_judge_response_instead_of_crashing(tmp_path, monkeypatch):
    set_azure_env(monkeypatch)
    rubric_dir = write_rubric(tmp_path / "rubrics", "answer_quality", judge="llm_as_judge")
    case = make_case(expected={"rubric_id": "answer_quality"})
    judge = LLMAsJudge(rubric_dir=rubric_dir, client=FakeAzure("not json at all"))
    s = await judge.grade(case, make_traj(say="Order 123 is on its way."))
    assert s.abstained is True
    assert s.passed is None
