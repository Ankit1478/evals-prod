"""The judge must ABSTAIN rather than guess whenever it cannot get a clean
answer - no rubric, no credentials, a malformed response. A wrong guess
here is worse than no answer, because an abstained score is excluded from
the pass average while a confidently wrong one would not be.
"""

import json
from pathlib import Path

from evalkit.graders.judge import JevJudge, augment_with_judge
from evalkit.schema.score import CaseResult, Score, Severity

from tests.conftest import make_case, make_traj


def write_rubric(rubric_dir: Path, rubric_id: str = "empathetic_refusal") -> Path:
    rubric_dir.mkdir(exist_ok=True)
    (rubric_dir / f"{rubric_id}.json").write_text(json.dumps({
        "instructions": "Is the refusal empathetic?",
        "criteria": {"true": "warm and apologetic", "false": "cold and procedural"},
    }))
    return rubric_dir


async def test_abstains_when_the_case_has_no_rubric(tmp_path):
    judge = JevJudge(rubric_dir=tmp_path)
    s = await judge.grade(make_case(), make_traj())
    assert s.abstained is True
    assert s.passed is None


async def test_abstains_when_the_rubric_file_is_missing(tmp_path):
    judge = JevJudge(rubric_dir=tmp_path)
    case = make_case(expected={"rubric_id": "does_not_exist"})
    s = await judge.grade(case, make_traj())
    assert s.abstained is True


async def test_abstains_when_credentials_are_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("JEV", raising=False)
    rubric_dir = write_rubric(tmp_path / "rubrics")
    judge = JevJudge(rubric_dir=rubric_dir)
    case = make_case(expected={"rubric_id": "empathetic_refusal"})
    s = await judge.grade(case, make_traj())
    assert s.abstained is True
    assert "JEV" in s.explanation


async def test_passes_on_a_high_confidence_true(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV", "token")
    rubric_dir = write_rubric(tmp_path / "rubrics")

    async def transport(url, headers, body):
        return {"answers": {"meets_rubric": {"noul": 0.92}}}

    judge = JevJudge(rubric_dir=rubric_dir, transport=transport)
    case = make_case(expected={"rubric_id": "empathetic_refusal"})
    traj = make_traj(say="I'm so sorry for the trouble - unfortunately I can't do that.")
    s = await judge.grade(case, traj)
    assert s.passed is True
    assert s.value == 0.92


async def test_fails_on_a_low_confidence_score(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV", "token")
    rubric_dir = write_rubric(tmp_path / "rubrics")

    async def transport(url, headers, body):
        return {"answers": {"meets_rubric": {"noul": 0.1}}}

    judge = JevJudge(rubric_dir=rubric_dir, transport=transport)
    case = make_case(expected={"rubric_id": "empathetic_refusal"})
    s = await judge.grade(case, make_traj(say="Request denied."))
    assert s.passed is False


async def test_abstains_on_a_malformed_response_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV", "token")
    rubric_dir = write_rubric(tmp_path / "rubrics")

    async def broken_transport(url, headers, body):
        return {"unexpected": "shape"}

    judge = JevJudge(rubric_dir=rubric_dir, transport=broken_transport)
    case = make_case(expected={"rubric_id": "empathetic_refusal"})
    s = await judge.grade(case, make_traj())
    assert s.abstained is True


async def test_augment_leaves_a_case_without_a_rubric_completely_unchanged():
    case = make_case()
    result = CaseResult(case_id=case.id, trial_index=0,
                        scores=[Score(grader="g", grader_version="1", value=1.0,
                                     passed=True, severity=Severity.MAJOR)],
                        passed=True, stop_reason="completed")
    judge = JevJudge(rubric_dir=Path("."))
    out = await augment_with_judge(result, case, make_traj(), judge)
    assert out is result


async def test_an_abstained_judge_score_cannot_sink_an_otherwise_passing_case(tmp_path):
    """No rubric file written -> the judge abstains. An abstained score must
    never be able to flip a case that already passed on its own merits."""
    case = make_case(expected={"rubric_id": "missing_rubric"})
    result = CaseResult(case_id=case.id, trial_index=0,
                        scores=[Score(grader="g", grader_version="1", value=1.0,
                                     passed=True, severity=Severity.MAJOR)],
                        passed=True, stop_reason="completed")
    judge = JevJudge(rubric_dir=tmp_path)
    out = await augment_with_judge(result, case, make_traj(), judge)
    assert out.passed is True
    assert any(s.grader == "judge.rubric" for s in out.scores)


async def test_a_confidently_failing_judge_score_can_sink_a_case(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV", "token")
    rubric_dir = write_rubric(tmp_path / "rubrics")

    async def transport(url, headers, body):
        return {"answers": {"meets_rubric": {"noul": 0.05}}}

    case = make_case(expected={"rubric_id": "empathetic_refusal"})
    result = CaseResult(case_id=case.id, trial_index=0,
                        scores=[Score(grader="g", grader_version="1", value=1.0,
                                     passed=True, severity=Severity.MAJOR)],
                        passed=True, stop_reason="completed")
    judge = JevJudge(rubric_dir=rubric_dir, transport=transport)
    out = await augment_with_judge(result, case, make_traj(say="Denied."), judge)
    assert out.passed is False


async def test_jev_model_comes_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV", "token")
    monkeypatch.setenv("JEV_MODEL", "jev-pinned-1")
    sent = {}

    async def transport(url, headers, body):
        sent.update(body)
        return {"answers": {"meets_rubric": {"noul": 0.9}}}

    judge = JevJudge(rubric_dir=write_rubric(tmp_path / "rubrics"), transport=transport)
    await judge.grade(make_case(expected={"rubric_id": "empathetic_refusal"}), make_traj())
    assert sent["model"] == "jev-pinned-1"
