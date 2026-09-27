"""Same question to the same judge -> answered once, not paid twice."""

from evalkit.graders.judge_cache import CachedJudge, cache_enabled
from evalkit.schema.score import Score, Severity

from tests.conftest import make_case, make_traj


class CountingJudge:
    name, version, severity = "judge.fake", "1", Severity.MAJOR

    def __init__(self, score_kwargs=None):
        self.calls = 0
        self._kw = score_kwargs or {"value": 1.0, "passed": True}

    def cache_key(self, case, traj):
        return {"case": case.id, "reply": traj.final_output}

    async def grade(self, case, traj):
        self.calls += 1
        return Score(grader=self.name, grader_version="1", severity=self.severity, **self._kw)


async def test_an_identical_question_is_answered_from_the_cache(tmp_path):
    inner = CountingJudge()
    judge = CachedJudge(inner, tmp_path / "c.jsonl")
    a = await judge.grade(make_case(), make_traj(say="same"))
    b = await judge.grade(make_case(), make_traj(say="same"))
    assert inner.calls == 1 and judge.hits == 1
    assert b.passed == a.passed and b.evidence["cached"] is True


async def test_the_cache_survives_to_the_next_run(tmp_path):
    path = tmp_path / "c.jsonl"
    await CachedJudge(CountingJudge(), path).grade(make_case(), make_traj(say="x"))
    inner = CountingJudge()
    await CachedJudge(inner, path).grade(make_case(), make_traj(say="x"))
    assert inner.calls == 0


async def test_a_different_reply_is_judged_fresh(tmp_path):
    inner = CountingJudge()
    judge = CachedJudge(inner, tmp_path / "c.jsonl")
    await judge.grade(make_case(), make_traj(say="one"))
    await judge.grade(make_case(), make_traj(say="two"))
    assert inner.calls == 2


async def test_an_error_abstain_is_never_cached(tmp_path):
    inner = CountingJudge({"value": 0.0, "passed": None, "abstained": True,
                           "evidence": {"error": "timeout"}})
    judge = CachedJudge(inner, tmp_path / "c.jsonl")
    await judge.grade(make_case(), make_traj())
    await judge.grade(make_case(), make_traj())
    assert inner.calls == 2


async def test_a_split_verdict_is_cached(tmp_path):
    inner = CountingJudge({"value": 0.0, "passed": None, "abstained": True,
                           "evidence": {"requires_human_review": True}})
    judge = CachedJudge(inner, tmp_path / "c.jsonl")
    await judge.grade(make_case(), make_traj())
    s = await judge.grade(make_case(), make_traj())
    assert inner.calls == 1 and s.evidence["requires_human_review"] is True


async def test_a_judge_that_does_not_apply_is_not_cached(tmp_path):
    class NotApplicable(CountingJudge):
        def cache_key(self, case, traj):
            return None
    inner = NotApplicable()
    judge = CachedJudge(inner, tmp_path / "c.jsonl")
    await judge.grade(make_case(), make_traj())
    await judge.grade(make_case(), make_traj())
    assert inner.calls == 2
    assert not (tmp_path / "c.jsonl").exists()


def test_a_torn_cache_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"key": "a", "score": {}}\n{"key": "b", "sco')
    CachedJudge(CountingJudge(), path)          # must not raise


def test_the_cache_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("EVAL_JUDGE_CACHE", "0")
    assert cache_enabled() is False
    monkeypatch.delenv("EVAL_JUDGE_CACHE")
    assert cache_enabled() is True


def test_changing_the_rubric_changes_the_key(tmp_path):
    """Edit one word of the rubric and every cached verdict for it is stale."""
    import json
    from evalkit.graders.judge import JevJudge

    d = tmp_path / "rubrics"
    d.mkdir()
    (d / "tone.json").write_text(json.dumps({"judge": "jev", "instructions": "be kind"}))
    j = JevJudge(d)
    case = make_case(expected={"rubric_id": "tone"})
    k1 = j.cache_key(case, make_traj())
    (d / "tone.json").write_text(json.dumps({"judge": "jev", "instructions": "be warm"}))
    assert j.cache_key(case, make_traj()) != k1
