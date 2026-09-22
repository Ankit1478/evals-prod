"""A judge that flips its verdict on identical input is producing noise.

No network: both judge models are fakes behind the real vendored
TwoModelJudge and StabilityRunner, so the stability logic under test is the
upstream code itself.
"""

import json
from itertools import cycle

from evalkit.judgeops import run_stability, stability_dataset

HIGH = {"correctness": 5, "relevance": 5, "completeness": 5, "clarity": 5}
LOW = {"correctness": 1, "relevance": 1, "completeness": 1, "clarity": 1}


class FakeClient:
    """Vendored JudgeTransport: .settings + .evaluate(prompt).

    Cycles through the given score sets, one per call - a single set means a
    perfectly consistent judge, alternating sets mean a judge that flips.
    """

    def __init__(self, deployment: str, case_id: str, *score_sets: dict):
        from llm_judge.settings import AzureJudgeSettings
        self.settings = AzureJudgeSettings(
            endpoint="https://example.openai.azure.com", api_key="k",
            deployment=deployment, api_version="2024-10-21")
        self._case_id = case_id
        self._scores = cycle(score_sets)

    def evaluate(self, prompt):
        from llm_judge.azure_client import RawJudgeResponse, TokenUsage
        content = json.dumps({
            "case_id": self._case_id,
            "scores": [{"criterion": c, "score": n, "evidence": "ev"}
                       for c, n in next(self._scores).items()],
            "summary": "summary",
        })
        return RawJudgeResponse(deployment=self.settings.deployment,
                                mode=prompt.mode,
                                rubric_version=prompt.rubric_version,
                                content=content, usage=TokenUsage())


def judge(terra: FakeClient, luna: FakeClient):
    from llm_judge.multi_judge import TwoModelJudge
    return TwoModelJudge(terra_client=terra, luna_client=luna)


def test_placeholder_labels_are_marked_draft_never_gold():
    """Stability never reads labels, so ours are placeholders - and they
    must be DRAFT so nothing that DOES read labels mistakes them for gold."""
    from llm_judge.dataset import ReviewStatus

    ds = stability_dataset([("s1", "Where is order 123?", "It is in transit.")])
    assert all(c.review_status is ReviewStatus.DRAFT for c in ds.cases)


def test_a_consistent_judge_is_stable():
    ds = stability_dataset([("s1", "Where is order 123?", "It is in transit.")])
    rep = run_stability(ds, judge(FakeClient("gpt-5.6-terra", "s1", HIGH),
                                  FakeClient("gpt-5.6-luna", "s1", HIGH)), repeats=3)
    assert rep.failed_calls == 0
    for s in rep.per_model.values():
        assert s.mean_repeat_consistency == 1.0
        assert s.unstable_case_ids == []


def test_a_judge_that_flips_on_identical_input_is_caught():
    """Terra says PASS, then FAIL, then PASS on the very same answer."""
    ds = stability_dataset([("s1", "Where is order 123?", "It is in transit.")])
    rep = run_stability(ds, judge(FakeClient("gpt-5.6-terra", "s1", HIGH, LOW),
                                  FakeClient("gpt-5.6-luna", "s1", HIGH)), repeats=3)
    by_model = {m.value: s for m, s in rep.per_model.items()}
    assert by_model["gpt-5.6-terra"].unstable_case_ids == ["s1"]
    assert by_model["gpt-5.6-terra"].mean_repeat_consistency < 1.0
    assert by_model["gpt-5.6-luna"].unstable_case_ids == []


def test_a_failed_call_is_recorded_as_a_failure_not_an_inconsistency():
    """Validity before score: the CLI reads failed_calls and reports INVALID."""

    class Broken(FakeClient):
        def evaluate(self, prompt):
            raise RuntimeError("network down")

    ds = stability_dataset([("s1", "Where is order 123?", "It is in transit.")])
    rep = run_stability(ds, judge(Broken("gpt-5.6-terra", "s1", HIGH),
                                  FakeClient("gpt-5.6-luna", "s1", HIGH)), repeats=2)
    assert rep.failed_calls == 2
