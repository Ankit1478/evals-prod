"""Judge quality operations: is the judge itself any good?

A judge that scores cases is only half the system. These are the checks
that say whether its scores can be believed at all - agreement with humans
(kappa), and resistance to answers that attack the judge.

Everything here delegates to the vendored LLM_AS_JUDGE subsystem
(vendor/llm_judge/). This module is glue: it converts Evals-prod's own
results and gold labels into the shapes that subsystem expects, and reports
honestly when a check cannot run yet.
"""

from __future__ import annotations

import json
from pathlib import Path

from evalkit.schema.case import Frozen
from evalkit.schema.score import CaseResult

# Our own bar. Upstream deliberately has no absolute floor - its calibration
# policy only tolerates a 0.02 kappa DROP against a baseline. An absolute
# floor is a product decision, so it lives here rather than pretending to be
# something the subsystem enforces.
KAPPA_FLOOR = 0.70


class AgreementReport(Frozen):
    cases: int
    agreed: int
    agreement_rate: float
    kappa: float | None
    meets_floor: bool
    floor: float = KAPPA_FLOOR


def load_gold_labels(path: Path) -> dict[str, str]:
    """Human verdicts, one JSON object per line: {"case_id": ..., "human_decision": "PASS"|"FAIL"}

    These are the ground truth a judge is measured against. They must be
    written by a person - generating them with a model measures nothing but
    whether two models agree with each other.
    """
    labels: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        row = json.loads(line)
        decision = str(row["human_decision"]).upper()
        if decision not in {"PASS", "FAIL"}:
            raise ValueError(f"{path}:{lineno} human_decision must be PASS or FAIL")
        labels[row["case_id"]] = decision
    return labels


def judge_decisions(results: list[CaseResult],
                    grader_prefix: str = "judge.") -> dict[str, str]:
    """What each judge grader decided per case, as PASS/FAIL.

    Abstained scores are skipped, not counted as failures - a judge that
    declined to answer has no opinion to compare against a human.
    """
    out: dict[str, str] = {}
    for r in results:
        for s in r.scores:
            if s.grader.startswith(grader_prefix) and not s.abstained and s.passed is not None:
                out[r.case_id] = "PASS" if s.passed else "FAIL"
    return out


def agreement(judge: dict[str, str], human: dict[str, str]) -> AgreementReport:
    """Cohen's kappa between the judge and the humans, on shared cases.

    kappa rather than raw agreement because raw agreement is flattering:
    if 90% of cases pass, a judge that blindly says PASS scores 90% while
    knowing nothing. kappa corrects for agreement by chance.
    """
    from llm_judge.reliability import cohens_kappa

    shared = sorted(set(judge) & set(human))
    if not shared:
        return AgreementReport(cases=0, agreed=0, agreement_rate=0.0,
                               kappa=None, meets_floor=False)

    a = [judge[c] for c in shared]
    b = [human[c] for c in shared]
    agreed = sum(1 for x, y in zip(a, b) if x == y)
    k = cohens_kappa(a, b)
    return AgreementReport(
        cases=len(shared), agreed=agreed,
        agreement_rate=agreed / len(shared),
        kappa=k,
        meets_floor=k is not None and k >= KAPPA_FLOOR,
    )


def stability_dataset(items: list[tuple[str, str, str]]) -> object:
    """(case_id, question, agent_answer) triples -> the vendored EvaluationDataset.

    The vendored case type insists on an expected label, but stability never
    reads one - it compares the judge with itself, not with a human. So the
    label is a placeholder, and it is marked DRAFT so it can never be
    mistaken for gold data by anything that does read labels.
    """
    from llm_judge.contracts import Criterion, EvaluationMode, ReferencePolicy
    from llm_judge.dataset import EvaluationCase, EvaluationDataset, ReviewStatus
    from llm_judge.rubric import ExampleKind

    return EvaluationDataset(cases=[
        EvaluationCase(
            case_id=case_id,
            mode=EvaluationMode.SCORE,
            reference_policy=ReferencePolicy.REFERENCE_FREE,
            question=question,
            candidate_answer=answer,
            case_kind=ExampleKind.BORDERLINE,
            expected_scores={c: 3 for c in Criterion},
            review_status=ReviewStatus.DRAFT,
            review_notes="placeholder label - stability compares the judge "
                         "with itself and never reads it",
        )
        for case_id, question, answer in items
    ])


def run_stability(dataset: object, evaluator, repeats: int = 3) -> object:
    """Ask each judge model the SAME question `repeats` times.

    A judge whose verdict changes between identical calls is producing
    noise, and every score it gives inherits that noise. Returns the
    vendored StabilityReport (per_model consistency, unstable_case_ids...).
    """
    from llm_judge.stability import StabilityRunner

    # allow_drafts: the labels above are placeholders by design.
    return StabilityRunner(evaluator).run(dataset, repeat_count=repeats,
                                          allow_drafts=True)


def run_adversarial(suite_path: Path, judge) -> object:
    """Attack the judge itself: answers that try to talk it into a verdict.

    `judge` must satisfy the vendored AdversarialJudge protocol. Returns the
    subsystem's AdversarialReport (resistance_rate, compromised_case_ids...).
    """
    from llm_judge.adversarial import AdversarialRunner, load_adversarial_jsonl

    suite = load_adversarial_jsonl(suite_path)
    # allow_drafts: the shipped example suite is review_status "draft", and
    # refusing to run it would mean the check can never be tried at all.
    return AdversarialRunner(judge).run(suite, allow_drafts=True)
