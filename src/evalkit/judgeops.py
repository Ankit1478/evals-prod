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


class GoldLabel(Frozen):
    """One human verdict. `reply_sha256` pins it to the exact reply the
    human read (written by `ef label`); a label without one is case-level,
    from before pinning, and matches whatever reply the run produced."""
    case_id: str
    human_decision: str
    reply_sha256: str | None = None
    reviewer: str = ""
    note: str = ""
    labeled_at: str = ""


def load_gold_label_rows(path: Path) -> list[GoldLabel]:
    rows: list[GoldLabel] = []
    if not path.exists():
        return rows
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        row = GoldLabel.model_validate_json(line)
        decision = row.human_decision.upper()
        if decision not in {"PASS", "FAIL"}:
            raise ValueError(f"{path}:{lineno} human_decision must be PASS or FAIL")
        rows.append(row.model_copy(update={"human_decision": decision}))
    return rows


def reply_of(result: CaseResult) -> str | None:
    """The reply a judge saw for this result (judges record it as evidence)."""
    for s in result.scores:
        if s.grader.startswith("judge.") and "reply" in s.evidence:
            return s.evidence["reply"]
    return None


def labels_for_run(rows: list[GoldLabel], results: list[CaseResult]) -> dict[str, str]:
    """case_id -> human verdict, for the labels that apply to THIS run.

    A pinned label counts only when the run produced the very reply the
    human read. Comparing a judge's verdict on one reply with a human's
    verdict on a different one would measure nothing.
    """
    from evalkit.review import reply_hash

    replies = {r.case_id: reply_hash(reply_of(r)) for r in results if reply_of(r) is not None}
    out: dict[str, str] = {}
    for row in rows:               # later lines win: a relabel replaces the old one
        if row.reply_sha256 is None or replies.get(row.case_id) == row.reply_sha256:
            out[row.case_id] = row.human_decision
    return out


ReplyKey = tuple  # (case_id, reply_sha256) - one judged reply, whatever trial produced it


def judged_replies(results: list[CaseResult], grader: str) -> dict[ReplyKey, str]:
    """One judge's PASS/FAIL per REPLY, not per case.

    With --trials k a case has k replies, each judged and each labellable on
    its own. Keying by case_id alone would keep only the last trial and throw
    the rest of the evidence away.
    """
    from evalkit.review import reply_hash

    out: dict[ReplyKey, str] = {}
    for r in results:
        reply = reply_of(r)
        for s in r.scores:
            if s.grader == grader and not s.abstained and s.passed is not None and reply is not None:
                out[(r.case_id, reply_hash(reply))] = "PASS" if s.passed else "FAIL"
    return out


def labels_by_reply(rows: list[GoldLabel], results: list[CaseResult]) -> dict[ReplyKey, str]:
    """Human verdicts per reply present in this run. A pinned label matches
    its own reply; an unpinned (case-level) one matches every reply of the case."""
    from evalkit.review import reply_hash

    keys = {(r.case_id, reply_hash(reply_of(r))) for r in results if reply_of(r) is not None}
    out: dict[ReplyKey, str] = {}
    for row in rows:
        for key in keys:
            if key[0] == row.case_id and row.reply_sha256 in (None, key[1]):
                out[key] = row.human_decision
    return out


def threshold_pairs(results: list[CaseResult], grader: str,
                    human: dict[ReplyKey, str]) -> list[tuple[float, str]]:
    """(judge probability, human verdict) per labelled reply, for suggest_threshold."""
    from evalkit.review import reply_hash

    pairs = []
    for r in results:
        reply = reply_of(r)
        if reply is None:
            continue
        key = (r.case_id, reply_hash(reply))
        for s in r.scores:
            if (s.grader == grader and not s.abstained and "threshold" in s.evidence
                    and key in human):
                pairs.append((s.value, human[key]))
    return pairs


def suggest_threshold(pairs: list[tuple[float, str]], min_labels: int = 10
                      ) -> tuple[float, float] | None:
    """(judge score, human verdict) pairs -> (best threshold, its agreement).

    Tries a cut between every pair of neighbouring scores and keeps the one
    that agrees with the humans most often. None with fewer than
    `min_labels` pairs - a threshold fitted to a handful of labels is noise.
    """
    if len(pairs) < min_labels:
        return None
    values = sorted({v for v, _ in pairs})
    cuts = [0.0] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [1.0]

    def agree(t: float) -> float:
        return sum((v >= t) == (h == "PASS") for v, h in pairs) / len(pairs)

    best = max(cuts, key=lambda t: (agree(t), -abs(t - 0.5)))
    return round(best, 3), agree(best)


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
