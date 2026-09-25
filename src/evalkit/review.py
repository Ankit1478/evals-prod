"""Human review: the verdict on cases the judges could not agree on.

When two judge models disagree, the judge abstains and the gate stops with
exit 4 (1b REVIEW). A person then reads the case and records PASS or FAIL
here - and the gate treats that as the case's answer-quality verdict.

A review is pinned to the exact reply it judged (its SHA-256). A new run
that produces a different reply needs a new review; an identical reply
(the same agent answering the same way) reuses the old one. A verdict on
one answer is never silently carried over to a different answer.

The file lives in the suite (suites/<suite>/reviews.jsonl) and is meant to
be committed: it is an append-only audit trail of who decided what, and
why. To change a verdict, append a new line - the last one wins.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import field_validator

from evalkit.graders.composite import decide
from evalkit.schema.case import Frozen
from evalkit.schema.score import CaseResult, Score

REVIEW_GRADER = "human.review"
REVIEWS_FILE = "reviews.jsonl"


class Review(Frozen):
    case_id: str
    reply_sha256: str
    human_decision: Literal["PASS", "FAIL"]
    reviewer: str
    note: str = ""
    reviewed_at: str

    @field_validator("reviewer")
    @classmethod
    def _named(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("a review must name its reviewer")
        return v.strip()


def reply_hash(reply: str | None) -> str:
    return hashlib.sha256((reply or "").strip().encode()).hexdigest()


def disputed(result: CaseResult) -> Score | None:
    """The judge score that is waiting on a human, if any."""
    return next((s for s in result.scores
                 if s.abstained and s.evidence.get("requires_human_review") is True),
                None)


def load_reviews(path: Path) -> dict[tuple[str, str], Review]:
    """(case_id, reply_sha256) -> the latest review of that reply."""
    reviews: dict[tuple[str, str], Review] = {}
    if not path.exists():
        return reviews
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            r = Review.model_validate_json(line)
        except ValueError as e:
            raise ValueError(f"{path}:{lineno} is not a valid review: {e}") from e
        reviews[(r.case_id, r.reply_sha256)] = r
    return reviews


def new_review(result: CaseResult, decision: str, reviewer: str,
               note: str = "") -> Review:
    """A review of this result's disputed reply. Raises if nothing is disputed."""
    score = disputed(result)
    if score is None:
        raise ValueError(f"{result.case_id} is not waiting on a human review")
    return Review(case_id=result.case_id,
                  reply_sha256=reply_hash(score.evidence.get("reply")),
                  human_decision=decision.upper(), reviewer=reviewer, note=note,
                  reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def append_review(path: Path, review: Review) -> None:
    with path.open("a") as f:
        f.write(review.model_dump_json() + "\n")


def apply_reviews(results: list[CaseResult],
                  reviews: dict[tuple[str, str], Review]) -> list[CaseResult]:
    """Fold human verdicts into the results they resolve.

    The human verdict takes the disputed judge's place, at the judge's
    severity: it settles the one question the judges split on. It does not
    override any other grader - a human PASS on answer quality does not undo
    a failed safety check.
    """
    out = []
    for r in results:
        score = None if not r.needs_review else disputed(r)   # already resolved: leave it
        review = score and reviews.get((r.case_id, reply_hash(score.evidence.get("reply"))))
        if not review:
            out.append(r)
            continue
        ok = review.human_decision == "PASS"
        human = Score(grader=REVIEW_GRADER, grader_version="1",
                      value=1.0 if ok else 0.0, passed=ok, severity=score.severity,
                      evidence={"reviewer": review.reviewer, "note": review.note,
                                "reviewed_at": review.reviewed_at,
                                "resolves": score.grader,
                                "reply_sha256": review.reply_sha256},
                      explanation=f"human review by {review.reviewer}: "
                                  f"{review.human_decision}"
                                  + (f" - {review.note}" if review.note else ""))
        scores = [*r.scores, human]
        out.append(r.model_copy(update={"scores": scores, "passed": decide(scores)}))
    return out
