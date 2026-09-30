"""Case lint: find weak or broken answer keys BEFORE paying for a run.

A grader can only check what the answer key says. Case 16 had no
answer_contains, so every rule-based grader passed a reply that never
answered the question and only the judge caught it. Lint points at gaps
like that while they are cheap to fix.

  error    the case cannot be graded as written (exit 3 in `ef run`)
  warning  it runs, but checks less than it looks like it does

Pure and generic: it reads only the Case schema and the suite's rubrics/.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from evalkit.schema.case import Case


@dataclass(frozen=True)
class Finding:
    level: str          # "error" | "warning"
    case_id: str
    message: str


def _phrases(entries: list) -> list[str]:
    return [p.lower() for e in entries for p in ([e] if isinstance(e, str) else e)]


def lint_cases(cases: list[Case], suite_dir: str | Path) -> list[Finding]:
    rubric_dir = Path(suite_dir) / "rubrics"
    out: list[Finding] = []

    for cid, n in Counter(c.id for c in cases).items():
        if n > 1:
            out.append(Finding("error", cid, f"case id used {n} times"))

    for c in cases:
        exp = c.expected

        def err(msg: str) -> None:
            out.append(Finding("error", c.id, msg))

        def warn(msg: str) -> None:
            out.append(Finding("warning", c.id, msg))

        required = {rc.tool for rc in exp.required_calls}
        both = sorted(required & set(exp.forbidden_tools))
        if both:
            err(f"tool(s) both required and forbidden: {', '.join(both)}")
        for rc in exp.required_calls:
            if rc.max_times is not None and rc.min_times > rc.max_times:
                err(f"{rc.tool}: min_times {rc.min_times} > max_times {rc.max_times}")
        if exp.should_refuse and exp.should_clarify:
            err("should_refuse and should_clarify are both true")
        clash = sorted(set(_phrases(exp.answer_contains)) & set(_phrases(exp.answer_not_contains)))
        if clash:
            err(f"phrase(s) both required and banned: {', '.join(clash)}")
        if exp.rubric_id and not (rubric_dir / f"{exp.rubric_id}.json").exists():
            err(f"rubric '{exp.rubric_id}' has no file in rubrics/")

        checks_anything = any([
            exp.required_calls, exp.forbidden_tools, exp.final_state,
            exp.forbidden_state_changes, exp.answer_contains,
            exp.answer_not_contains, exp.answer_json_schema, exp.rubric_id,
            exp.should_refuse, exp.should_clarify,
        ])
        if not checks_anything:
            warn("the answer key checks nothing - any reply passes")
        if exp.reference_answer and not exp.rubric_id:
            warn("reference_answer is set but no judge reads it (no rubric_id)")
        if exp.reference_answer and not exp.answer_contains:
            warn("only the judge checks the answer - add answer_contains for "
                 "the fact a correct reply must state")

    return out
