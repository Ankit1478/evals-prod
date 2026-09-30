"""Lint flags answer keys that cannot grade, or grade less than they seem."""

from evalkit.lint import lint_cases
from evalkit.schema.case import load_cases

from tests.conftest import make_case


def findings(suite, *cases):
    return {(f.level, f.message.split(" - ")[0]) for f in lint_cases(list(cases), suite)}


def test_a_sound_case_is_clean(tmp_path):
    case = make_case({"required_calls": [{"tool": "cancel_order"}]})
    assert lint_cases([case], tmp_path) == []


def test_errors(tmp_path):
    bad = make_case({"required_calls": [{"tool": "refund_order", "min_times": 2, "max_times": 1}],
                     "forbidden_tools": ["refund_order"],
                     "should_refuse": True, "should_clarify": True,
                     "answer_contains": [["Done", "ok"]], "answer_not_contains": ["done"],
                     "rubric_id": "missing"})
    got = {m for level, m in findings(tmp_path, bad, bad) if level == "error"}
    assert got == {"case id used 2 times",
                   "tool(s) both required and forbidden: refund_order",
                   "refund_order: min_times 2 > max_times 1",
                   "should_refuse and should_clarify are both true",
                   "phrase(s) both required and banned: done",
                   "rubric 'missing' has no file in rubrics/"}


def test_warnings(tmp_path):
    (tmp_path / "rubrics").mkdir()
    (tmp_path / "rubrics" / "answer_quality.json").write_text("{}")
    empty = make_case({}, id="empty")
    judge_only = make_case({"rubric_id": "answer_quality", "reference_answer": "Yes, delivered."},
                           id="judge_only")
    unread = make_case({"answer_contains": ["x"], "reference_answer": "x"}, id="unread")
    got = {(f.case_id, f.message.split(" - ")[0]) for f in lint_cases([empty, judge_only, unread], tmp_path)}
    assert got == {("empty", "the answer key checks nothing"),
                   ("judge_only", "only the judge checks the answer"),
                   ("unread", "reference_answer is set but no judge reads it (no rubric_id)")}


def test_the_support_suite_has_no_errors():
    cases = load_cases("suites/support-agent/cases/regression.jsonl")
    assert not [f for f in lint_cases(cases, "suites/support-agent") if f.level == "error"]
