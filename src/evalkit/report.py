"""Turn a run's scores.jsonl into one HTML page a human can open and read.

No JS framework, no build step - a single self-contained file, because a
report nobody can open without installing something is a report nobody
reads.
"""

from __future__ import annotations

import html
from pathlib import Path

from evalkit.schema.score import CaseResult
from evalkit.stats.summary import CaseSummary, SuiteSummary

_STYLE = """
body { font-family: -apple-system, sans-serif; margin: 2rem; color: #1a1a1a; }
h1 { margin-bottom: 0; }
.sub { color: #666; margin-top: 0.25rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }
th, td { text-align: left; padding: 0.4rem 0.7rem; border-bottom: 1px solid #eee; }
th { color: #666; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; }
.pass { color: #1a7f37; font-weight: 600; }
.fail { color: #cf222e; font-weight: 600; }
.review { color: #9a6700; font-weight: 600; }
.why { color: #666; font-size: 0.9rem; }
.badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
        font-size: 0.8rem; font-weight: 600; }
.badge.ok { background: #dafbe1; color: #1a7f37; }
.badge.bad { background: #ffebe9; color: #cf222e; }
"""


def _case_rows(results: list[CaseResult]) -> str:
    rows = []
    for r in results:
        if r.passed and r.needs_review:
            mark = '<span class="review">REVIEW</span>'
            why = "; ".join(html.escape(x) for x in r.review_reasons)
        else:
            mark = ('<span class="pass">PASS</span>' if r.passed
                    else '<span class="fail">FAIL</span>')
            why = "; ".join(html.escape(s.explanation) for s in r.scores if s.passed is False)
        rows.append(f"<tr><td>{mark}</td><td>{html.escape(r.case_id)}</td>"
                    f"<td>trial {r.trial_index}</td>"
                    f'<td class="why">{why or "-"}</td></tr>')
    return "\n".join(rows)


def _suite_rows(per_suite: list[SuiteSummary]) -> str:
    rows = []
    for s in per_suite:
        badge = "ok" if s.pass_hat_k == 1.0 else "bad"
        flaky = ", ".join(s.flaky_cases) or "-"
        review = ", ".join(s.pending_cases) or "-"
        rows.append(
            f"<tr><td>{html.escape(s.kind)}</td><td>{s.cases}</td>"
            f"<td>{s.pass_at_1:.0%}</td>"
            f'<td><span class="badge {badge}">{s.pass_hat_k:.0%}</span></td>'
            f"<td>{s.ci_low:.0%} - {s.ci_high:.0%}</td>"
            f"<td>{html.escape(flaky)}</td>"
            f'<td class="review">{html.escape(review)}</td></tr>')
    return "\n".join(rows)


def write_html_report(run_dir: Path, run_id: str, results: list[CaseResult],
                      per_suite: list[SuiteSummary]) -> Path:
    passed = sum(1 for r in results if r.passed)
    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>eval report {html.escape(run_id)}</title>
<style>{_STYLE}</style></head>
<body>
<h1>run {html.escape(run_id)}</h1>
<p class="sub">{passed}/{len(results)} case-trials passed</p>

<h2>reliability</h2>
<table>
<tr><th>kind</th><th>cases</th><th>pass@1</th><th>pass^k</th><th>95% CI</th><th>flaky</th><th>review</th></tr>
{_suite_rows(per_suite)}
</table>

<h2>cases</h2>
<table>
<tr><th></th><th>case</th><th>trial</th><th>why it failed</th></tr>
{_case_rows(results)}
</table>
</body></html>
"""
    out = run_dir / "report.html"
    out.write_text(body)
    return out
