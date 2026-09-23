"""ef - the command line. Box 2 gives it one command: `ef run`."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import typer
from rich.console import Console
from rich.table import Table

from evalkit.adapters.echo import EchoAdapter
from evalkit.adapters.inprocess import InProcessAdapter
from evalkit.graders.composite import grade_all
from evalkit.graders.judge import JevJudge, augment_with_judge
from evalkit.graders.llm_as_judge import LLMAsJudge
from evalkit.report import write_html_report
from evalkit.stats.compare import diff_suites
from evalkit.stats.gate import load_results, run_gate
from evalkit.stats.summary import summarise_cases, summarise_suite
from evalkit.harness.runner import run_suite
from evalkit.schema.case import load_cases
from evalkit.schema.trajectory import StepType, Trajectory
from evalkit.store import JsonlStore, RunSummary

app = typer.Typer(add_completion=False, help="evalkit - agent evaluation harness")
console = Console()


@app.callback()
def main() -> None:
    """Keeps `ef` in subcommand mode, so `ef run`, `ef show`, `ef gate` all work."""


@app.command()
def run(
    suite: str = typer.Argument(..., help="path to a suite, e.g. suites/support-agent"),
    dataset: str = typer.Option("regression", help="which cases file to run"),
    trials: int = typer.Option(1, help="how many times to run each case"),
    adapter_name: str = typer.Option("echo", "--adapter",
                                     help="echo (scripted fake) or inprocess (real agent)"),
    target: str = typer.Option("agent.support_agent:run_agent", "--target",
                               help="module:function, for --adapter inprocess"),
    model: str | None = typer.Option(None, "--model",
                                     help="the agent's model (default: AGENT_MODEL in .env), "
                                          "e.g. openai:gpt-4o-mini, anthropic:claude-opus-5, "
                                          "bedrock:anthropic.claude-sonnet-5, "
                                          "foundry:claude-opus-5, azure:<deployment>"),
    user_model: str | None = typer.Option(None, "--user-model",
                                          help="model that plays the user in multi-turn "
                                               "cases (default: USER_MODEL in .env)"),
):
    """Run every case against the agent and save the recordings."""
    import os

    load_dotenv()                       # picks up keys and model choices from .env
    # A flag wins; otherwise .env decides. Either way the manifest records
    # the model actually used, so a run stays reproducible.
    model = model or os.environ.get("AGENT_MODEL")
    user_model = user_model or os.environ.get("USER_MODEL")
    suite_dir = Path(suite)
    cases = load_cases(suite_dir / "cases" / f"{dataset}.jsonl")

    if adapter_name == "echo":
        adapter = EchoAdapter(suite_dir / "echo_script.json")
    elif adapter_name == "inprocess":
        adapter = InProcessAdapter(target, model=model, user_model=user_model)
    else:
        raise typer.BadParameter(f"unknown adapter: {adapter_name}")

    console.print(f"[bold]{len(cases)}[/bold] cases  x  [bold]{trials}[/bold] trial(s)"
                  f"  ->  adapter [cyan]{adapter.name}[/cyan]\n")

    faults_path = suite_dir / "faults.json"
    faults = json.loads(faults_path.read_text()) if faults_path.exists() else {}
    faults = {k: v for k, v in faults.items() if not k.startswith("_")}

    store = JsonlStore(Path("runs"))
    run_id, trajectories = asyncio.run(
        run_suite(adapter, cases, Path("runs"), trials, faults_by_case=faults,
                  store=store)
    )

    by_id = {c.id: c for c in cases}
    results = [grade_all(by_id[t.case_id], t) for t in trajectories]

    # Judges are async and networked - only pay for them on cases that
    # actually ask for one. Every other case is untouched by their presence.
    # Each judge abstains on rubrics addressed to the other one.
    if any(c.expected.rubric_id for c in cases):
        rubric_dir = suite_dir / "rubrics"
        judges = [JevJudge(rubric_dir=rubric_dir), LLMAsJudge(rubric_dir=rubric_dir)]

        async def _apply_judges() -> list:
            out = []
            for r, t in zip(results, trajectories):
                for j in judges:
                    r = await augment_with_judge(r, by_id[r.case_id], t, j)
                out.append(r)
            return out

        results = asyncio.run(_apply_judges())

    # Save the verdicts next to the recordings, so `ef gate` can read them
    # later without re-running (and re-paying for) the agent.
    run_dir = Path("runs") / run_id
    for r in results:
        store.write_result(r, kind=by_id[r.case_id].kind)

    table = Table(title=f"run {run_id}", header_style="bold")
    table.add_column("", justify="center", width=4, no_wrap=True)
    table.add_column("case")
    table.add_column("why it failed", max_width=62, overflow="fold")

    for r in results:
        if r.passed:
            table.add_row("[green]PASS[/green]", r.case_id, "[dim]-[/dim]")
        else:
            why = "\n".join(
                f"[red]{s.grader}[/red]  {s.explanation}"
                for s in r.scores if s.passed is False
            )
            table.add_row("[red]FAIL[/red]", f"[red]{r.case_id}[/red]", why)

    console.print(table)

    kinds = {c.id: c.kind for c in cases}
    per_case = summarise_cases(results, kinds)
    per_suite = summarise_suite(per_case)

    stats = Table(title="reliability", header_style="bold")
    stats.add_column("suite")
    stats.add_column("cases", justify="right")
    stats.add_column("pass@1", justify="right")
    stats.add_column(f"pass^{trials}", justify="right")
    stats.add_column("95% CI", justify="center")
    stats.add_column("flaky")

    for s_ in per_suite:
        colour = "green" if s_.pass_hat_k == 1.0 else "red"
        stats.add_row(
            s_.kind, str(s_.cases),
            f"{s_.pass_at_1:.0%}",
            f"[{colour}]{s_.pass_hat_k:.0%}[/{colour}]",
            f"{s_.ci_low:.0%} - {s_.ci_high:.0%}",
            ", ".join(s_.flaky_cases) or "[dim]none[/dim]",
        )
    console.print()
    console.print(stats)

    flaky = [c.case_id for c in per_case if c.flaky]
    if flaky:
        console.print(f"\n[yellow]flaky:[/yellow] "
                      + ", ".join(f"{c.case_id} ({c.passed}/{c.trials})"
                                  for c in per_case if c.flaky))
        console.print("[dim]passed sometimes, failed sometimes - the most "
                      "dangerous result there is[/dim]")

    reliable = sum(1 for c in per_case if c.pass_hat_k == 1.0)
    console.print(f"\n[bold]{reliable}/{len(per_case)} cases passed every "
                  f"trial[/bold]  [dim](pass^{trials})[/dim]")
    console.print(f"[dim]recordings: runs/{run_id}/trajectories/[/dim]")

    manifest = json.loads((run_dir / "manifest.json").read_text())
    store.finish_run(RunSummary(
        run_id=run_id, suite=manifest["suite"], started_at=manifest["started_at"],
        finished_at=datetime.now(timezone.utc).isoformat(),
        cases=len(cases), trials=trials, results=len(results),
        passed=sum(1 for r in results if r.passed)))

    report_path = write_html_report(run_dir, run_id, results, per_suite)
    console.print(f"[dim]report: {report_path}[/dim]")


@app.command()
def runs(
    suite: str | None = typer.Option(None, help="only runs of this suite"),
    limit: int = typer.Option(20, help="how many to show"),
):
    """List past runs, newest first."""
    table = Table(header_style="bold")
    for col in ("run", "suite", "cases", "trials", "passed", "finished"):
        table.add_column(col)
    for s_ in JsonlStore(Path("runs")).list_runs(suite=suite, limit=limit):
        passed = f"{s_.passed}/{s_.results}" if s_.results else "[dim]-[/dim]"
        table.add_row(s_.run_id, s_.suite, str(s_.cases), str(s_.trials), passed,
                      s_.finished_at[:19] if s_.finished_at
                      else "[yellow]open[/yellow]" if not s_.results
                      else "[dim]no summary[/dim]")
    console.print(table)


if __name__ == "__main__":
    app()


def _latest_run(runs_dir: Path) -> Path:
    runs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    if not runs:
        raise typer.BadParameter("no runs yet - do `ef run suites/support-agent` first")
    return runs[-1]


@app.command()
def trace(
    case_id: str = typer.Argument(..., help="which case to look at"),
    run: str | None = typer.Option(None, help="run id (default: the latest run)"),
    trial: int = typer.Option(0, help="which trial"),
):
    """Read one recording in human-readable form.

    Looking at your data must be frictionless. It is the highest-leverage
    habit in the whole system - graders are only trusted once you have read
    many trajectories by hand.
    """
    runs_dir = Path("runs")
    run_dir = runs_dir / run if run else _latest_run(runs_dir)
    path = run_dir / "trajectories" / f"{case_id}__trial{trial}.json"
    if not path.exists():
        raise typer.BadParameter(f"no recording at {path}")

    t = Trajectory.model_validate_json(path.read_text())

    console.print(f"\n[bold]{t.case_id}[/bold]  [dim]trial {t.trial_index}  ·  "
                  f"run {t.run_id}[/dim]")
    console.print(f"stop: [cyan]{t.stop_reason.value}[/cyan]"
                  + (f"  [red]{t.error}[/red]" if t.error else "") + "\n")

    for s in t.steps:
        n = f"[dim]{s.index:>2}[/dim]"
        if s.type is StepType.USER:
            console.print(f"{n}  [blue]user[/blue]       {s.content}")
        elif s.type is StepType.ASSISTANT:
            console.print(f"{n}  [green]assistant[/green]  {s.content}")
        elif s.type is StepType.TOOL_CALL and s.tool_call:
            c = s.tool_call
            colour = {"committed": "green", "acknowledged": "yellow",
                      "attempted": "yellow", "failed": "red"}[c.status.value]
            line = (f"{n}  [magenta]tool[/magenta]       {c.name}"
                    f"({json.dumps(c.arguments)})  "
                    f"[{colour}]{c.status.value.upper()}[/{colour}]")
            if c.error:
                line += f"  [red]{c.error}[/red]"
            console.print(line)
        elif s.type is StepType.TOOL_RESULT:
            console.print(f"{n}  [dim]result     {s.content}[/dim]")

    if t.state_diff:
        console.print("\n[bold]state changes:[/bold]")
        for c in t.state_diff:
            console.print(f"     [yellow]{c.entity}[/yellow].{c.field}  "
                          f"{c.before!r} [dim]->[/dim] [bold]{c.after!r}[/bold]")
    else:
        console.print("\n[bold]state changes:[/bold] [dim]none - "
                      "the database is unchanged[/dim]")
    console.print(f"[dim]tokens {t.usage.input_tokens} in / "
                  f"{t.usage.output_tokens} out  ·  {t.latency_ms} ms[/dim]\n")


@app.command()
def gate(
    run: str | None = typer.Option(None, help="run id (default: the latest run)"),
    suite: str = typer.Option("suites/support-agent", help="suite holding thresholds.json"),
):
    """Decide: ship or do not ship. Exits 0 (pass), 2 (failed), 3 (invalid).

    CI reads the exit code. A crashed eval must exit 3 - never 0.
    """
    run_dir = Path("runs") / run if run else _latest_run(Path("runs"))
    suite_dir = Path(suite)
    thresholds = json.loads((suite_dir / "thresholds.json").read_text())
    results, kinds = load_results(run_dir)
    verdict = run_gate(results, kinds, thresholds, suite_dir=suite_dir)

    console.print(f"\n[bold]gate[/bold]  [dim]{run_dir.name}[/dim]\n")
    for st in verdict.stages:
        mark = "[green]PASS[/green]" if st.passed else "[red]FAIL[/red]"
        console.print(f"  {mark}  [bold]{st.stage}[/bold]  {st.detail}")

    if verdict.passed:
        console.print("\n[bold green]SHIP IT[/bold green]  [dim]exit 0[/dim]\n")
    else:
        word = "INVALID RUN" if verdict.exit_code == 3 else "BLOCKED"
        console.print(f"\n[bold red]{word}[/bold red]  "
                      f"[dim]stopped at {verdict.blocked_by} · "
                      f"exit {verdict.exit_code}[/dim]\n")
    raise typer.Exit(verdict.exit_code)


@app.command()
def report(
    run: str | None = typer.Option(None, help="run id (default: the latest run)"),
):
    """Rebuild runs/<id>/report.html from a saved scores.jsonl."""
    run_dir = Path("runs") / run if run else _latest_run(Path("runs"))
    results, kinds = load_results(run_dir)
    per_case = summarise_cases(results, kinds)
    per_suite = summarise_suite(per_case)
    path = write_html_report(run_dir, run_dir.name, results, per_suite)
    console.print(f"[green]wrote[/green] {path}")


@app.command()
def diff(
    before: str = typer.Option(..., help="run id of the baseline"),
    after: str = typer.Option(..., help="run id of the new run"),
):
    """Compare two runs case-by-case. Tells you if a change is REAL or NOISE.

    Uses a paired bootstrap: each case is matched by id across both runs,
    so a flaky case cannot masquerade as two different data points.
    """
    runs_dir = Path("runs")
    b_results, b_kinds = load_results(runs_dir / before)
    a_results, a_kinds = load_results(runs_dir / after)
    b_summary = summarise_cases(b_results, b_kinds)
    a_summary = summarise_cases(a_results, a_kinds)
    diffs = diff_suites(b_summary, a_summary)

    if not diffs:
        console.print("[yellow]no cases in common between these two runs[/yellow]")
        raise typer.Exit(1)

    table = Table(title=f"diff  {before}  ->  {after}", header_style="bold")
    table.add_column("kind")
    table.add_column("cases", justify="right")
    table.add_column("improved", justify="right")
    table.add_column("regressed", justify="right")
    table.add_column("mean delta", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("verdict")

    for d in diffs:
        if d.significant and d.mean_delta > 0:
            verdict = "[green]REAL IMPROVEMENT[/green]"
        elif d.significant and d.mean_delta < 0:
            verdict = "[red]REAL REGRESSION[/red]"
        else:
            verdict = "[yellow]NOISE[/yellow]  [dim](not enough evidence)[/dim]"
        table.add_row(d.kind, str(d.cases), str(d.improved), str(d.regressed),
                     f"{d.mean_delta:+.0%}",
                     f"{d.ci_low:+.0%} .. {d.ci_high:+.0%}", verdict)
    console.print(table)


@app.command("judge-check")
def judge_check(
    run: str | None = typer.Option(None, help="run id (default: the latest run)"),
    suite: str = typer.Option("suites/support-agent", help="suite directory"),
):
    """Is the JUDGE any good? Agreement with humans, and rubric approval.

    A judge that scores your cases is only half the system - this is the
    half that says whether those scores can be believed.
    """
    from evalkit.judgeops import (agreement, judge_decisions, load_gold_labels)
    from evalkit.stats.gate import check_rubric_approval

    load_dotenv()
    run_dir = Path("runs") / run if run else _latest_run(Path("runs"))
    suite_dir = Path(suite)
    results, _kinds = load_results(run_dir)

    console.print(f"\n[bold]judge check[/bold]  [dim]{run_dir.name}[/dim]\n")

    # --- rubric governance -------------------------------------------------
    ok, detail = check_rubric_approval(suite_dir / "rubric_approval.json")
    mark = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
    console.print(f"  {mark}  [bold]rubric approval[/bold]  {detail}")

    # --- agreement with humans --------------------------------------------
    gold_path = suite_dir / "gold_labels.jsonl"
    judged = judge_decisions(results)
    if not judged:
        console.print("  [yellow]SKIP[/yellow]  [bold]agreement[/bold]  "
                      "no judge produced a verdict in this run "
                      "(every judge abstained)")
    elif not gold_path.exists():
        console.print(f"  [yellow]SKIP[/yellow]  [bold]agreement[/bold]  "
                      f"no human labels at {gold_path} - a judge cannot be "
                      f"validated without them")
    else:
        rep = agreement(judged, load_gold_labels(gold_path))
        if rep.kappa is None:
            console.print("  [yellow]SKIP[/yellow]  [bold]agreement[/bold]  "
                          f"{rep.cases} shared case(s) - too few, or all one verdict")
        else:
            m = "[green]PASS[/green]" if rep.meets_floor else "[red]FAIL[/red]"
            console.print(f"  {m}  [bold]agreement[/bold]  kappa {rep.kappa:.2f} "
                          f"(floor {rep.floor:.2f})  ·  agreed {rep.agreed}/{rep.cases} "
                          f"({rep.agreement_rate:.0%})")

    console.print("\n[dim]kappa, not raw agreement: if 90% of cases pass, a judge "
                  "that always says PASS looks 90% right while knowing nothing.[/dim]\n")


@app.command("judge-attack")
def judge_attack(
    suite_file: str = typer.Option(
        "vendor/llm_judge/datasets/adversarial_cases.example.jsonl",
        help="adversarial suite jsonl"),
):
    """Attack the JUDGE: can a candidate answer talk it into a verdict?

    The agent under test is not involved. This measures whether the judge
    itself can be steered by text embedded in what it is grading - a judge
    that can be talked into PASS is not a judge.
    """
    from evalkit.judgeops import run_adversarial

    load_dotenv()
    try:
        from evalkit.graders.llm_as_judge import build_two_model_judge, judge_backend
        judge = build_two_model_judge(*judge_backend())
    except Exception as e:
        console.print(f"[red]cannot build the judge:[/red] {type(e).__name__}: {e}")
        console.print("[dim]set OPENAI_API_KEY in .env, and JUDGE_MODEL / "
                      "JUDGE_MODEL_2 to two different models[/dim]")
        raise typer.Exit(3)

    try:
        report = run_adversarial(Path(suite_file), judge)
    except Exception as e:
        console.print(f"[red]attack run failed:[/red] {type(e).__name__}: {e}")
        raise typer.Exit(3)

    # Validity before score, same rule as the gate. A probe that never
    # reached the judge is not a probe the judge failed - reporting
    # "resistance 0%" for an unreachable endpoint would be the exact
    # confusion this harness exists to prevent.
    if report.error_cases:
        console.print(f"\n[bold red]INVALID[/bold red]  "
                      f"{report.error_cases}/{report.total_cases} probe(s) "
                      f"errored - the judge was not reached")
        console.print(f"[dim]{', '.join(report.error_case_ids[:4])}[/dim]")
        console.print("[dim]this is not a resistance score: an unreachable "
                      "judge has not been tested at all[/dim]")
        raise typer.Exit(3)

    rate = report.resistance_rate
    colour = "green" if rate == 1.0 else "red"
    console.print(f"\n[bold]judge attack[/bold]  resistance "
                  f"[{colour}]{rate:.0%}[/{colour}]  "
                  f"[dim]({report.total_cases} probes)[/dim]")
    if report.compromised_case_ids:
        console.print(f"[red]compromised:[/red] "
                      f"{', '.join(report.compromised_case_ids)}")
        console.print("[dim]the judge was talked into the wrong verdict - "
                      "every score it produced is suspect[/dim]")
    raise typer.Exit(0 if rate == 1.0 else 2)


@app.command("judge-stability")
def judge_stability(
    run: str | None = typer.Option(None, help="run id whose answers to re-judge (default: latest)"),
    suite: str = typer.Option("suites/support-agent", help="suite directory"),
    repeats: int = typer.Option(3, help="identical calls per case per model (min 2)"),
    all_cases: bool = typer.Option(False, "--all-cases",
                                   help="re-judge every case, not only answer-quality ones"),
):
    """Ask the judges the SAME question several times. Same answer each time?

    A judge that flips its verdict on identical input is producing noise,
    and every score it gives inherits that noise.
    """
    from evalkit.graders.llm_as_judge import (JUDGE_KEY, build_two_model_judge,
                                              judge_backend)
    from evalkit.judgeops import run_stability, stability_dataset

    load_dotenv()
    run_dir = Path("runs") / run if run else _latest_run(Path("runs"))
    suite_dir = Path(suite)
    cases = load_cases(suite_dir / "cases" / "regression.jsonl")

    def addressed_here(c) -> bool:
        path = suite_dir / "rubrics" / f"{c.expected.rubric_id}.json"
        return (c.expected.rubric_id is not None and path.exists()
                and json.loads(path.read_text()).get("judge") == JUDGE_KEY)

    items = []
    for c in cases:
        if not (all_cases or addressed_here(c)):
            continue
        tpath = run_dir / "trajectories" / f"{c.id}__trial0.json"
        if not tpath.exists():
            continue
        answer = (Trajectory.model_validate_json(tpath.read_text()).final_output or "").strip()
        question = "\n".join(m.content for m in c.input.messages if m.role == "user").strip()
        if answer and question:
            items.append((c.id, question, answer))

    if not items:
        console.print("[yellow]no answers to re-judge in this run[/yellow]")
        raise typer.Exit(3)

    try:
        judge = build_two_model_judge(*judge_backend())
    except Exception as e:
        console.print(f"[red]cannot build the judge:[/red] {type(e).__name__}: {e}")
        raise typer.Exit(3)

    calls = len(items) * 2 * repeats
    console.print(f"\n[bold]judge stability[/bold]  {len(items)} case(s) x 2 models "
                  f"x {repeats} repeats = {calls} calls\n")
    report = run_stability(stability_dataset(items), judge, repeats=repeats)

    # Validity before score: a failed call is not an inconsistent verdict.
    if report.failed_calls:
        console.print(f"[bold red]INVALID[/bold red]  {report.failed_calls}/"
                      f"{report.total_calls} call(s) failed - that is not a "
                      f"stability result")
        raise typer.Exit(3)

    table = Table(header_style="bold")
    table.add_column("model")
    table.add_column("cases", justify="right")
    table.add_column("consistency", justify="right")
    table.add_column("stable", justify="right")
    table.add_column("unstable", justify="right")
    table.add_column("unstable cases", overflow="fold")
    unstable_any = False
    for model, s in report.per_model.items():
        unstable_any |= s.unstable_cases > 0
        c = s.mean_repeat_consistency
        colour = "green" if s.unstable_cases == 0 else "red"
        table.add_row(model.value, str(s.evaluated_cases),
                      "-" if c is None else f"[{colour}]{c:.0%}[/{colour}]",
                      str(s.stable_cases), str(s.unstable_cases),
                      ", ".join(s.unstable_case_ids) or "[dim]none[/dim]")
    console.print(table)
    for w in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    console.print("[dim]consistency = share of repeats matching the most common "
                  "verdict. Below 100% means the judge is partly guessing.[/dim]\n")
    raise typer.Exit(2 if unstable_any else 0)
