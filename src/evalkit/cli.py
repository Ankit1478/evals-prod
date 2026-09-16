"""ef - the command line. Box 2 gives it one command: `ef run`."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from evalkit.adapters.echo import EchoAdapter
from evalkit.graders.composite import grade_all
from evalkit.stats.gate import load_results, run_gate
from evalkit.harness.runner import run_suite
from evalkit.schema.case import load_cases
from evalkit.schema.trajectory import StepType, Trajectory

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
):
    """Run every case against the agent and save the recordings."""
    suite_dir = Path(suite)
    cases = load_cases(suite_dir / "cases" / f"{dataset}.jsonl")
    adapter = EchoAdapter(suite_dir / "echo_script.json")

    console.print(f"[bold]{len(cases)}[/bold] cases  x  [bold]{trials}[/bold] trial(s)"
                  f"  ->  adapter [cyan]{adapter.name}[/cyan]\n")

    run_id, trajectories = asyncio.run(
        run_suite(adapter, cases, Path("runs"), trials)
    )

    by_id = {c.id: c for c in cases}
    results = [grade_all(by_id[t.case_id], t) for t in trajectories]

    # Save the verdicts next to the recordings, so `ef gate` can read them
    # later without re-running (and re-paying for) the agent.
    run_dir = Path("runs") / run_id
    with (run_dir / "scores.jsonl").open("w") as f:
        for r in results:
            row = r.model_dump(mode="json")
            row["kind"] = by_id[r.case_id].kind
            f.write(json.dumps(row) + "\n")

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

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pct = 100 * passed / total if total else 0
    colour = "green" if passed == total else "red"
    console.print(f"\n[bold {colour}]{passed}/{total} passed  ({pct:.0f}%)[/bold {colour}]")
    console.print(f"[dim]recordings: runs/{run_id}/trajectories/[/dim]")


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

    console.print(f"\n[bold]state changes:[/bold] "
                  + (f"{len(t.state_diff)}" if t.state_diff
                     else "[dim]none (no fake DB yet - that is Box 6)[/dim]"))
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
    thresholds = json.loads((Path(suite) / "thresholds.json").read_text())
    results, kinds = load_results(run_dir)
    verdict = run_gate(results, kinds, thresholds)

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
