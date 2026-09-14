"""ef - the command line. Box 2 gives it one command: `ef run`."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from evalkit.adapters.echo import EchoAdapter
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

    table = Table(title=f"run {run_id}", header_style="bold")
    table.add_column("case")
    table.add_column("tools called")
    table.add_column("stop", justify="center")
    table.add_column("said", max_width=46, overflow="ellipsis")

    for t in trajectories:
        tools = ", ".join(c.name for c in t.tool_calls) or "[dim]none[/dim]"
        table.add_row(t.case_id, tools, t.stop_reason.value, t.final_output or "")

    console.print(table)
    console.print(f"\n[dim]recordings:[/dim] runs/{run_id}/trajectories/")
    console.print("[yellow]no grading yet - that is Box 4.[/yellow]")


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
