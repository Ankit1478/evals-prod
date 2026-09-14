"""ef - the command line. Box 2 gives it one command: `ef run`."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from evalkit.adapters.echo import EchoAdapter
from evalkit.harness.runner import run_suite
from evalkit.schema.case import load_cases

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
