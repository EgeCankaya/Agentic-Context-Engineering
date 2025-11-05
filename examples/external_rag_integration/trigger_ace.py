"""Run ACE iterations on logged conversations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rich_print

from agentic_context_engineering import ACERunner, Playbook

app = typer.Typer(add_completion=False)


def load_tasks(dataset_path: Path) -> List[str]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [item["input"] for item in data]


@app.command()
def run(
    iterations: int = typer.Option(2, help="Number of ACE iterations to run"),
    tasks_path: str = typer.Option(
        "outputs/conversations/sample_dataset.json",
        help="Path to ACE-formatted dataset",
    ),
    playbook_path: Optional[str] = typer.Option(None, help="Starting playbook (defaults to latest)"),
):
    """Execute ACE Generator → Reflector → Curator cycles on your dataset."""

    runner = ACERunner()
    dataset_path = Path(tasks_path)
    tasks = load_tasks(dataset_path)

    rich_print(f"Running ACE on {len(tasks)} tasks for {iterations} iterations...")

    results = runner.run_iterations(
        num_iterations=iterations,
        tasks=tasks,
        playbook_path=playbook_path,
    )

    final_playbook: Playbook = results["final_playbook"]
    rich_print(f"[bold green]ACE complete[/bold green] - Generated playbook version {final_playbook.version}")


if __name__ == "__main__":
    app()
