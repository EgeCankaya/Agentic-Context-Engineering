"""
Command-line interface for ACE system.
Provides commands for running iterations, evaluation, and system management.
"""

import json
import logging
import os

# Import verify_setup functions
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .playbook_schema import Playbook
from .runners import ACERunner

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from verify_setup import check_gpu, check_ollama, test_generation

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, config, verbose):
    """ACE - Agentic Context Engineering CLI"""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Store config path in context
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.option("--iterations", "-i", type=int, help="Number of iterations to run")
@click.option("--tasks", "-t", help="Path to tasks file (JSON)")
@click.option("--playbook", "-p", help="Path to initial playbook")
@click.option("--output", "-o", help="Output directory")
@click.pass_context
def run(ctx, iterations, tasks, playbook, output):
    """Run ACE iterations"""
    try:
        # Initialize runner
        runner = ACERunner(config_path=ctx.obj["config_path"])

        # Load tasks if provided
        task_list = None
        if tasks:
            with open(tasks) as f:
                task_data = json.load(f)
                task_list = [item["input"] for item in task_data]

        # Set output directory
        if output:
            runner.output_dir = Path(output)
            runner.output_dir.mkdir(exist_ok=True)

        # Run iterations
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Running ACE iterations...", total=None)

            results = runner.run_iterations(num_iterations=iterations, tasks=task_list, playbook_path=playbook)

        # Display results
        _display_results(results)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.option("--playbook", "-p", required=True, help="Path to playbook to evaluate")
@click.option("--dataset", "-d", help="Path to test dataset")
@click.option("--output", "-o", help="Output file for results")
@click.pass_context
def evaluate(ctx, playbook, dataset, output):
    """Evaluate a playbook against test dataset"""
    try:
        # Initialize runner
        runner = ACERunner(config_path=ctx.obj["config_path"])

        # Load playbook
        playbook_obj = Playbook.from_yaml(playbook)

        # Load dataset if provided
        test_dataset = None
        if dataset:
            with open(dataset) as f:
                test_dataset = json.load(f)

        # Run evaluation
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Evaluating playbook...", total=None)

            results = runner.evaluate_playbook(playbook_obj, test_dataset)

        # Display evaluation results
        _display_evaluation_results(results)

        # Save results if output specified
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"[green]Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.option("--from-version", "-f", required=True, help="First playbook version")
@click.option("--to-version", "-t", required=True, help="Second playbook version")
@click.option("--dataset", "-d", help="Path to test dataset")
@click.option("--output", "-o", help="Output file for comparison")
@click.pass_context
def diff(ctx, from_version, to_version, dataset, output):
    """Compare two playbook versions"""
    try:
        # Initialize runner
        runner = ACERunner(config_path=ctx.obj["config_path"])

        # Load test dataset if provided
        test_dataset = None
        if dataset:
            with open(dataset) as f:
                test_dataset = json.load(f)

        # Run comparison
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Comparing playbooks...", total=None)

            results = runner.compare_playbooks(from_version, to_version, test_dataset)

        # Display comparison results
        _display_comparison_results(results)

        # Save results if output specified
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"[green]Comparison saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.option("--format", "-f", type=click.Choice(["csv", "json"]), default="json", help="Export format")
@click.option("--output", "-o", required=True, help="Output file path")
@click.option("--results-dir", "-r", help="Results directory to export from")
@click.pass_context
def export(ctx, format, output, results_dir):
    """Export results to CSV or JSON"""
    try:
        # Set results directory
        if results_dir:
            results_path = Path(results_dir)
        else:
            runner = ACERunner(config_path=ctx.obj["config_path"])
            results_path = runner.output_dir

        # Find latest results file
        results_files = list(results_path.glob("ace_results_*.json"))
        if not results_files:
            console.print("[red]No results files found[/red]")
            return

        latest_results = max(results_files, key=lambda f: f.stat().st_mtime)

        # Load results
        with open(latest_results) as f:
            results = json.load(f)

        # Export in requested format
        if format == "csv":
            _export_to_csv(results, output)
        else:
            with open(output, "w") as f:
                json.dump(results, f, indent=2, default=str)

        console.print(f"[green]Results exported to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.pass_context
def check_gpu(ctx):
    """Check GPU and Ollama setup"""
    try:
        console.print("[bold blue]ACE System Health Check[/bold blue]")

        # Check GPU
        gpu_ok = check_gpu()
        if not gpu_ok:
            console.print("[red]X GPU check failed[/red]")
            return

        # Check Ollama
        llm = check_ollama()
        if not llm:
            console.print("[red]X Ollama check failed[/red]")
            return

        # Test generation
        gen_ok = test_generation(llm)

        # Summary
        if gpu_ok and llm and gen_ok:
            console.print("[green]✓ ALL CHECKS PASSED - Ready for ACE development![/green]")
        else:
            console.print("[yellow]! Some checks failed - review errors above[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.option("--version", "-v", help="Playbook version to export")
@click.option("--output", "-o", required=True, help="Output file path")
@click.pass_context
def export_context(ctx, version, output):
    """Export playbook context for Project 2 integration"""
    try:
        # Initialize runner
        runner = ACERunner(config_path=ctx.obj["config_path"])

        # Find playbook file
        if version:
            playbook_path = runner.output_dir / f"playbook_v{version}.yaml"
        else:
            # Find latest playbook
            playbook_files = list(runner.output_dir.glob("playbook_v*.yaml"))
            if not playbook_files:
                console.print("[red]No playbook files found[/red]")
                return
            playbook_path = max(playbook_files, key=lambda f: f.stat().st_mtime)

        # Load playbook
        playbook = Playbook.from_yaml(str(playbook_path))

        # Export context
        context = {
            "system_prompt": playbook.context.system_instructions,
            "heuristics": [h.rule for h in playbook.context.heuristics],
            "examples": [{"input": ex.input, "output": ex.output} for ex in playbook.context.few_shot_examples],
            "constraints": playbook.context.constraints,
            "version": playbook.version,
            "metadata": {
                "iteration": playbook.metadata.iteration,
                "accuracy": playbook.metadata.performance_metrics.accuracy,
                "convergence_status": playbook.metadata.convergence_status,
            },
        }

        # Save context
        with open(output, "w") as f:
            json.dump(context, f, indent=2)

        console.print(f"[green]Context exported to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


def _display_results(results: dict):
    """Display ACE iteration results"""
    console.print("\n[bold green]ACE Iteration Results[/bold green]")

    # Summary table
    table = Table(title="Iteration Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Iterations", str(results["total_iterations"]))
    table.add_row("Convergence Reached", str(results["convergence_reached"]))
    table.add_row("Runtime (seconds)", f"{results['runtime_seconds']:.2f}")

    if results["final_playbook"]:
        table.add_row("Final Version", results["final_playbook"].version)
        table.add_row("Heuristics", str(len(results["final_playbook"].context.heuristics)))
        table.add_row("Examples", str(len(results["final_playbook"].context.few_shot_examples)))

    console.print(table)

    # Iteration details
    if results["iterations"]:
        console.print("\n[bold blue]Iteration Details[/bold blue]")
        for i, iteration in enumerate(results["iterations"], 1):
            if iteration["success"]:
                metrics = iteration.get("metrics", {})
                console.print(f"Iteration {i}: ✓ (Score: {metrics.get('avg_reflection_score', 0.0):.3f})")
            else:
                console.print(f"Iteration {i}: ✗ ({iteration.get('error', 'Unknown error')})")


def _display_evaluation_results(results: dict):
    """Display evaluation results"""
    console.print("\n[bold green]Evaluation Results[/bold green]")

    # Performance metrics
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta")

    metrics = results.get("quantitative_metrics", {})
    table.add_row("BLEU Score", f"{metrics.get('bleu', 0.0):.3f}")
    table.add_row("ROUGE-1", f"{metrics.get('rouge1', 0.0):.3f}")
    table.add_row("Exact Match", f"{metrics.get('exact_match', 0.0):.3f}")
    table.add_row("Semantic Similarity", f"{metrics.get('semantic_similarity', 0.0):.3f}")

    console.print(table)

    # Task accuracy
    task_accuracy = results.get("task_accuracy", {})
    if task_accuracy:
        console.print("\n[bold blue]Task Accuracy[/bold blue]")
        for metric, score in task_accuracy.items():
            console.print(f"{metric}: {score:.3f}")


def _display_comparison_results(results: dict):
    """Display comparison results"""
    console.print("\n[bold green]Playbook Comparison[/bold green]")

    # Version info
    console.print(f"Comparing {results['playbook1_version']} vs {results['playbook2_version']}")

    # Improvements
    improvements = results.get("metric_improvements", {})
    if improvements:
        console.print("\n[bold blue]Metric Improvements[/bold blue]")
        for metric, improvement in improvements.items():
            color = "green" if improvement > 0 else "red" if improvement < 0 else "white"
            console.print(f"{metric}: {improvement:+.3f}", style=color)

    # Overall improvement
    overall = results.get("overall_improvement", 0.0)
    color = "green" if overall > 0 else "red" if overall < 0 else "white"
    console.print(f"\nOverall Improvement: {overall:+.1%}", style=color)


def _export_to_csv(results: dict, output_path: str):
    """Export results to CSV format"""
    import pandas as pd

    # Extract metrics history
    metrics_history = results.get("metrics_history", {})
    if metrics_history:
        df = pd.DataFrame(metrics_history)
        df.to_csv(output_path, index=False)
    else:
        # Create basic CSV with summary
        summary_data = {
            "total_iterations": [results.get("total_iterations", 0)],
            "convergence_reached": [results.get("convergence_reached", False)],
            "runtime_seconds": [results.get("runtime_seconds", 0.0)],
        }
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    cli()
