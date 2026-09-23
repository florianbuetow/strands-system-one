"""Score every run, check the published runs against the board, and render the results tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from openjev.benchmark import run_path
from openjev.config import Config
from openjev.jevals import Task, load_board, load_task, reference_run_path
from openjev.scoring import Metrics, Run, check_against_board, metrics, prior_run, read_run, resamples


@dataclass(frozen=True)
class Row:
    """One results row."""

    run: Run
    metrics: Metrics


@dataclass(frozen=True)
class TaskResults:
    """All rows for one task, plus local runs that are not finished yet."""

    task: Task
    rows: list[Row]
    incomplete: list[str]


def score_task(config: Config, task_id: str) -> TaskResults:
    """Score the label prior, the published reference runs and every finished local run.

    Raises:
        ValueError: If a published run does not reproduce its board row.
    """
    bench = config.benchmark
    task = load_task(bench, task_id)
    samples = resamples(len(task.items), bench.bootstrap_resamples, bench.bootstrap_seed)
    board = load_board(bench)
    full_max = config.verbalized.full_listing_max_options
    rows = [Row(prior_run(task, bench.epochs), metrics(task, prior_run(task, bench.epochs), bench.epochs, samples))]
    for system in bench.reference_systems:
        run = read_run(reference_run_path(bench, system, task_id), task, full_max, local=False)
        result = metrics(task, run, bench.epochs, samples)
        check_against_board(task, run, result, board)
        rows.append(Row(run, result))
    incomplete: list[str] = []
    expected = len(task.items) * bench.epochs
    for model in config.models:
        for method in bench.methods:
            path = run_path(config, model, method, task_id)
            if not path.exists():
                incomplete.append(f"{model.display_name} ({method}): not started")
                continue
            run = read_run(path, task, full_max, local=True)
            if len(run.decisions) < expected:
                incomplete.append(f"{model.display_name} ({method}): {len(run.decisions)}/{expected} decisions")
                continue
            rows.append(Row(run, metrics(task, run, bench.epochs, samples)))
    rows.sort(key=lambda row: row.metrics.decision_score, reverse=True)
    return TaskResults(task=task, rows=rows, incomplete=incomplete)


def _fmt(value: float | None, pattern: str) -> str:
    return "—" if value is None else format(value, pattern)


def _cells(row: Row) -> list[str]:
    m = row.metrics
    method = {"logprobs": "readout", "verbalized": "verbalized"}
    name = f"{row.run.display_name} (local)" if row.run.local else row.run.display_name
    source = method[row.run.probability_source] if row.run.local else row.run.probability_source
    return [
        name,
        source,
        f"{m.decision_score:.1f} [{m.ci_low:.1f}, {m.ci_high:.1f}]",
        f"{m.accuracy:.1%}",
        _fmt(None if m.ece is None else m.ece * 100, ".1f"),
        f"{m.valid_rate:.1%}",
        _fmt(m.p50_ms, ".0f"),
        _fmt(m.p95_ms, ".0f"),
        _fmt(m.decisions_per_second, ".2f"),
    ]


def _headers() -> list[str]:
    return ["System", "Probabilities", "Decision Score [95% CI]", "Accuracy", "ECE", "Valid", "p50 ms", "p95 ms", "Decisions/s"]


def render(results: list[TaskResults], console: Console) -> None:
    """Print one table per task."""
    for result in results:
        table = Table(
            title=f"{result.task.id} · {result.task.primitive} · {len(result.task.items)} items × 5 repeats", title_justify="left"
        )
        for index, header in enumerate(_headers()):
            table.add_column(header, justify="left" if index < 2 else "right", no_wrap=True)
        for row in result.rows:
            table.add_row(*_cells(row), style="bold green" if row.run.local else None)
        console.print(table)
        console.print("  ECE = calibration gap in points. Latency and decisions/s of local rows: one request at a time on this machine.")
        for line in result.incomplete:
            console.print(f"  [yellow]not scored yet:[/yellow] {line}")
        console.print()


def write_markdown(config: Config, results: list[TaskResults], path: Path) -> None:
    """Write the tables and notes as Markdown."""
    lines = [
        "# openjev benchmark: local System One agents vs Jev",
        "",
        f"Benchmark data: Jevals (jevals.com), release {config.benchmark.release}, suite {config.benchmark.suite}. CC-BY-4.0.",
        "Item text: Banking77 (CC BY 4.0, PolyAI), HelpSteer2 (CC BY 4.0, NVIDIA), PubMedQA (MIT).",
        "",
        "Local rows (models served by LM Studio) are scored with the same code as the published rows;",
        "the published rows are recomputed from their run logs and checked against the published board.",
        "Latency and decisions/s for published rows were measured by jevals over the internet at concurrency 4;",
        "local rows are measured one request at a time on this machine. Decisions/s = decisions / summed decision time.",
        "",
    ]
    for result in results:
        lines += [
            f"## {result.task.id} ({result.task.primitive})",
            "",
            "| " + " | ".join(_headers()) + " |",
            "|" + "---|" * len(_headers()),
        ]
        lines += ["| " + " | ".join(_cells(row)) + " |" for row in result.rows]
        lines += [f"- not scored yet: {line}" for line in result.incomplete]
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
