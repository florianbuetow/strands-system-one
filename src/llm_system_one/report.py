"""Score every run, check the published runs against the board, and render the results tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from llm_system_one.benchmark import run_path
from llm_system_one.config import Config
from llm_system_one.jevals import Task, load_board, load_task, reference_run_path
from llm_system_one.scoring import Metrics, Run, check_against_board, metrics, prior_run, read_run, resamples


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
        "# llm-system-one benchmark: local System One agents vs Jev",
        "",
        f"Benchmark data: Jevals (jevals.com), release {config.benchmark.release}, suite {config.benchmark.suite}. CC-BY-4.0. "
        "Item text: Banking77 (CC BY 4.0, PolyAI), HelpSteer2 (CC BY 4.0, NVIDIA), PubMedQA (MIT).",
        "",
        "## What we ran and where the results came from",
        "",
        "**We did not call Jev or run the six published reference LLMs ourselves.** Their rows use answer logs published by Jevals, "
        "rescored with our code. Only the rows marked **(local)** come from models we ran ourselves through LM Studio: "
        "Qwen3 0.6B, MiniCPM5 2B and Qwen3.5 4B, each with readout and verbalized probabilities. "
        "The Label prior row is a calculated baseline, not a model run.",
        "",
        f"`just fetch` downloads the required files from `{config.benchmark.jevals_repository}`, pinned to commit "
        f"`{config.benchmark.jevals_commit}`, into `{config.benchmark.input_dir}/jevals/{config.benchmark.jevals_commit}/`. "
        f"These include Jev's answer logs (`runs/jev__<task>__{config.benchmark.suite}.jsonl`), the other reference logs, "
        f"and the published board (`releases/{config.benchmark.release}/board.json`). "
        "The logs contain the published probabilities and timings for each item and repeat.",
        "",
        "## How we scored and checked the results",
        "",
        "`read_run` and `metrics` in `src/llm_system_one/scoring.py` score both the published logs "
        "and our local runs using the Jevals formulas. "
        "`check_against_board` checks the recomputed reference Decision Score, accuracy, loss, ECE, repeat/order flip rates, "
        "and p50/p95 latency against the published board. Report generation stops if a checked value differs beyond rounding tolerance. "
        "All reference rows included below passed these checks. This verifies our rescoring of the published logs; "
        "it does not independently reproduce the reference systems' inference runs.",
        "",
        "Two reported quantities are our own calculations rather than values copied from the board:",
        "",
        "- **95% confidence intervals:** we calculated these using an item-cluster bootstrap "
        f"with {config.benchmark.bootstrap_resamples:,} resamples and random seed `{config.benchmark.bootstrap_seed}`. "
        "They can differ from the published intervals and aren't board-checked.",
        (
            "- **Decisions/s:** we computed this as the number of decisions divided by the summed decision time in seconds, "
            "pulled directly from the run logs. It isn't a published board metric, nor does it represent wall-clock "
            "throughput under concurrent load."
        ),
        "",
        "## Timing and interpretation limits",
        "",
        (
            "Reference latency comes straight from the published logs, where Jevals measured those requests over the "
            "internet at concurrency 4. Local latency comes from our own runs, processing one request at a time on this "
            "machine. We didn't remeasure Jev's latency. The scoring code is shared, but the execution conditions aren't, "
            "so these tables aren't a controlled speed comparison. We also didn't measure cost or energy use. "
            "Point-estimate rankings on their own don't establish statistically significant differences."
        ),
        "",
        "## Findings",
        "",
        (
            "This interpretation covers the September 18, 2026 release and the local runs shown below. Jev pairs strong "
            "decision quality with low reported latency. It scores higher on the Decision Score than every local model "
            "across all three tasks, though it doesn't lead every comparison: Gemini 3.8 Flash actually scores higher on "
            "PubMedQA and Banking77. These comparisons rely on Jevals' published answers that we rescored, rather than "
            "fresh tests of those services."
        ),
        "",
        (
            "Each task asks a system to pick an answer and assign probabilities to the possible options. PubMedQA tests "
            "medical yes/no decisions, Banking77 tests banking intent classification, and HelpSteer2 tests ratings of "
            "response helpfulness. Jev's published probabilities are labelled native. The six reference LLMs verbalize "
            "probabilities in their output. For our local models, we compare that approach with readout, which extracts "
            "probabilities from next-token log probabilities. The tables compare complete model-and-method combinations; "
            "they don't isolate the effect of model size or architecture."
        ),
        "",
        (
            "Accuracy shows how often the selected answer is correct. Decision Score evaluates the probability distribution "
            "against a baseline that just uses label frequencies without reading the question. Zero means matching that "
            "baseline's loss, and a negative score means doing worse. ECE measures the gap between confidence and observed "
            "correctness, where lower values indicate better calibration under that metric. Validity only tells us whether "
            "the output met the answer format. A system can produce valid answers, or even higher accuracy, while assigning "
            "less useful probabilities."
        ),
        "",
        (
            "On PubMedQA, Gemini leads with a Decision Score of 73.0 and 92.5% accuracy, compared with Jev's 69.0 and "
            "91.3%. On Banking77, Gemini again leads at 74.1 and 84.6%, versus Jev's 67.8 and 79.7%. GLM-5.3 sits close to "
            "Jev on Banking77, with a score of 66.8. Qwen3.8 Flash, DeepSeek V4.1 Flash, Mistral Medium 3.5, and Mercury "
            "2.5 all show lower score point estimates than Jev on these two tasks. Jev is competitive, but these results "
            "don't make it the quality leader on every task. The intervals for Jev and Gemini overlap, so we'd need a "
            "paired analysis of their score differences to assess statistical significance. Overlap alone establishes "
            "neither a difference nor equivalence."
        ),
        "",
        (
            "Jev also doesn't have the lowest calibration error. On Banking77, for example, its ECE is 9.8, compared with "
            "2.8 for DeepSeek and 3.2 for Gemini. DeepSeek still carries a lower Decision Score than Jev. Calibration error "
            "gives us one useful view of the probabilities, but it doesn't capture everything the decision metric rewards."
        ),
        "",
        (
            "HelpSteer2 is the difficult case for everyone. Jev has the highest score point estimate at 9.2, followed by "
            "GLM at 7.8 and Gemini at 4.6. All three 95% intervals include zero. None of the systems in this table has an "
            "interval entirely above the label-prior baseline on this task, so the apparent ranking shouldn't be read as a "
            "clear demonstration that the leading system adds value over that baseline."
        ),
        "",
        (
            "Qwen3.5 4B with readout is the strongest local combination by Decision Score on all three tasks. It reaches "
            "45.3 on PubMedQA and 51.9 on Banking77, compared with Jev's 69.0 and 67.8. Its corresponding accuracies are "
            "82.1% and 69.9%, versus Jev's 91.3% and 79.7%. It serves as useful evidence that a small local model can beat "
            "the label-prior baseline on these tasks, while still leaving a substantial gap to Jev's published results."
        ),
        "",
        (
            "On HelpSteer2, Qwen 4B readout actually has the highest accuracy in the table, at 44.1%, ahead of Jev's 41.3%. Yet "
            "its Decision Score is -11.4, compared with Jev's 9.2. This is the clearest example of why choosing a system on "
            "accuracy alone can be misleading when downstream decisions depend on its probabilities."
        ),
        "",
        (
            "The smaller models are less convincing. Qwen3 0.6B has negative Decision Scores on every task with both "
            "methods. MiniCPM5 2B has positive scores on Banking77, but negative point estimates on PubMedQA and "
            "HelpSteer2. Those results describe these models and configurations; they don't establish a universal minimum "
            "model size for decision tasks."
        ),
        "",
        (
            "For Qwen 4B, readout improves Decision Score over verbalized output on every task: 19.0 to 45.3 on PubMedQA, "
            "-37.4 to -11.4 on HelpSteer2, and 44.9 to 51.9 on Banking77. Readout also gives valid outputs for every local "
            "model on every task. That formatting reliability is useful, but it isn't a guarantee of good probabilities."
        ),
        "",
        (
            "MiniCPM5 shows why readout isn't an automatic improvement. On HelpSteer2 it raises accuracy from 31.0% to "
            "41.3%, while lowering Decision Score from -39.3 to -60.7. On Banking77 it also raises accuracy while lowering "
            "the score. The model, task, and extraction method need to be evaluated together."
        ),
        "",
        (
            "In the published logs, Jev has the lowest median and p95 latency among the reference systems on every task. Its "
            "median latency ranges from 438 to 478 ms, whereas Gemini's ranges from 1,636 to 1,887 ms. These logs show an "
            "attractive quality-and-latency balance for Jev, even where Gemini has higher quality point estimates. They describe "
            "the published measurement conditions, not a service guarantee or a speed measurement we independently repeated."
        ),
        "",
        (
            "The local results add another tradeoff. Qwen 4B readout has median latencies of 526 ms on PubMedQA and 370 ms "
            "on HelpSteer2, but 4,185 ms on Banking77. On that last task, verbalized output takes 1,924 ms: readout's "
            "better score comes with more than twice the median latency. The tables alone don't establish the cause of this "
            "slowdown. Local and published timings use different execution conditions, so they can't establish a controlled "
            "speed advantage."
        ),
        "",
        (
            "Running locally gives control over where inference happens and which model is served, at the cost of managing "
            "the hardware and accepting the quality and latency measured for that setup. This benchmark doesn't establish a "
            "cost, energy, or privacy-compliance advantage. For these tasks, Jev's published results set a stronger "
            "decision-quality reference than our local models; Qwen 4B readout is the strongest local candidate we tested. "
            "A deployment decision should still check the actual task's probabilities, error consequences, and latency "
            "requirements."
        ),
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
