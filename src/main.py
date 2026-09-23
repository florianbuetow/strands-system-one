"""openjev: Strands System One agents on local models, benchmarked against Jev on the jevals suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.table import Table

from openjev.agents import SystemOneAgent
from openjev.benchmark import run_benchmark
from openjev.config import Config, Method, ModelConfig, load_config
from openjev.jevals import download_jevals, load_task
from openjev.questions import Option, Question
from openjev.report import render, score_task, write_markdown
from openjev.sources import prepare_items


def _models(config: Config, keys: list[str] | None) -> list[ModelConfig]:
    return config.models if keys is None else [config.model(key) for key in keys]


def _fetch(config: Config, console: Console) -> None:
    download_jevals(config.benchmark)
    table = Table(title="jevals suite (items checked against their source rows)", title_justify="left")
    for column in ("Task", "Primitive", "Options", "Items", "Labels match", "State hash matches jevals"):
        table.add_column(column)
    for task_id in config.benchmark.tasks:
        task = load_task(config.benchmark, task_id)
        items = prepare_items(task, config.sources[task_id], config.benchmark.input_dir)
        matches = sum(1 for item in items if item.hash_matches)
        table.add_row(
            task.id, task.primitive, str(len(task.options)), str(len(items)), f"{len(items)}/{len(items)}", f"{matches}/{len(items)}"
        )
    console.print(table)


def _report(config: Config, console: Console) -> None:
    results = [score_task(config, task_id) for task_id in config.benchmark.tasks]
    render(results, console)
    path = config.benchmark.report_dir / "report.md"
    write_markdown(config, results, path)
    console.print(f"Report written to {path}")


def _question(path: Path) -> Question:
    data: dict[str, Any] = json.loads(path.read_text())
    options = cast(dict[str, str | None], data["options"])
    return Question(
        primitive=data["primitive"],
        state=json.dumps(data["state"], ensure_ascii=False),
        instructions=data["instructions"],
        options=tuple(Option(name, description) for name, description in options.items()),
    )


def _decide(config: Config, question_path: Path, models: list[ModelConfig], console: Console) -> None:
    question = _question(question_path)
    table = Table(title=f"{question.primitive}: {question.instructions}", title_justify="left")
    for column in ("Model", "Method", "Pick", "Probabilities", "Time ms", "First output ms"):
        table.add_column(column)
    methods: list[Method] = ["readout", "verbalized"]
    for model in models:
        for method in methods:
            agent = SystemOneAgent(config.provider, model, method, config.readout, config.verbalized)
            answer = agent.answer(question)
            top = sorted(answer.probabilities.items(), key=lambda pair: pair[1], reverse=True)[:3]
            first = answer.decision.first_token_seconds
            table.add_row(
                model.display_name,
                method,
                str(answer.pick),
                ", ".join(f"{name} {p:.2f}" for name, p in top),
                f"{answer.decision.seconds * 1000:.0f}",
                "—" if first is None else f"{first * 1000:.0f}",
            )
    console.print(table)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the TOML config")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="Download the pinned jevals suite and source datasets, and check every item")
    bench = commands.add_parser("benchmark", help="Run the suite through the agents (resumes where it stopped)")
    bench.add_argument("--model", action="append", help="Model key to run (repeatable); all configured models when omitted")
    commands.add_parser("report", help="Score all runs and print the results tables")
    decide = commands.add_parser("decide", help="Answer one question with every model and both methods")
    decide.add_argument("--question", type=Path, required=True, help="JSON file with primitive, state, instructions, options")
    decide.add_argument("--model", action="append", help="Model key (repeatable); all configured models when omitted")
    return parser


def main(argv: list[str]) -> int:
    """Run the command line interface."""
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    console = Console()
    match args.command:
        case "fetch":
            _fetch(config, console)
        case "benchmark":
            run_benchmark(config, _models(config, args.model), config.benchmark.tasks, config.benchmark.methods, console)
        case "report":
            _report(config, console)
        case "decide":
            _decide(config, args.question, _models(config, args.model), console)
        case _:
            raise ValueError(f"Unknown command {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
