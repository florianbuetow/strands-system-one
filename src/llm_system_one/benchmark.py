"""Run the jevals suite through the System One agents, resumably, with live status."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from llm_system_one.agents import Decision, SystemOneAgent
from llm_system_one.config import Config, Method, ModelConfig
from llm_system_one.jevals import Task, load_task
from llm_system_one.ordering import option_order, order_seed
from llm_system_one.probabilities import Listing, normalize, pick
from llm_system_one.questions import Question
from llm_system_one.sources import PreparedItem, prepare_items


def run_path(config: Config, model: ModelConfig, method: Method, task_id: str) -> Path:
    """Local run log path, named like the jevals logs: <system>__<task>__<suite>.jsonl."""
    return config.benchmark.output_dir / "runs" / f"{run_system(model, method)}__{task_id}__{config.benchmark.suite}.jsonl"


def run_system(model: ModelConfig, method: Method) -> str:
    """System id of a local run."""
    return f"{model.key}-{method}"


def build_question(task: Task, prepared: PreparedItem, epoch: int) -> tuple[Question, int]:
    """The question for one repeat; choice options are shuffled exactly as jevals shuffles them."""
    seed = order_seed(epoch)
    options = task.options
    if task.primitive == "choice":
        by_name = {option.name: option for option in task.options}
        options = tuple(by_name[name] for name in option_order(prepared.item.item_id, task.option_names, seed))
    return Question(task.primitive, prepared.state, task.instructions, options), seed


def _output(task: Task, decision: Decision) -> dict[str, Any] | None:
    if decision.stated is None:
        return None
    names = task.option_names
    vector = normalize(Listing(probabilities=decision.stated), names, decision.mode)
    if task.primitive == "noul":
        return {"type": "noul", "noul": vector[names.index("yes")]}
    if task.primitive == "choice":
        index = pick(task.primitive, vector, names, list(decision.stated))
        if index is None:
            raise ValueError("A choice answer always has a pick")
        return {"type": "choice", "choice": names[index], "probabilities": decision.stated}
    return {"type": "score", "score": sum(index * p for index, p in enumerate(vector)), "probabilities": decision.stated}


def _header(config: Config, task: Task, model: ModelConfig, method: Method) -> dict[str, Any]:
    settings = config.readout.model_dump() if method == "readout" else config.verbalized.model_dump()
    return {
        "type": "header",
        "suite": config.benchmark.suite,
        "bench": task.id,
        "system": run_system(model, method),
        "config": {
            "id": run_system(model, method),
            "display_name": model.display_name,
            "interface": "readout" if method == "readout" else "adapter",
            "kind": "local",
            "model": model.model_id,
            "probability_source": "logprobs" if method == "readout" else "verbalized",
            "probability_host": config.provider.base_url,
            "assistant_prefill": model.assistant_prefill,
            "settings": settings,
        },
        "started_at": datetime.now(UTC).isoformat(),
    }


def _done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(line["item_id"], line["epoch"]) for line in lines[1:] if "type" not in line}


def _duration(seconds: float) -> str:
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


@dataclass
class _Session:
    """Pace of everything run in this invocation, for the all-runs ETA."""

    remaining: int
    decisions: int = 0
    seconds: float = 0.0

    def eta(self) -> str:
        if self.decisions == 0:
            return "unknown"
        return _duration(self.remaining * self.seconds / self.decisions)


@dataclass
class _Tally:
    """Running numbers for one run."""

    decisions: int = 0
    correct: int = 0
    valid: int = 0
    seconds: float = 0.0

    def status(self) -> str:
        if self.decisions == 0:
            return "warming up"
        return (
            f"acc {self.correct / self.decisions:6.1%} · valid {self.valid / self.decisions:6.1%} · "
            f"{self.seconds / self.decisions * 1000:6.0f} ms/decision · {self.decisions / self.seconds:5.2f} decisions/s"
        )

    def eta(self, remaining: int) -> str:
        if self.decisions == 0:
            return "unknown"
        return _duration(remaining * self.seconds / self.decisions)


def _log(console: Console, label: str, completed: int, total: int, tally: _Tally, session: _Session) -> None:
    """One plain status line; it appears in terminals and in redirected output alike."""
    console.print(
        f"[{datetime.now().strftime('%H:%M:%S')}] {label} {completed:>5}/{total} {completed / total:6.1%} · "
        f"{tally.status()} · ETA run {tally.eta(total - completed)} · ETA all runs {session.eta()}",
        highlight=False,
        soft_wrap=True,
    )


def _run_one(
    config: Config,
    task: Task,
    items: list[PreparedItem],
    model: ModelConfig,
    method: Method,
    progress: Progress,
    overall: TaskID,
    session: _Session,
) -> None:
    path = run_path(config, model, method, task.id)
    done = _done(path)
    epochs = config.benchmark.epochs
    total = len(items) * epochs
    label = f"{model.display_name:<12} {method:<10} {task.id:<10}"
    bar = progress.add_task(label, total=total, completed=len(done), status="done" if len(done) == total else "warming up")
    if len(done) == total:
        progress.console.print(f"[{datetime.now().strftime('%H:%M:%S')}] {label} already complete, skipped", highlight=False)
        return
    progress.console.print(
        f"[{datetime.now().strftime('%H:%M:%S')}] {label} starting: {total - len(done)} of {total} decisions to go", highlight=False
    )
    agent = SystemOneAgent(config.provider, model, method, config.readout, config.verbalized)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_header(config, task, model, method)) + "\n")
    agent.decide(build_question(task, items[0], 0)[0])  # discarded warm-up call, as in jevals
    tally = _Tally()
    completed = len(done)
    last_log = time.monotonic()
    with path.open("a") as log:
        for prepared in items:
            for epoch in range(epochs):
                if (prepared.item.item_id, epoch) in done:
                    continue
                question, seed = build_question(task, prepared, epoch)
                decision = agent.decide(question)
                output = _output(task, decision)
                log.write(
                    json.dumps(
                        {
                            "item_id": prepared.item.item_id,
                            "epoch": epoch,
                            "target": prepared.item.target,
                            "order_seed": seed,
                            "output": output,
                            "raw": decision.raw,
                            "model": model.model_id,
                            "usage": {"input": decision.input_tokens, "output": decision.output_tokens},
                            "seconds": round(decision.seconds, 4),
                            "first_token_seconds": decision.first_token_seconds,
                            "malformed": decision.malformed,
                            "refusal": False,
                            "retries": decision.retries,
                            "at": datetime.now(UTC).isoformat(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log.flush()
                tally.decisions += 1
                tally.seconds += decision.seconds
                if output is not None:
                    tally.valid += 1
                    vector = normalize(Listing(probabilities=decision.stated or {}), task.option_names, decision.mode)
                    if pick(task.primitive, vector, task.option_names, list(decision.stated or {})) == prepared.item.target:
                        tally.correct += 1
                completed += 1
                session.decisions += 1
                session.seconds += decision.seconds
                session.remaining -= 1
                progress.update(bar, advance=1, status=tally.status())
                progress.update(overall, advance=1)
                if time.monotonic() - last_log >= config.benchmark.status_interval_seconds:
                    _log(progress.console, label, completed, total, tally, session)
                    last_log = time.monotonic()
    progress.update(bar, status=f"done · {tally.status()}")
    _log(progress.console, label, completed, total, tally, session)


def run_benchmark(config: Config, models: list[ModelConfig], task_ids: list[str], methods: list[Method], console: Console) -> None:
    """Run every (task, model, method) combination; finished decisions are skipped."""
    prepared = {
        task_id: (
            load_task(config.benchmark, task_id),
            prepare_items(load_task(config.benchmark, task_id), config.sources[task_id], config.benchmark.input_dir),
        )
        for task_id in task_ids
    }
    total = sum(len(items) * config.benchmark.epochs for _, items in prepared.values()) * len(models) * len(methods)
    done = sum(len(_done(run_path(config, model, method, task_id))) for task_id in task_ids for model in models for method in methods)
    columns = (
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=24),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
    )
    session = _Session(remaining=total - done)
    with Progress(*columns, console=console) as progress:
        overall = progress.add_task(f"{'all runs':<35}", total=total, completed=done, status="")
        progress.console.print(f"Benchmark: {done}/{total} decisions already done, {total - done} to go", highlight=False)
        for task, items in prepared.values():
            for model in models:
                for method in methods:
                    _run_one(config, task, items, model, method, progress, overall, session)
