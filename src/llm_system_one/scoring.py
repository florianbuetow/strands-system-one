"""Score run logs with the jevals formulas (https://jevals.com/methodology/).

The same code scores the published jevals runs and the local runs, and the published runs are
checked against the published board, so a scoring mistake cannot go unnoticed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_system_one.jevals import Task
from llm_system_one.ordering import mulberry32
from llm_system_one.probabilities import Listing, ListingMode, normalize, pick


@dataclass(frozen=True)
class Scored:
    """One scored decision."""

    item_id: str
    epoch: int
    target: int
    vector: list[float]
    pick: int | None
    answered: bool
    seconds: float


@dataclass(frozen=True)
class Run:
    """A run log: its header and its scored decisions."""

    system: str
    display_name: str
    probability_source: str
    local: bool
    decisions: list[Scored]


@dataclass(frozen=True)
class Metrics:
    """Board metrics for one run on one task."""

    decision_score: float
    ci_low: float
    ci_high: float
    accuracy: float
    ece: float | None
    loss: float
    repeat_flip_rate: float | None
    order_flip_rate: float | None
    p50_ms: float | None
    p95_ms: float | None
    valid_rate: float
    decisions: int
    decisions_per_second: float | None


def _mode(interface: str, option_count: int, full_listing_max_options: int) -> ListingMode:
    return "top" if interface == "adapter" and option_count > full_listing_max_options else "full"


def _vector(output: dict[str, Any], task: Task, mode: ListingMode) -> tuple[list[float], list[str]]:
    names = task.option_names
    if output["type"] == "noul":
        p_yes = float(output["noul"])
        return [p_yes if name == "yes" else 1 - p_yes for name in names], []
    stated = {str(name): float(value) for name, value in output["probabilities"].items()}
    listed_first = [str(output["choice"])] if output["type"] == "choice" else []
    return normalize(Listing(probabilities=stated), names, mode), listed_first + list(stated)


def read_run(path: Path, task: Task, full_listing_max_options: int, local: bool) -> Run:
    """Read and score a run log in the jevals format.

    Raises:
        ValueError: If the log does not start with a header or covers another task.
    """
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    header = lines[0]
    if header["type"] != "header" or header["bench"] != task.id:
        raise ValueError(f"{path} is not a run log for task {task.id}")
    config = header["config"]
    mode = _mode(config["interface"], len(task.options), full_listing_max_options)
    uniform = [1 / len(task.options)] * len(task.options)
    decisions: list[Scored] = []
    for line in lines[1:]:
        if "type" in line:
            continue  # a resumed run repeats its header
        answered = not line["malformed"] and not line["refusal"] and line["output"] is not None
        if answered:
            vector, listed_first = _vector(line["output"], task, mode)
            chosen = pick(task.primitive, vector, task.option_names, listed_first)
        else:
            vector, chosen = uniform, None
        decisions.append(Scored(line["item_id"], line["epoch"], line["target"], vector, chosen, answered, float(line["seconds"])))
    return Run(
        system=header["system"],
        display_name=config["display_name"],
        probability_source=config["probability_source"],
        local=local,
        decisions=decisions,
    )


def loss(task: Task, vector: list[float], target: int) -> float:
    """Multiclass Brier score (noul, choice) or ranked probability score (score)."""
    if task.primitive == "score":
        cumulative = 0.0
        total = 0.0
        for index in range(len(vector) - 1):
            cumulative += vector[index]
            total += (cumulative - (1.0 if target <= index else 0.0)) ** 2
        return total / (len(vector) - 1)
    return sum((p - (1.0 if index == target else 0.0)) ** 2 for index, p in enumerate(vector))


def prior_vector(task: Task) -> list[float]:
    """The label prior: base rates of the evaluated items."""
    total = sum(task.class_counts.values())
    return [task.class_counts[name] / total for name in task.option_names]


def prior_run(task: Task, epochs: int) -> Run:
    """The label-prior baseline as a run with instant answers."""
    vector = prior_vector(task)
    chosen = pick(task.primitive, vector, task.option_names, [])
    decisions = [Scored(item.item_id, epoch, item.target, vector, chosen, True, 0.0) for item in task.items for epoch in range(epochs)]
    return Run(system="label-prior", display_name="Label prior", probability_source="base rates", local=False, decisions=decisions)


def resamples(item_count: int, count: int, seed: int) -> list[list[int]]:
    """Item-cluster bootstrap resamples, shared by every row of a task (paired comparisons)."""
    random = mulberry32(seed)
    return [[int(random() * item_count) for _ in range(item_count)] for _ in range(count)]


def _nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _ece(decisions: list[Scored]) -> float | None:
    answered = [decision for decision in decisions if decision.answered]
    if not answered:
        return None
    bins: dict[int, list[tuple[float, float]]] = {}
    for decision in answered:
        confidence = max(decision.vector) if decision.pick is None else decision.vector[decision.pick]
        correct = 1.0 if decision.pick == decision.target else 0.0
        bins.setdefault(min(9, math.floor(math.floor(100 * confidence + 0.5) / 10)), []).append((confidence, correct))
    return sum(
        len(members) / len(answered) * abs(sum(c for _, c in members) / len(members) - sum(p for p, _ in members) / len(members))
        for members in bins.values()
    )


def _picks(decisions: list[Scored]) -> dict[str, dict[int, Scored]]:
    by_item: dict[str, dict[int, Scored]] = {}
    for decision in decisions:
        by_item.setdefault(decision.item_id, {})[decision.epoch] = decision
    return by_item


def _flip_rate(decisions: list[Scored], epochs: list[int]) -> float | None:
    eligible = [
        [by_epoch[epoch].pick for epoch in epochs]
        for by_epoch in _picks(decisions).values()
        if all(epoch in by_epoch and by_epoch[epoch].answered for epoch in epochs)
    ]
    if not eligible:
        return None
    return sum(1 for picks in eligible if len(set(picks)) > 1) / len(eligible)


def _item_losses(task: Task, decisions: list[Scored]) -> list[float]:
    by_item = _picks(decisions)
    return [sum(loss(task, d.vector, d.target) for d in by_item[item.item_id].values()) / len(by_item[item.item_id]) for item in task.items]


def metrics(task: Task, run: Run, epochs: int, samples: list[list[int]]) -> Metrics:
    """Compute board metrics for a complete run.

    Raises:
        ValueError: If any (item, repeat) is missing or duplicated.
    """
    keys = {(d.item_id, d.epoch) for d in run.decisions}
    expected = {(item.item_id, epoch) for item in task.items for epoch in range(epochs)}
    if keys != expected or len(run.decisions) != len(expected):
        raise ValueError(f"{run.system} on {task.id}: {len(keys & expected)}/{len(expected)} decisions present")
    losses = _item_losses(task, run.decisions)
    prior = prior_vector(task)
    prior_losses = [loss(task, prior, item.target) for item in task.items]
    system_loss = sum(losses) / len(losses)
    prior_loss = sum(prior_losses) / len(prior_losses)
    boot = sorted(100 * (1 - sum(losses[i] for i in sample) / sum(prior_losses[i] for i in sample)) for sample in samples)
    busy = sum(d.seconds for d in run.decisions)
    answered_ms = [d.seconds * 1000 for d in run.decisions if d.answered] if busy > 0 else []
    return Metrics(
        decision_score=100 * (1 - system_loss / prior_loss),
        ci_low=boot[math.floor(0.025 * len(boot))],
        ci_high=boot[math.ceil(0.975 * len(boot)) - 1],
        accuracy=sum(1 for d in run.decisions if d.pick == d.target) / len(run.decisions),
        ece=_ece(run.decisions) if run.system != "label-prior" else None,
        loss=system_loss,
        repeat_flip_rate=_flip_rate(run.decisions, [0, 1]) if epochs >= 2 else None,
        order_flip_rate=_flip_rate(run.decisions, [0, 2, 3, 4]) if task.primitive == "choice" and epochs >= 5 else None,
        p50_ms=_nearest_rank(answered_ms, 0.5),
        p95_ms=_nearest_rank(answered_ms, 0.95),
        valid_rate=sum(1 for d in run.decisions if d.answered) / len(run.decisions),
        decisions=len(run.decisions),
        decisions_per_second=len(run.decisions) / busy if busy > 0 else None,
    )


def check_against_board(task: Task, run: Run, result: Metrics, board: list[dict[str, Any]]) -> None:
    """Fail if a published run's recomputed metrics differ from the published board row.

    Raises:
        ValueError: On any mismatch beyond the board's rounding.
    """
    rows = [row for row in board if row["slug"] == run.system and row["primitive"] == task.primitive]
    if len(rows) != 1:
        raise ValueError(f"Expected one board row for {run.system}/{task.primitive}, found {len(rows)}")
    row = rows[0]
    checks: list[tuple[str, float | None, float | None, float]] = [
        ("decision_score", result.decision_score, row["decision_score"], 0.001),
        ("accuracy", result.accuracy, row["accuracy"], 0.0001),
        ("loss", result.loss, row["loss"], 0.0001),
        ("ece_10_bin", result.ece, row["ece_10_bin"], 0.0001),
        ("repeat_flip_rate", result.repeat_flip_rate, row["repeat_flip_rate"], 0.0001),
        ("order_flip_rate", result.order_flip_rate, row["order_flip_rate"], 0.0001),
        ("p50_ms", result.p50_ms, row["p50_ms"], 1),
        ("p95_ms", result.p95_ms, row["p95_ms"], 1),
    ]
    for name, ours, published, tolerance in checks:
        if (ours is None) != (published is None) or (ours is not None and published is not None and abs(ours - published) > tolerance):
            raise ValueError(f"{run.system}/{task.id}: recomputed {name}={ours} but the board says {published}")
