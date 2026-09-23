"""The jevals formulas on small hand-checkable runs."""

import json
from pathlib import Path
from typing import Any

import pytest

from openjev.jevals import SuiteItem, Task
from openjev.questions import Option
from openjev.scoring import check_against_board, loss, metrics, prior_run, prior_vector, read_run, resamples

Answers = dict[tuple[str, int], dict[str, Any] | None]


def noul_task() -> Task:
    items = tuple(SuiteItem(f"i{n}", n, "", target) for n, target in enumerate([1, 1, 0, 1]))
    return Task(
        id="yn",
        primitive="noul",
        dataset="d",
        split="s",
        hf_revision="r",
        instructions="Is it?",
        options=(Option("no", None), Option("yes", None)),
        state_fields=("text",),
        class_counts={"no": 1, "yes": 3},
        items=items,
    )


def score_task() -> Task:
    return Task(
        id="sc",
        primitive="score",
        dataset="d",
        split="s",
        hf_revision="r",
        instructions="How good?",
        options=tuple(Option(str(level), None) for level in range(3)),
        state_fields=("text",),
        class_counts={"0": 1, "1": 1, "2": 1},
        items=(SuiteItem("s0", 0, "", 2),),
    )


def write_run(path: Path, task: Task, answers: Answers, interface: str) -> Path:
    header = {
        "type": "header",
        "bench": task.id,
        "system": "sys",
        "config": {"display_name": "System", "interface": interface, "probability_source": "verbalized"},
    }
    lines = [json.dumps(header)]
    targets = {item.item_id: item.target for item in task.items}
    for (item_id, epoch), output in answers.items():
        line = {
            "item_id": item_id,
            "epoch": epoch,
            "target": targets[item_id],
            "output": output,
            "malformed": output is None,
            "refusal": False,
            "seconds": 0.5 + epoch,
        }
        lines.append(json.dumps(line))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_brier_and_ranked_probability_score() -> None:
    assert loss(noul_task(), [0.2, 0.8], 1) == pytest.approx(0.08)
    # cumulative P = [0.2, 0.5], target level 2 -> Y = [0, 0]: (0.04 + 0.25) / 2
    assert loss(score_task(), [0.2, 0.3, 0.5], 2) == pytest.approx(0.145)


def test_prior_scores_zero() -> None:
    task = noul_task()
    samples = resamples(len(task.items), 200, 1)
    result = metrics(task, prior_run(task, 2), 2, samples)
    assert prior_vector(task) == [0.25, 0.75]
    assert result.decision_score == pytest.approx(0)
    assert result.accuracy == pytest.approx(0.75)
    assert result.p50_ms is None


def test_perfect_run_scores_100_and_malformed_is_uniform(tmp_path: Path) -> None:
    task = noul_task()
    perfect: Answers = {(item.item_id, epoch): {"type": "noul", "noul": float(item.target)} for item in task.items for epoch in range(2)}
    run = read_run(write_run(tmp_path / "p.jsonl", task, perfect, "native"), task, 10, local=True)
    result = metrics(task, run, 2, resamples(4, 200, 1))
    assert result.decision_score == pytest.approx(100)
    assert result.accuracy == 1
    assert result.ece == pytest.approx(0)
    assert result.repeat_flip_rate == 0

    broken: Answers = dict(perfect)
    broken[("i0", 0)] = None
    run = read_run(write_run(tmp_path / "b.jsonl", task, broken, "native"), task, 10, local=True)
    result = metrics(task, run, 2, resamples(4, 200, 1))
    assert result.accuracy == pytest.approx(7 / 8)
    assert result.valid_rate == pytest.approx(7 / 8)
    # item i0 loses (0.5^2 + 0.5^2) / 2 over its two repeats; the prior loses 0.125 per item on average
    prior_loss = sum(loss(task, prior_vector(task), item.target) for item in task.items) / 4
    assert result.decision_score == pytest.approx(100 * (1 - (0.25 / 4) / prior_loss))


def test_repeat_flips_and_latency(tmp_path: Path) -> None:
    task = noul_task()
    answers: Answers = {(item.item_id, epoch): {"type": "noul", "noul": 0.9} for item in task.items for epoch in range(2)}
    answers[("i2", 1)] = {"type": "noul", "noul": 0.1}
    run = read_run(write_run(tmp_path / "r.jsonl", task, answers, "native"), task, 10, local=True)
    result = metrics(task, run, 2, resamples(4, 200, 1))
    assert result.repeat_flip_rate == pytest.approx(0.25)
    assert result.p50_ms == pytest.approx(500)
    assert result.p95_ms == pytest.approx(1500)
    assert result.decisions_per_second == pytest.approx(8 / (4 * 0.5 + 4 * 1.5))


def test_missing_decisions_are_rejected(tmp_path: Path) -> None:
    task = noul_task()
    answers: Answers = {("i0", 0): {"type": "noul", "noul": 0.9}}
    run = read_run(write_run(tmp_path / "m.jsonl", task, answers, "native"), task, 10, local=True)
    with pytest.raises(ValueError, match="1/8 decisions"):
        metrics(task, run, 2, resamples(4, 200, 1))


def test_board_mismatch_is_reported(tmp_path: Path) -> None:
    task = noul_task()
    answers: Answers = {(item.item_id, epoch): {"type": "noul", "noul": 0.9} for item in task.items for epoch in range(2)}
    run = read_run(write_run(tmp_path / "x.jsonl", task, answers, "native"), task, 10, local=False)
    result = metrics(task, run, 2, resamples(4, 200, 1))
    row = {
        "slug": "sys",
        "primitive": "noul",
        "decision_score": result.decision_score,
        "accuracy": result.accuracy,
        "loss": result.loss,
        "ece_10_bin": result.ece,
        "repeat_flip_rate": result.repeat_flip_rate,
        "order_flip_rate": None,
        "p50_ms": result.p50_ms,
        "p95_ms": result.p95_ms,
    }
    check_against_board(task, run, result, [row])
    with pytest.raises(ValueError, match="accuracy"):
        check_against_board(task, run, result, [{**row, "accuracy": 0.1}])


def test_adapter_choice_spreads_unlisted_mass(tmp_path: Path) -> None:
    items = (SuiteItem("c0", 0, "", 0),)
    names = [f"o{i}" for i in range(12)]
    task = Task("ch", "choice", "d", "s", "r", "Which?", tuple(Option(n, None) for n in names), ("text",), dict.fromkeys(names, 1), items)
    answers: Answers = {("c0", 0): {"type": "choice", "choice": "o0", "probabilities": {"o0": 0.8}}}
    run = read_run(write_run(tmp_path / "c.jsonl", task, answers, "adapter"), task, 10, local=False)
    assert run.decisions[0].vector[0] == pytest.approx(0.8)
    assert run.decisions[0].vector[5] == pytest.approx(0.2 / 11)
