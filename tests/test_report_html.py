"""The HTML report: collecting the scored runs and filling the page template."""

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from llm_system_one.config import Config, load_config
from llm_system_one.jevals import Task
from llm_system_one.questions import Option
from llm_system_one.report import Row, TaskResults
from llm_system_one.report_data import collect
from llm_system_one.report_html import render_html
from llm_system_one.scoring import Metrics, Run, Scored
from tests.report_numbers import JEV, PRIOR, plain

CONFIG = Path("config/llm-system-one.toml")


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config(CONFIG)


def embedded(page: str) -> dict[str, object]:
    match = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', page, flags=re.S)
    assert match is not None
    data: dict[str, object] = json.loads(match.group(1))
    return data


def test_every_marker_is_filled_and_the_numbers_are_embedded(config: Config) -> None:
    page = render_html(config, plain())
    assert "<!--@" not in page
    data = embedded(page)
    assert data["local"] == ["big-readout", "big-verbalized", "small-readout", "small-verbalized"]
    assert data["table"] == {
        "pubmedqa": [JEV, "big-readout", "big-verbalized", "small-readout", "small-verbalized", PRIOR],
        "banking77": [JEV, "big-readout", "big-verbalized", "small-readout", "small-verbalized", PRIOR],
    }
    assert "Jev beats every local model on every task." in page


def test_text_from_the_numbers_cannot_break_the_page(config: Config) -> None:
    data = plain()
    hostile = "Big</script><b>x"
    tasks = [
        replace(task, scores=[replace(s, name=hostile, label=hostile) if s.system == "big-readout" else s for s in task.scores])
        for task in data.tasks
    ]
    page = render_html(config, replace(data, tasks=tasks))
    assert "</script><b>x" not in page
    assert "Big&lt;/script&gt;&lt;b&gt;x" in page
    systems = embedded(page)["systems"]
    assert isinstance(systems, dict)
    assert systems["big-readout"]["name"] == "Big</script><b>x"


def row(system: str, name: str, source: str, local: bool, ds: float) -> Row:
    decision = Scored(item_id="a", epoch=0, target=0, vector=[0.7, 0.3], pick=0, answered=True, seconds=0.4)
    run = Run(system=system, display_name=name, probability_source=source, local=local, decisions=[decision])
    metrics = Metrics(
        decision_score=ds,
        ci_low=ds - 5,
        ci_high=ds + 5,
        accuracy=0.5,
        ece=0.05,
        loss=0.2,
        repeat_flip_rate=None,
        order_flip_rate=None,
        p50_ms=400.0,
        p95_ms=600.0,
        valid_rate=1.0,
        decisions=1,
        decisions_per_second=2.0,
    )
    return Row(run, metrics)


def pubmedqa() -> Task:
    return Task(
        id="pubmedqa",
        primitive="noul",
        dataset="d",
        split="s",
        hf_revision="r",
        instructions="i",
        options=(Option("no", None), Option("yes", None)),
        state_fields=(),
        class_counts={"no": 1, "yes": 1},
        items=(),
    )


def test_models_without_every_run_are_left_out_and_listed(config: Config) -> None:
    task = pubmedqa()
    rows = [
        row(JEV, "Jev", "native", False, 60),
        row(PRIOR, "Label prior", "base rates", False, 0),
        row("qwen3.5-4b-readout", "Qwen3.5 4B", "logprobs", True, 40),
        row("qwen3.5-4b-verbalized", "Qwen3.5 4B", "verbalized", True, 20),
        row("qwen3-0.6b-readout", "Qwen3 0.6B", "logprobs", True, 10),
    ]
    data = collect(config, [TaskResults(task=task, rows=rows, incomplete=["Qwen3 0.6B (verbalized): not started"])])
    assert [model.name for model in data.models] == ["Qwen3.5 4B"]
    assert [score.system for score in data.tasks[0].of_kind("local")] == ["qwen3.5-4b-readout", "qwen3.5-4b-verbalized"]
    assert "Qwen3 0.6B: left out until both methods finish every task" in data.incomplete
    assert "pubmedqa: Qwen3 0.6B (verbalized): not started" in data.incomplete


def test_a_report_without_jev_fails(config: Config) -> None:
    with pytest.raises(ValueError, match="needs Jev and the label prior"):
        collect(config, [TaskResults(task=pubmedqa(), rows=[row(PRIOR, "Label prior", "base rates", False, 0)], incomplete=[])])
