"""Write the benchmark report as one HTML page: charts drawn from the scored runs, text from the findings rules.

The page layout, styles and chart code live in report_template.html. This module fills its <!--@name--> markers with
escaped text and embeds the report numbers as JSON for the charts.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from llm_system_one import findings
from llm_system_one.config import Config
from llm_system_one.findings import Example, Finding, count, listing, num, pct
from llm_system_one.report import TaskResults
from llm_system_one.report_data import ReportData, Score, TaskScores, collect

TEMPLATE = Path(__file__).with_name("report_template.html")


def write_html(config: Config, results: list[TaskResults], path: Path) -> None:
    """Write the HTML report for the scored runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(config, collect(config, results)))


def render_html(config: Config, data: ReportData) -> str:
    """Fill the page template.

    Raises:
        ValueError: If the template lacks a marker or keeps one that nothing fills.
    """
    page = TEMPLATE.read_text()
    for name, value in _fragments(config, data).items():
        marker = f"<!--@{name}-->"
        if page.count(marker) != 1:
            raise ValueError(f"{TEMPLATE.name} must contain {marker} exactly once")
        page = page.replace(marker, value)
    if "<!--@" in page:
        raise ValueError(f"{TEMPLATE.name} has a marker that nothing fills")
    return page


def _e(text: str) -> str:
    return html.escape(text, quote=True)


def _fragments(config: Config, data: ReportData) -> dict[str, str]:
    accuracy, calibration = findings.accuracy_example(data), findings.calibration_example(data)
    bench = config.benchmark
    return {
        "release": _e(bench.release),
        "model-count": _e(count(len(data.models))),
        "task-count": _e(count(len(data.tasks))),
        "spec": _spec(config, data),
        "incomplete": _incomplete(data),
        "scatter-tasks": _e(_task_phrases(data)),
        "scatter-caption": _e(findings.scatter_caption(data)),
        "scatter-notes": _items(findings.scatter_notes(data)),
        "findings": _findings(findings.summary(data)),
        "quality-intro": _e(findings.quality_intro(data)),
        "quality-notes": _items(findings.quality_notes(data)),
        "method-caption": _e(findings.method_caption(data)),
        "method-notes": _items(findings.method_notes(data)),
        "latency-intro": _e(findings.latency_intro(data)),
        "latency-notes": _items(findings.latency_notes(data)),
        "tasks": "".join(_task_card(task) for task in data.tasks),
        "prior-note": _e(_prior_note(data)),
        "ds-track": _ds_track(data),
        "ds-example": _e(_ds_example(data)),
        "interval": _e(
            f"LLM-based-system-one’s item-cluster bootstrap: {bench.bootstrap_resamples:,} resamples of whole items, "
            f"seed {bench.bootstrap_seed}. Not checked against the published board."
        ),
        "accuracy-title": _e(accuracy.title),
        "accuracy-caption": _e(accuracy.caption),
        "calibration-title": _e(calibration.title),
        "calibration-caption": _e(calibration.caption),
        "speed-limit": _e(findings.speed_limit(data)),
        "colophon": _colophon(config),
        "data": _json(_payload(data, accuracy, calibration)),
    }


def _items(notes: list[str]) -> str:
    return "".join(f"<li>{_e(note)}</li>" for note in notes)


def _findings(found: list[Finding]) -> str:
    return "".join(f"<li><b>{_e(finding.title)}</b><span>{_e(finding.body)}</span></li>" for finding in found)


def _spec(config: Config, data: ReportData) -> str:
    bench = config.benchmark
    sizes = sorted({task.items for task in data.tasks})
    tasks = f"{len(data.tasks)} × {sizes[0]} items" if len(sizes) == 1 else f"{len(data.tasks)} tasks"
    cells = [
        ("Suite", f"Jevals {bench.suite}"),
        ("Release", bench.release),
        ("Tasks", tasks),
        ("Intervals", f"95%, {bench.bootstrap_resamples:,} bootstrap resamples"),
    ]
    return "".join(f"<div><dt>{_e(key)}</dt><dd>{_e(value)}</dd></div>" for key, value in cells) + _systems(data)


def _systems(data: ReportData) -> str:
    builds: dict[tuple[str, str], list[str]] = {}
    for model in data.models:
        builds.setdefault((model.build, model.machine), []).append(model.name)
    local = "".join(
        f'<p>{_e(", ".join(names))}<span class="sg-sub">{_e(build)} · {_e(machine)}</span></p>'
        for (build, machine), names in builds.items()
    )
    return (
        '<div class="systems"><dt>Systems</dt><dd>'
        '<div class="sysgrp"><p class="sg-h"><i class="mk jev"></i>System One model · published logs</p><p>Jev (TypeSafe)</p></div>'
        f'<div class="sysgrp"><p class="sg-h"><i class="mk loc"></i>Local models · LM Studio</p>{local}</div>'
        "</dd></div>"
    )


def _incomplete(data: ReportData) -> str:
    if not data.incomplete:
        return ""
    return f'<p class="small">Not in this report yet: {_e("; ".join(data.incomplete))}.</p>'


def _task_phrases(data: ReportData) -> str:
    return f"{count(len(data.tasks))} tasks: {listing([f'{task.text.phrase} ({task.text.name})' for task in data.tasks])}"


def _task_card(task: TaskScores) -> str:
    facts = [f"{len(task.options)} {task.text.unit}", f"Loss: {_loss(task.primitive)}", f"Label prior accuracy: {pct(task.prior.accuracy)}"]
    return (
        f'<article class="task"><h3>{_e(task.text.name)}</h3><p class="type">{_e(_answer_kind(task.primitive))}</p>'
        f'<p>{_e(task.text.description)}</p><ul class="facts">{"".join(f"<li>{_e(fact)}</li>" for fact in facts)}</ul></article>'
    )


def _prior_note(data: ReportData) -> str:
    text = "Label prior accuracy is what always picking the most common answer achieves."
    task = min(data.tasks, key=lambda t: t.prior.accuracy)
    if task.prior.accuracy <= 2 * 100 / len(task.options):
        text += f" On {task.text.name} the {task.text.unit} are spread so evenly that this reaches only {pct(task.prior.accuracy)}."
    return text


def _position(score: Score) -> str:
    return f"{min(100.0, max(0.0, (score.decision_score + 100) / 2)):.1f}%"


def _ds_marks(data: ReportData) -> tuple[TaskScores, Score, Score]:
    task = data.tasks[0]
    return task, task.jev, min(task.of_kind("local"), key=lambda score: score.decision_score)


def _ds_track(data: ReportData) -> str:
    _, jev, low = _ds_marks(data)
    hollow = " hollow" if low.method == "verbalized" else ""
    marks = [(low, f"m loc{hollow}"), (jev, "m jev")]
    values = "".join(f'<span class="v" style="left:{_position(s)}">{_e(num(s.decision_score))}</span>' for s, _ in marks)
    dots = "".join(f'<i class="{cls}" style="left:{_position(s)}"></i>' for s, cls in marks)
    return f'<i class="ds-zero"></i>{values}{dots}'


def _ds_example(data: ReportData) -> str:
    task, jev, low = _ds_marks(data)
    look = "open orange" if low.method == "verbalized" else "orange"
    return f"Examples from {task.text.name}: Jev at {num(jev.decision_score)} (blue) and {low.label} at {num(low.decision_score)} ({look})."


def _colophon(config: Config) -> str:
    bench = config.benchmark
    repo = f"https://github.com/{bench.jevals_repository}"
    commit = f"{repo}/tree/{bench.jevals_commit}"
    return (
        "<p>Generated by <code>just report</code> together with <code>reports/benchmark/report.md</code>.</p>"
        f'<p>Benchmark data: Jevals (<a href="https://jevals.com" target="_blank" rel="noopener">jevals.com</a>), release '
        f"{_e(bench.release)}, suite {_e(bench.suite)}, CC-BY-4.0. Item text: Banking77 (CC BY 4.0, PolyAI), HelpSteer2 "
        f'(CC BY 4.0, NVIDIA), PubMedQA (MIT). Answer logs and board from <a href="{_e(repo)}" target="_blank" rel="noopener">'
        f'<code>{_e(bench.jevals_repository)}</code></a> at commit <a href="{_e(commit)}" target="_blank" rel="noopener">'
        f"<code>{_e(bench.jevals_commit)}</code></a>.</p>"
    )


def _chart_type(task: TaskScores) -> str:
    if task.primitive == "score":
        return f"score, {task.options[0]}–{task.options[-1]}"
    if task.primitive == "choice":
        return f"choice of {len(task.options)}"
    return _answer_kind(task.primitive)


def _numbers(score: Score) -> dict[str, float | None]:
    return {
        "ds": score.decision_score,
        "lo": score.ci_low,
        "hi": score.ci_high,
        "acc": score.accuracy,
        "ece": score.ece,
        "valid": score.valid,
        "p50": score.p50_ms,
        "p95": score.p95_ms,
        "dps": score.decisions_per_second,
    }


def _payload(data: ReportData, accuracy: Example, calibration: Example) -> dict[str, object]:
    systems = {
        s.system: {"name": s.name, "method": s.method, "kind": _css_kind(s.kind), "hollow": s.kind == "local" and s.method == "verbalized"}
        for s in data.tasks[0].scores
    }
    return {
        "tasks": [{"id": task.id, "name": task.text.name, "type": _chart_type(task)} for task in data.tasks],
        "systems": systems,
        "results": {task.id: {s.system: _numbers(s) for s in task.scores} for task in data.tasks},
        "local": data.local_systems,
        "reference": data.reference,
        "models": [{"name": m.name, "readout": m.readout, "verbalized": m.verbalized} for m in data.models],
        "machines": [
            {"name": machine, "ids": [system for model in models for system in (model.readout, model.verbalized)]}
            for machine, models in data.machines()
        ],
        "table": {task.id: [s.system for s in task.scores if s.kind != "reference"] for task in data.tasks},
        "examples": {
            "accuracy": {"task": accuracy.task, "ids": accuracy.systems},
            "calibration": {"task": calibration.task, "ids": calibration.systems},
        },
    }


def _json(payload: dict[str, object]) -> str:
    # "<" is escaped so no string in the data can close the <script> element that holds it.
    return json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")


def _css_kind(kind: str) -> str:
    return {"jev": "jev", "reference": "ref", "local": "loc", "prior": "prior"}[kind]


def _loss(primitive: str) -> str:
    return {"noul": "Brier score", "choice": "Brier score", "score": "ranked probability score"}[primitive]


def _answer_kind(primitive: str) -> str:
    return {"noul": "yes / no", "choice": "choice", "score": "score"}[primitive]
