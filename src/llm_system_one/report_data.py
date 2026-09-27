"""The numbers the benchmark report shows, collected from the scored runs.

Values are rounded to the precision the report prints, so every sentence that compares two systems agrees with the
tables and charts beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import mean
from typing import Literal

from llm_system_one.benchmark import run_system
from llm_system_one.config import Config, Method, ModelConfig, ReportConfig
from llm_system_one.report import Row, TaskResults
from llm_system_one.scoring import Run

Kind = Literal["jev", "reference", "local", "prior"]
METHODS: tuple[Method, Method] = ("readout", "verbalized")


@dataclass(frozen=True)
class TaskText:
    """How the report names and describes a task."""

    name: str
    phrase: str
    description: str
    unit: str


TASK_TEXT = {
    "pubmedqa": TaskText(
        name="PubMedQA",
        phrase="biomedical yes/no questions",
        description=(
            "Answer a biomedical research question yes or no from passages of a PubMed abstract. The system states the probability of yes."
        ),
        unit="answers",
    ),
    "helpsteer2": TaskText(
        name="HelpSteer2",
        phrase="helpfulness ratings",
        description="Rate how helpful a response is, from 0 (not helpful at all) to 4 (extremely helpful).",
        unit="levels",
    ),
    "banking77": TaskText(
        name="Banking77",
        phrase="banking intents",
        description="Name the intent behind a bank customer’s message, such as card_arrival or cancel_transfer.",
        unit="intents",
    ),
}


@dataclass(frozen=True)
class Score:
    """One system on one task, in report units: percent, ECE points and milliseconds."""

    system: str
    name: str
    label: str
    kind: Kind
    method: str
    decision_score: float
    ci_low: float
    ci_high: float
    accuracy: float
    ece: float | None
    valid: float
    p50_ms: float | None
    p95_ms: float | None
    decisions_per_second: float | None
    options_with_probability: float


@dataclass(frozen=True)
class TaskScores:
    """Every system scored on one task, best Decision Score first."""

    id: str
    text: TaskText
    primitive: str
    options: list[str]
    items: int
    scores: list[Score]

    def score(self, system: str) -> Score:
        """Return one system's score.

        Raises:
            KeyError: If the system was not scored on this task.
        """
        for score in self.scores:
            if score.system == system:
                return score
        raise KeyError(f"{system!r} has no score on {self.id}")

    def of_kind(self, *kinds: Kind) -> list[Score]:
        """Scores of the given kinds, best Decision Score first."""
        return [score for score in self.scores if score.kind in kinds]

    @property
    def jev(self) -> Score:
        """Jev's score."""
        return self.score("jev")

    @property
    def prior(self) -> Score:
        """The label prior's score."""
        return self.score("label-prior")


@dataclass(frozen=True)
class LocalModel:
    """A local model with finished readout and verbalized runs on every task."""

    name: str
    build: str
    machine: str
    readout: str
    verbalized: str


@dataclass(frozen=True)
class ReportData:
    """Everything the report shows."""

    tasks: list[TaskScores]
    models: list[LocalModel]
    reference: list[str]
    incomplete: list[str]
    top_logprobs: int
    rules: ReportConfig

    @property
    def local_systems(self) -> list[str]:
        """Local system ids: models by average Decision Score, readout before verbalized."""
        return [system for model in self.models for system in (model.readout, model.verbalized)]

    def machines(self) -> list[tuple[str, list[LocalModel]]]:
        """Local models grouped by the machine that served them, the machine with the most models first."""
        groups: dict[str, list[LocalModel]] = {}
        for model in self.models:
            groups.setdefault(model.machine, []).append(model)
        return sorted(groups.items(), key=lambda group: -len(group[1]))


def collect(config: Config, results: list[TaskResults]) -> ReportData:
    """Collect the report numbers, keeping only local models that finished every task with both methods.

    Raises:
        ValueError: If the methods are not readout and verbalized, a task lacks Jev or the label prior, or a local
            run matches no configured model.
    """
    if sorted(config.benchmark.methods) != sorted(METHODS):
        raise ValueError(f"The report compares readout with verbalized; benchmark.methods is {config.benchmark.methods}")
    tasks = [_task_scores(config, result) for result in results]
    complete = [model for model in config.models if all(_has_runs(task, model) for task in tasks)]
    kept = {run_system(model, method) for model in complete for method in METHODS}
    tasks = [replace(task, scores=[s for s in task.scores if s.kind != "local" or s.system in kept]) for task in tasks]
    models = sorted((_local_model(model) for model in complete), key=lambda m: -_average(tasks, [m.readout, m.verbalized]))
    reference = sorted((s.system for s in tasks[0].of_kind("reference")), key=lambda system: -_average(tasks, [system]))
    left_out = [f"{model.display_name}: left out until both methods finish every task" for model in config.models if model not in complete]
    incomplete = [f"{result.task.id}: {line}" for result in results for line in result.incomplete] + left_out
    return ReportData(
        tasks=tasks,
        models=models,
        reference=reference,
        incomplete=incomplete,
        top_logprobs=config.readout.top_logprobs,
        rules=config.report,
    )


def _task_scores(config: Config, result: TaskResults) -> TaskScores:
    task = result.task
    if task.id not in TASK_TEXT:
        raise ValueError(f"No report text for task {task.id!r}; add it to TASK_TEXT in report_data.py")
    scores = [_score(config, row) for row in result.rows]
    kinds = {score.kind for score in scores}
    if "jev" not in kinds or "prior" not in kinds:
        raise ValueError(f"The report needs Jev and the label prior on {task.id}; add 'jev' to benchmark.reference_systems")
    return TaskScores(
        id=task.id, text=TASK_TEXT[task.id], primitive=task.primitive, options=task.option_names, items=len(task.items), scores=scores
    )


def _identify(config: Config, run: Run) -> tuple[Kind, str]:
    if run.system == "jev":
        return "jev", run.probability_source
    if run.system == "label-prior":
        return "prior", run.probability_source
    if not run.local:
        return "reference", run.probability_source
    for model in config.models:
        for method in METHODS:
            if run_system(model, method) == run.system:
                return "local", method
    raise ValueError(f"Local run {run.system!r} matches no configured model and method")


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _score(config: Config, row: Row) -> Score:
    run, m = row.run, row.metrics
    kind, method = _identify(config, run)
    if kind in ("jev", "local") and (m.p50_ms is None or m.p95_ms is None):
        raise ValueError(f"{run.display_name} ({method}) has no timed decisions")
    return Score(
        system=run.system,
        name=run.display_name,
        label=f"{run.display_name} {method}" if kind == "local" else run.display_name,
        kind=kind,
        method=method,
        decision_score=round(m.decision_score, 1),
        ci_low=round(m.ci_low, 1),
        ci_high=round(m.ci_high, 1),
        accuracy=round(m.accuracy * 100, 1),
        ece=_rounded(None if m.ece is None else m.ece * 100, 1),
        valid=round(m.valid_rate * 100, 1),
        p50_ms=_rounded(m.p50_ms, 0),
        p95_ms=_rounded(m.p95_ms, 0),
        decisions_per_second=_rounded(m.decisions_per_second, 2),
        options_with_probability=round(mean(sum(1 for p in d.vector if p > 0) for d in run.decisions), 1),
    )


def _has_runs(task: TaskScores, model: ModelConfig) -> bool:
    systems = {score.system for score in task.scores}
    return all(run_system(model, method) in systems for method in METHODS)


def _local_model(model: ModelConfig) -> LocalModel:
    return LocalModel(
        name=model.display_name,
        build=model.build,
        machine=model.machine,
        readout=run_system(model, "readout"),
        verbalized=run_system(model, "verbalized"),
    )


def _average(tasks: list[TaskScores], systems: list[str]) -> float:
    return mean(task.score(system).decision_score for task in tasks for system in systems)
