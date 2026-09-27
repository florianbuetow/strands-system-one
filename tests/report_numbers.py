"""Hand-made report numbers for the findings and HTML report tests."""

from llm_system_one.config import ReportConfig
from llm_system_one.report_data import TASK_TEXT, Kind, LocalModel, ReportData, Score, TaskScores

JEV = "jev"
PRIOR = "label-prior"

BANKING_OPTIONS = [f"intent_{index}" for index in range(77)]
NAMES = {
    JEV: "Jev",
    PRIOR: "Label prior",
    "big-readout": "Big",
    "big-verbalized": "Big",
    "small-readout": "Small",
    "small-verbalized": "Small",
}


def score(
    system: str,
    kind: Kind,
    ds: float,
    method: str = "native",
    acc: float = 50.0,
    lo: float | None = None,
    p50: float = 400.0,
    coverage: float = 2.0,
) -> Score:
    name = NAMES.get(system, system)
    timed = kind != "prior"
    return Score(
        system=system,
        name=name,
        label=f"{name} {method}" if kind == "local" else name,
        kind=kind,
        method=method,
        decision_score=ds,
        ci_low=ds - 5 if lo is None else lo,
        ci_high=ds + 5,
        accuracy=acc,
        ece=5.0 if timed else None,
        valid=100.0,
        p50_ms=p50 if timed else None,
        p95_ms=p50 * 1.5 if timed else None,
        decisions_per_second=2.0 if timed else None,
        options_with_probability=coverage,
    )


def local(
    model: str,
    readout: float,
    verbalized: float,
    acc: float = 50.0,
    lo: float | None = None,
    p50: float = 400.0,
    coverage: float = 2.0,
) -> list[Score]:
    return [
        score(f"{model}-readout", "local", readout, "readout", acc=acc, lo=lo, p50=p50, coverage=coverage),
        score(f"{model}-verbalized", "local", verbalized, "verbalized"),
    ]


def base(jev: float, ref: float = 70.0) -> list[Score]:
    return [score(JEV, "jev", jev), score(PRIOR, "prior", 0.0, "base rates"), score("ref", "reference", ref, "verbalized")]


def task(task_id: str, scores: list[Score]) -> TaskScores:
    options = ["no", "yes"] if task_id == "pubmedqa" else BANKING_OPTIONS
    primitive = "noul" if task_id == "pubmedqa" else "choice"
    ordered = sorted(scores, key=lambda s: -s.decision_score)
    return TaskScores(id=task_id, text=TASK_TEXT[task_id], primitive=primitive, options=options, items=300, scores=ordered)


def report(pubmedqa: list[Score], banking77: list[Score], machines: tuple[str, str] = ("Mac", "Mac")) -> ReportData:
    models = [
        LocalModel(name="Big", build="MLX 8-bit", machine=machines[0], readout="big-readout", verbalized="big-verbalized"),
        LocalModel(name="Small", build="MLX 8-bit", machine=machines[1], readout="small-readout", verbalized="small-verbalized"),
    ]
    tasks = [task("pubmedqa", pubmedqa), task("banking77", banking77)]
    rules = ReportConfig(sparse_readout_share=0.5, valid_rate_floor=90.0, slow_readout_factor=2.0)
    return ReportData(tasks=tasks, models=models, reference=["ref"], incomplete=[], top_logprobs=10, rules=rules)


def plain() -> ReportData:
    """A report where Jev leads everywhere and nothing is unusual."""
    return report(base(60) + local("big", 40, 30) + local("small", 10, 5), base(50) + local("big", 45, 20) + local("small", 15, 5))
