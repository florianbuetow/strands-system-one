"""Rule-based findings: every result-dependent sentence in the HTML report, computed from the numbers.

The rules read the rounded report numbers, so their sentences agree with the tables and charts. A rule that finds
nothing worth saying returns nothing rather than a placeholder sentence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

from llm_system_one.report_data import LocalModel, ReportData, Score, TaskScores

type Pair = tuple[TaskScores, Score, Score]


@dataclass(frozen=True)
class Finding:
    """A short claim and the numbers behind it."""

    title: str
    body: str


@dataclass(frozen=True)
class Example:
    """Systems on one task that show a metric disagreeing with the Decision Score."""

    task: str
    systems: list[str]
    title: str
    caption: str


def _signed(text: str, value: float) -> str:
    return f"−{text}" if value < 0 else text


def num(value: float) -> str:
    """Format a number to one decimal, with a real minus sign."""
    return _signed(f"{abs(value):,.1f}", value)


def whole(value: float) -> str:
    """Format a number without decimals, with thousands separators and a real minus sign."""
    return _signed(f"{abs(value):,.0f}", value)


def pct(value: float) -> str:
    """Format a percentage."""
    return f"{num(value)}%"


def ms(value: float) -> str:
    """Format a time in milliseconds."""
    return f"{whole(value)} ms"


def count(n: int) -> str:
    """Spell out counts below ten."""
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    return words[n] if n < len(words) else str(n)


def listing(items: list[str]) -> str:
    """Join items as 'a, b and c'."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def median(score: Score) -> float:
    """Median time per decision.

    Raises:
        ValueError: If the system has no timed decisions.
    """
    if score.p50_ms is None:
        raise ValueError(f"{score.label} has no median time")
    return score.p50_ms


def _p95(score: Score) -> float:
    if score.p95_ms is None:
        raise ValueError(f"{score.label} has no p95 time")
    return score.p95_ms


def _ece(score: Score) -> float:
    if score.ece is None:
        raise ValueError(f"{score.label} has no calibration error")
    return score.ece


def _ms_range(values: list[float]) -> str:
    low, high = min(values), max(values)
    return ms(low) if low == high else f"{whole(low)}–{whole(high)} ms"


def _interval(score: Score) -> str:
    return f"{num(score.ci_low)} to {num(score.ci_high)}"


def _best_local(task: TaskScores) -> Score:
    return max(task.of_kind("local"), key=lambda score: score.decision_score)


def _contenders(task: TaskScores) -> list[Score]:
    return task.of_kind("jev", "local")


def _pairs(data: ReportData, model: LocalModel) -> list[Pair]:
    return [(task, task.score(model.verbalized), task.score(model.readout)) for task in data.tasks]


def _where(tasks: list[str], total: int) -> str:
    if len(tasks) == total:
        return "on every task"
    if not tasks:
        return "on no task"
    return f"only on {tasks[0]}" if len(tasks) == 1 else f"on {listing(tasks)}"


def _other_machines(data: ReportData) -> str:
    groups = data.machines()
    parts = [f"{listing([model.name for model in models])} ran on the {machine}" for machine, models in groups[1:]]
    return f"{listing(parts)}, so those timings are not comparable with the models on the {groups[0][0]}."


# Summary


def summary(data: ReportData) -> list[Finding]:
    """The headline findings, in reading order."""
    found = [_jev_vs_local(data), _baseline(data), _accuracy_vs_score(data), _readout_vs_verbalized(data), _latency(data)]
    return [finding for finding in found if finding is not None]


def _jev_vs_local(data: ReportData) -> Finding:
    ahead = [task.text.name for task in data.tasks if task.jev.decision_score > _best_local(task).decision_score]
    if len(ahead) == len(data.tasks):
        title = "Jev beats every local model on every task."
    elif ahead:
        title = f"Jev beats every local model on {listing(ahead)}."
    else:
        title = "A local model matches or beats Jev on every task."
    jev = listing([f"{num(task.jev.decision_score)} on {task.text.name}" for task in data.tasks])
    best = listing([f"{num(_best_local(t).decision_score)} on {t.text.name} ({_best_local(t).label})" for t in data.tasks])
    return Finding(title, f"Jev scores {jev}. The best local results are {best}.")


def _clear(task: TaskScores) -> list[Score]:
    return [score for score in _contenders(task) if score.ci_low > 0]


def _baseline(data: ReportData) -> Finding | None:
    task = min(data.tasks, key=lambda t: len(_clear(t)))
    clear = _clear(task)
    if len(clear) > 2:
        return None
    if clear:
        verb = "beats" if len(clear) == 1 else "beat"
        title = f"Only {listing([score.label for score in clear])} clearly {verb} the baseline on {task.text.name}."
    else:
        title = f"No system clearly beats the baseline on {task.text.name}."
    return Finding(title, _interval_sentence(task, clear))


def _interval_sentence(task: TaskScores, clear: list[Score]) -> str:
    jev = task.jev
    if jev.ci_low > 0:
        where = "lies entirely above zero"
    elif jev.ci_high < 0:
        where = "lies entirely below zero"
    else:
        where = "includes zero"
    text = f"Jev scores {num(jev.decision_score)} there, and its 95% interval ({_interval(jev)}) {where}."
    local = [score for score in clear if score.kind == "local"]
    if local:
        noun, verb = ("interval", "lies") if len(local) == 1 else ("intervals", "lie")
        text += f" The {noun} of {listing([f'{s.label} ({_interval(s)})' for s in local])} {verb} entirely above zero."
    return text


def _accuracy_case(task: TaskScores) -> tuple[Score, Score] | None:
    pool = _contenders(task)
    accurate = max(pool, key=lambda score: score.accuracy)
    best = max(pool, key=lambda score: score.decision_score)
    return None if accurate.decision_score >= best.decision_score else (accurate, best)


def _accuracy_vs_score(data: ReportData) -> Finding | None:
    cases = [(task, case) for task in data.tasks if (case := _accuracy_case(task)) is not None]
    if not cases:
        return None
    task, (accurate, best) = max(cases, key=lambda found: found[1][1].decision_score - found[1][0].decision_score)
    return Finding(
        "Higher accuracy can hide worse probabilities.",
        f"On {task.text.name}, {accurate.label} has the highest accuracy ({pct(accurate.accuracy)}) and a Decision Score of "
        f"{num(accurate.decision_score)}. {best.label} has {pct(best.accuracy)} accuracy and scores {num(best.decision_score)}.",
    )


def _lower_clause(model: LocalModel, lower: list[Pair]) -> str:
    rising = [task.text.name for task, verbalized, readout in lower if readout.accuracy > verbalized.accuracy]
    if rising and len(rising) == len(lower):
        tail = " while accuracy rises"
    elif rising:
        tail = f" while accuracy rises on {listing(rising)}"
    else:
        tail = ""
    return f"{model.name} it lowers the score on {listing([task.text.name for task, _, _ in lower])}{tail}"


def _readout_vs_verbalized(data: ReportData) -> Finding:
    better = [m.name for m in data.models if all(r.decision_score > v.decision_score for _, v, r in _pairs(data, m))]
    lower = [
        _lower_clause(model, found)
        for model in data.models
        if (found := [(t, v, r) for t, v, r in _pairs(data, model) if r.decision_score < v.decision_score])
    ]
    sentences = [f"It raises the score of {listing(better)} on every task."] if better else []
    if lower:
        sentences.append("For " + "; for ".join(lower) + ".")
    if lower:
        title = "Readout is not an automatic improvement."
    elif len(better) == len(data.models):
        title = "Readout beats verbalized for every local model."
    else:
        title = "Readout never scores lower than verbalized."
    return Finding(title, " ".join(sentences) if sentences else "Readout and verbalized give the same scores.")


def _latency(data: ReportData) -> Finding:
    local = [median(score) for task in data.tasks for score in task.of_kind("local")]
    machines = len(data.machines())
    where = f", on {count(machines)} different machines" if machines > 1 else ""
    return Finding(
        "Latency is not directly comparable.",
        f"Jev’s published median is {_ms_range([median(task.jev) for task in data.tasks])} per decision. The local runs range "
        f"from {ms(min(local))} to {ms(max(local))}, timed one request at a time{where}.",
    )


# Fig. 1: speed vs score


def scatter_caption(data: ReportData) -> str:
    """Caption of the speed-vs-score scatter."""
    text = (
        "Up is a higher Decision Score, where 0 is the label prior. Left is a shorter median time per decision, on a log scale. "
        f"Gray points are {count(len(data.reference))} reference LLMs from Jevals’ published logs, shown for context only; "
        "LLM-based-system-one did not run them. Published and local timings come from different conditions (Fig. 4), so read "
        "horizontal gaps between them as indicative only."
    )
    return f"{text} {_other_machines(data)}" if len(data.machines()) > 1 else text


def scatter_notes(data: ReportData) -> list[str]:
    """Notes under the speed-vs-score scatter."""
    wins = [
        f"{score.label} on {task.text.name}"
        for task in data.tasks
        for score in task.of_kind("reference", "local")
        if median(score) < median(task.jev) and score.decision_score > task.jev.decision_score
    ]
    if wins:
        return [f"Faster than Jev and higher-scoring: {listing(wins)}."]
    return [
        "No system lands above and to the left of Jev on any task: none of the systems with a shorter median time than Jev scores higher."
    ]


# Fig. 2: Decision Score per task


def quality_intro(data: ReportData) -> str:
    """Introduction to the Decision Score chart."""
    beaten = [
        f"{score.label} on {task.text.name}"
        for task in data.tasks
        for score in task.of_kind("local")
        if score.decision_score > task.jev.decision_score
    ]
    text = "Right of zero means lower loss than the label prior. The dashed blue line is Jev’s score; "
    if not beaten:
        return text + "every local model lands to the left of it on every task."
    return text + f"{listing(beaten)} {'lands' if len(beaten) == 1 else 'land'} to the right of it."


def quality_notes(data: ReportData) -> list[str]:
    """Notes under the Decision Score chart."""
    notes = [_gaps(data), _clearing(data)]
    zero = [f"{task.text.name} ({_interval(task.jev)})" for task in data.tasks if task.jev.ci_low <= 0 <= task.jev.ci_high]
    if zero:
        notes.append(f"Jev’s own 95% interval includes zero on {listing(zero)}.")
    return notes + _coverage(data)


def _gaps(data: ReportData) -> str:
    ahead: list[str] = []
    behind: list[str] = []
    level: list[str] = []
    for task in data.tasks:
        gap = round(task.jev.decision_score - _best_local(task).decision_score, 1)
        if gap > 0:
            ahead.append(f"{num(gap)} on {task.text.name}")
        elif gap < 0:
            behind.append(f"{num(-gap)} on {task.text.name}")
        else:
            level.append(task.text.name)
    subject = "the best local system"
    parts: list[str] = []
    if ahead:
        parts.append(f"ahead of {subject} by {listing(ahead)}")
        subject = "it"
    if behind:
        parts.append(f"behind {subject} by {listing(behind)}")
        subject = "it"
    if level:
        parts.append(f"level with {subject} on {listing(level)}")
    return f"In points, Jev is {listing(parts)}."


def _clearing(data: ReportData) -> str:
    parts: list[str] = []
    for model in data.models:
        tasks = [
            task.text.name for task in data.tasks if any(task.score(system).ci_low > 0 for system in (model.readout, model.verbalized))
        ]
        where = _where(tasks, len(data.tasks))
        parts.append(f"{model.name} {where}" if parts else f"{model.name} clears it {where}")
    return f"Counting only whole 95% intervals above zero as clearing the label prior, {listing(parts)}."


def _coverage(data: ReportData) -> list[str]:
    notes: list[str] = []
    for task in data.tasks:
        if len(task.options) <= data.top_logprobs:
            continue
        readout = [score for score in task.of_kind("local") if score.method == "readout"]
        floor = data.rules.sparse_readout_share * len(task.options)
        sparse = [score for score in readout if score.options_with_probability < floor]
        if sparse:
            wide = [score for score in readout if score.options_with_probability >= floor]
            notes.append(_coverage_note(task, sparse, wide, data.top_logprobs))
    return notes


def _coverage_note(task: TaskScores, sparse: list[Score], wide: list[Score], top_logprobs: int) -> str:
    verb, these = ("puts", "this score is") if len(sparse) == 1 else ("put", "these scores are")
    compare = ""
    if wide:
        widest = max(wide, key=lambda score: score.options_with_probability)
        compare = f", against {num(widest.options_with_probability)} for {widest.label}"
    return (
        f"{listing([score.label for score in sparse])} {verb} probability on only "
        f"{listing([num(score.options_with_probability) for score in sparse])} of the {len(task.options)} {task.text.unit} on "
        f"{task.text.name} on average{compare}. The readout sees at most the {top_logprobs} most likely tokens and gives every "
        f"other answer probability 0, so {these} not like-for-like with the other readout rows."
    )


# Fig. 3: readout vs verbalized


def method_caption(data: ReportData) -> str:
    """Caption of the readout-vs-verbalized chart."""
    text = (
        "Each line runs from the verbalized score to the readout score. Changes are differences between point estimates, "
        "readout minus verbalized; accuracy changes are in percentage points."
    )
    missed = [
        f"{task.text.name} ({score.label}, {pct(score.valid)} valid)"
        for task in data.tasks
        for model in data.models
        if (score := task.score(model.readout)).valid < 100
    ]
    if not missed:
        return f"{text} Readout returned a valid answer for every model on every task."
    return f"{text} Readout answers that missed the format: {listing(missed)}."


def method_notes(data: ReportData) -> list[str]:
    """Notes under the readout-vs-verbalized chart."""
    notes = [_model_note(data, model) for model in data.models]
    return notes + [
        f"Only {pct(verbalized.valid)} of {model.name}’s verbalized {task.text.name} answers met the format."
        for model in data.models
        for task, verbalized, _ in _pairs(data, model)
        if verbalized.valid < data.rules.valid_rate_floor
    ]


def _step(pair: Pair) -> str:
    task, verbalized, readout = pair
    return f"{num(verbalized.decision_score)} to {num(readout.decision_score)} on {task.text.name}"


def _drop(pair: Pair) -> str:
    task, verbalized, readout = pair
    rise = ""
    if readout.accuracy > verbalized.accuracy:
        rise = f", while accuracy rises from {pct(verbalized.accuracy)} to {pct(readout.accuracy)}"
    return f"{task.text.name} ({num(verbalized.decision_score)} to {num(readout.decision_score)}{rise})"


def _model_note(data: ReportData, model: LocalModel) -> str:
    pairs = _pairs(data, model)
    up = [pair for pair in pairs if pair[2].decision_score > pair[1].decision_score]
    down = [pair for pair in pairs if pair[2].decision_score < pair[1].decision_score]
    if len(up) == len(pairs):
        text = f"{model.name} improves with readout on every task: {listing([_step(pair) for pair in pairs])}."
    elif len(down) == len(pairs):
        text = f"{model.name} scores lower with readout on every task: {listing([_step(pair) for pair in pairs])}."
    elif not down:
        text = f"{model.name} improves with readout on {listing([pair[0].text.name for pair in up])} and ties on the rest."
    elif not up:
        text = f"{model.name} drops with readout on {listing([_drop(pair) for pair in down])} and ties on the rest."
    else:
        text = f"{model.name} improves with readout on {listing([pair[0].text.name for pair in up])} but drops on "
        text += f"{listing([_drop(pair) for pair in down])}."
    if all(pair[1].decision_score < 0 and pair[2].decision_score < 0 for pair in pairs):
        text += " It stays below zero on every task with both methods."
    return text


# Fig. 4: latency


def latency_intro(data: ReportData) -> str:
    """Introduction to the latency chart."""
    machines = len(data.machines())
    where = f", on {count(machines)} machines" if machines > 1 else ""
    return (
        "Jev’s times come from Jevals’ published logs, measured over the internet at concurrency 4. LLM-based-system-one timed "
        f"the local runs one request at a time{where}. The conditions differ, so this is not a controlled speed comparison."
    )


def latency_notes(data: ReportData) -> list[str]:
    """Notes under the latency chart."""
    jev = [task.jev for task in data.tasks]
    notes = [
        f"Jev’s published medians are {_ms_range([median(s) for s in jev])} and its p95 {_ms_range([_p95(s) for s in jev])}. "
        "These describe Jevals’ measurement conditions, not a service guarantee."
    ]
    for machine, models in data.machines():
        times = [median(task.score(system)) for task in data.tasks for model in models for system in (model.readout, model.verbalized)]
        notes.append(f"On the {machine}, local medians are {_ms_range(times)}.")
    notes += [
        f"{model.name} readout takes a median {ms(median(readout))} on {task.text.name}, {num(median(readout) / median(verbalized))} "
        f"times its verbalized median ({ms(median(verbalized))})."
        for model in data.models
        for task, verbalized, readout in _pairs(data, model)
        if median(readout) >= data.rules.slow_readout_factor * median(verbalized)
    ]
    if len(data.machines()) > 1:
        notes.append(_other_machines(data))
    return notes


# Fig. 5: metrics that disagree with the Decision Score


def _pick(data: ReportData, metric: Callable[[Score], float]) -> tuple[TaskScores, Score, Score]:
    candidates = [(task, a, b) for task in data.tasks for a, b in combinations(_contenders(task), 2)]
    task, a, b = max(candidates, key=lambda c: abs(c[1].decision_score - c[2].decision_score) / (abs(metric(c[1]) - metric(c[2])) + 1))
    high, low = sorted((a, b), key=lambda score: -score.decision_score)
    return task, high, low


def accuracy_example(data: ReportData) -> Example:
    """The pair with the most similar accuracy and the most different Decision Score."""
    task, high, low = _pick(data, lambda score: score.accuracy)
    equal = high.accuracy == low.accuracy
    shown = pct(high.accuracy) if equal else f"{pct(high.accuracy)} and {pct(low.accuracy)}"
    return Example(
        task=task.id,
        systems=[high.system, low.system, task.prior.system],
        title=f"{task.text.name}: {'equal' if equal else 'similar'} accuracy, different scores",
        caption=(
            f"{high.label} and {low.label} pick the right answer {'equally often' if equal else 'about equally often'} "
            f"({shown}) but score {num(high.decision_score)} and {num(low.decision_score)}. The label prior, which never reads "
            f"the question, gets {pct(task.prior.accuracy)}."
        ),
    )


def calibration_example(data: ReportData) -> Example:
    """The pair with the most similar calibration error and the most different Decision Score."""
    task, high, low = _pick(data, _ece)
    equal = _ece(high) == _ece(low)
    shown = num(_ece(high)) if equal else f"{num(_ece(high))} and {num(_ece(low))}"
    return Example(
        task=task.id,
        systems=[high.system, low.system],
        title=f"{task.text.name}: {'equal' if equal else 'similar'} calibration error, different scores",
        caption=(
            f"{high.label} and {low.label} have {'the same' if equal else 'almost the same'} calibration error ({shown}) but "
            f"score {num(high.decision_score)} and {num(low.decision_score)}. Calibration error is one useful view of the "
            "probabilities; it does not capture everything the Decision Score rewards."
        ),
    )


# Limits


def speed_limit(data: ReportData) -> str:
    """The latency limit, naming the machines when the local models ran on more than one."""
    text = "Published and local latencies come from different execution conditions."
    groups = data.machines()
    if len(groups) > 1:
        where = "; ".join(f"{listing([model.name for model in models])} on the {machine}" for machine, models in groups)
        text += f" The local models ran on {count(len(groups))} machines: {where}."
    return f"{text} Decisions/s divides decisions by summed decision time and is not throughput under concurrent load."
