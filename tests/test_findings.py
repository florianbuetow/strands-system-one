"""Findings rules on hand-made report numbers."""

from llm_system_one.findings import accuracy_example, latency_notes, quality_intro, quality_notes, scatter_notes, summary
from tests.report_numbers import JEV, PRIOR, base, local, plain, report, score


def test_jev_leading_every_task_is_the_first_finding() -> None:
    first = summary(plain())[0]
    assert first.title == "Jev beats every local model on every task."
    assert "The best local results are 40.0 on PubMedQA (Big readout) and 45.0 on Banking77 (Big readout)." in first.body


def test_a_local_model_ahead_of_jev_changes_the_title_and_the_chart_intro() -> None:
    data = report(base(60) + local("big", 40, 30) + local("small", 10, 5), base(50) + local("big", 55, 20) + local("small", 15, 5))
    assert summary(data)[0].title == "Jev beats every local model on PubMedQA."
    assert quality_intro(data).endswith("Big readout on Banking77 lands to the right of it.")


def test_baseline_finding_names_the_only_interval_above_zero() -> None:
    pubmedqa = [
        score(JEV, "jev", 8.0, lo=-4.0),
        score(PRIOR, "prior", 0.0, "base rates"),
        *local("big", 4.8, -20, lo=0.2),
        *local("small", -30, -40),
    ]
    data = report(pubmedqa, base(50) + local("big", 45, 20) + local("small", 15, 5))
    titles = [finding.title for finding in summary(data)]
    assert "Only Big readout clearly beats the baseline on PubMedQA." in titles


def test_accuracy_finding_picks_the_most_accurate_system_with_a_lower_score() -> None:
    pubmedqa = [
        score(JEV, "jev", 9.0, acc=55.0),
        score(PRIOR, "prior", 0.0, "base rates"),
        *local("big", 5, 1),
        *local("small", -10, -20, acc=60.0),
    ]
    data = report(pubmedqa, base(50) + local("big", 45, 20) + local("small", 15, 5))
    finding = next(f for f in summary(data) if f.title == "Higher accuracy can hide worse probabilities.")
    assert finding.body == (
        "On PubMedQA, Small readout has the highest accuracy (60.0%) and a Decision Score of −10.0. Jev has 55.0% accuracy and scores 9.0."
    )


def test_accuracy_finding_is_left_out_when_accuracy_and_score_agree() -> None:
    assert all(f.title != "Higher accuracy can hide worse probabilities." for f in summary(plain()))


def test_readout_finding_separates_improvements_from_drops() -> None:
    banking77 = [
        *base(50),
        *local("big", 45, 20),
        score("small-readout", "local", 10, "readout", acc=60.0),
        score("small-verbalized", "local", 15, "verbalized", acc=40.0),
    ]
    data = report(base(60) + local("big", 40, 30) + local("small", 10, 5), banking77)
    finding = next(f for f in summary(data) if f.title == "Readout is not an automatic improvement.")
    assert finding.body == "It raises the score of Big on every task. For Small it lowers the score on Banking77 while accuracy rises."


def test_sparse_readout_is_flagged_only_where_a_task_has_more_answers_than_logprobs() -> None:
    pubmedqa = base(60) + local("big", 40, 30, coverage=1.0) + local("small", 10, 5)
    banking77 = base(50) + local("big", 45, 20, coverage=6.1) + local("small", 15, 5, coverage=70.0)
    notes = quality_notes(report(pubmedqa, banking77))
    sparse = [note for note in notes if "puts probability on only" in note]
    assert len(sparse) == 1
    assert sparse[0].startswith("Big readout puts probability on only 6.1 of the 77 intents on Banking77 on average, against 70.0")


def test_slow_readout_and_other_machines_get_latency_notes() -> None:
    banking77 = [
        *base(50),
        score("big-readout", "local", 45, "readout", p50=4185.0),
        score("big-verbalized", "local", 20, "verbalized", p50=1924.0),
        *local("small", 15, 5),
    ]
    data = report(base(60) + local("big", 40, 30) + local("small", 10, 5), banking77, machines=("Mac", "Server"))
    notes = latency_notes(data)
    assert "Big readout takes a median 4,185 ms on Banking77, 2.2 times its verbalized median (1,924 ms)." in notes
    assert notes[-1] == "Small ran on the Server, so those timings are not comparable with the models on the Mac."


def test_accuracy_example_prefers_equal_accuracy_with_the_widest_score_gap() -> None:
    pubmedqa = [
        score(JEV, "jev", 9.2, acc=41.3),
        score(PRIOR, "prior", 0.0, "base rates", acc=41.7),
        *local("big", 5, 1, acc=30.0),
        *local("small", -60.7, -20, acc=41.3),
    ]
    example = accuracy_example(report(pubmedqa, base(50) + local("big", 45, 20) + local("small", 15, 5)))
    assert example.systems == [JEV, "small-readout", PRIOR]
    assert "pick the right answer equally often (41.3%) but score 9.2 and −60.7" in example.caption


def test_scatter_note_names_systems_faster_and_better_than_jev() -> None:
    pubmedqa = [score(JEV, "jev", 60), score(PRIOR, "prior", 0.0, "base rates"), score("ref", "reference", 70, "verbalized", p50=300.0)]
    data = report(pubmedqa + local("big", 40, 30) + local("small", 10, 5), base(50) + local("big", 45, 20) + local("small", 15, 5))
    assert scatter_notes(data) == ["Faster than Jev and higher-scoring: ref on PubMedQA."]
