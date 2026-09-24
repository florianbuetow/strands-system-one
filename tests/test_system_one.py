"""LocalSystemOneClient: TypeSafe-SDK-shaped calls answered by local models."""

from pathlib import Path

import pytest
from strands.models.openai import OpenAIModel

from openjev.agents import UnansweredQuestionError
from openjev.system_one import (
    Choice,
    ChoiceAnswer,
    LocalSystemOneClient,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Score,
    ScoreAnswer,
    confidence,
)
from tests.fakes import FakeClient, fake_get_client

CONFIG = Path(__file__).resolve().parent.parent / "config" / "openjev.toml"


def install(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> None:
    monkeypatch.setattr(OpenAIModel, "_get_client", fake_get_client(client))


def test_confidence_matches_typesafe_formula() -> None:
    assert confidence([1.0, 0.0, 0.0]) == 1.0
    assert confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert confidence([0.9, 0.06, 0.04]) == pytest.approx((3 * 0.9 - 1) / 2)
    assert confidence([0.5, 0.5]) == pytest.approx(0.0)


def test_three_question_types_in_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = [
        '{"probabilities": {"yes": 0.9, "no": 0.1}}',
        '{"probabilities": {"angry": 0.7, "frustrated": 0.2, "calm": 0.1}}',
        '{"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}',
    ]
    client = FakeClient(replies=replies, logprobs={})
    install(monkeypatch, client)
    response = LocalSystemOneClient.from_config(CONFIG, "verbalized").system_one(
        model="qwen3.5-4b",
        state={"document": "I was charged twice. Please fix this ASAP."},
        questions={
            "billing": Noul(instructions="Is this ticket about billing?", criteria=None),
            "tone": Choice(instructions="What is the customer's tone?", criteria={"calm": None, "frustrated": None, "angry": None}),
            "urgency": Score(instructions="How urgent is this ticket?", criteria=["can wait", "this week", "today"]),
        },
    )
    assert response.model == "qwen3.5-4b-mlx"
    assert response.nouls["billing"].noul == pytest.approx(0.9)
    tone = response.choices["tone"]
    assert tone.choice == "angry"
    assert tone.probabilities == pytest.approx({"calm": 0.1, "frustrated": 0.2, "angry": 0.7})
    assert tone.confidence == pytest.approx((3 * 0.7 - 1) / 2)
    urgency = response.scores["urgency"]
    assert urgency.score == pytest.approx(0.2 + 2 * 0.7)
    assert urgency.legend == {"0": "can wait", "1": "this week", "2": "today"}
    assert [answer.type for answer in response.answers.values()] == ["noul", "choice", "score"]
    assert isinstance(response.answers["billing"], NoulAnswer)
    assert isinstance(response.answers["tone"], ChoiceAnswer)
    assert isinstance(response.answers["urgency"], ScoreAnswer)
    assert '"document": "I was charged twice. Please fix this ASAP."' in client.requests[0]["messages"][0]["content"][0]["text"]


def test_noul_criteria_reach_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(replies=[], logprobs={"": [("1", 0.6), ("0", 0.4)]})
    install(monkeypatch, client)
    criteria = NoulCriteria(true="Mentions a prior attempt", false="No sign of previous contact")
    response = LocalSystemOneClient.from_config(CONFIG, "readout").system_one(
        model="qwen3-0.6b",
        state="I have asked three times now.",
        questions={"is_repeat_contact": Noul(instructions="Has the customer contacted support before?", criteria=criteria)},
    )
    assert response.nouls["is_repeat_contact"].noul == pytest.approx(0.6)
    prompt = client.requests[0]["messages"][0]["content"][0]["text"]
    assert "0. no: No sign of previous contact" in prompt
    assert "1. yes: Mentions a prior attempt" in prompt
    assert "STATE (JSON):\nI have asked three times now." in prompt


def test_unknown_model_key_fails() -> None:
    with pytest.raises(KeyError, match="known keys"):
        LocalSystemOneClient.from_config(CONFIG, "readout").system_one(
            model="jev-latest", state="x", questions={"q": Noul(instructions="Is it?", criteria=None)}
        )


def test_invalid_answer_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(replies=["no json"] * 3, logprobs={})
    install(monkeypatch, client)
    with pytest.raises(UnansweredQuestionError):
        LocalSystemOneClient.from_config(CONFIG, "verbalized").system_one(
            model="minicpm5-2b", state="x", questions={"q": Choice(instructions="Which?", criteria={"a": None, "b": None})}
        )
