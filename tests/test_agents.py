"""SystemOneAgent: Strands agents answering typed questions through the two providers."""

from pathlib import Path

import pytest
from strands.models.openai import OpenAIModel

from openjev.agents import SystemOneAgent, UnansweredQuestionError
from openjev.config import Config, Method, load_config
from openjev.questions import Option, Question
from tests.fakes import FakeClient, fake_get_client

CONFIG = Path(__file__).resolve().parent.parent / "config" / "openjev.toml"


@pytest.fixture
def config() -> Config:
    return load_config(CONFIG)


def agent(config: Config, method: Method) -> SystemOneAgent:
    return SystemOneAgent(config.provider, config.models[0], method, config.readout, config.verbalized)


def install(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> None:
    monkeypatch.setattr(OpenAIModel, "_get_client", fake_get_client(client))


def test_readout_noul_reads_p_yes_from_logprobs(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(replies=[], logprobs={"": [("1", 0.8), ("0", 0.15), ("Yes", 0.05)]})
    install(monkeypatch, client)
    p_yes = agent(config, "readout").noul('{"x": 1}', "Is it?", yes="It is", no="It is not")
    assert p_yes == pytest.approx(0.8 / 0.95)
    request = client.requests[0]
    assert request["logprobs"] is True
    assert request["top_logprobs"] == config.readout.top_logprobs
    assert request["messages"][-1] == {"role": "assistant", "content": config.models[0].assistant_prefill}
    assert "0. no: It is not" in request["messages"][0]["content"][0]["text"]


def test_readout_choice_over_many_options_reads_digit_by_digit(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(replies=[], logprobs={"": [("1", 0.9), ("0", 0.1)], "1": [("2", 1.0)], "0": [("3", 1.0)]})
    install(monkeypatch, client)
    options: dict[str, str | None] = {f"intent_{i}": None for i in range(15)}
    answer = agent(config, "readout").choice("{}", "Which intent?", options)
    assert answer.pick == "intent_12"
    assert answer.probabilities["intent_12"] == pytest.approx(0.9)
    assert answer.probabilities["intent_3"] == pytest.approx(0.1)
    assert len(client.requests) == 3


def test_verbalized_choice_parses_json_and_sends_prefill(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(replies=['{"probabilities": {"b": 0.9, "a": 0.1}}'], logprobs={})
    install(monkeypatch, client)
    answer = agent(config, "verbalized").choice("{}", "Which?", {"a": None, "b": None, "c": None})
    assert answer.pick == "b"
    assert answer.probabilities == pytest.approx({"a": 0.1, "b": 0.9, "c": 0.0})
    request = client.requests[0]
    assert request["temperature"] == config.verbalized.temperature
    assert request["messages"][-1] == {"role": "assistant", "content": config.models[0].assistant_prefill}


def test_verbalized_retries_malformed_replies(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(replies=["not json", '{"probabilities": {"z": 1}}', '{"probabilities": {"yes": 0.7, "no": 0.3}}'], logprobs={})
    install(monkeypatch, client)
    question = Question("noul", "{}", "Is it?", (Option("no", None), Option("yes", None)))
    decision = agent(config, "verbalized").decide(question)
    assert decision.retries == 2
    assert not decision.malformed
    assert decision.stated == {"yes": 0.7, "no": 0.3}


def test_verbalized_gives_up_after_retries(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(replies=["no"] * (config.verbalized.malformed_retries + 1), logprobs={})
    install(monkeypatch, client)
    with pytest.raises(UnansweredQuestionError, match="no valid answer"):
        agent(config, "verbalized").score("{}", "How good?", ["bad", "ok", "good"])
    assert len(client.requests) == config.verbalized.malformed_retries + 1
