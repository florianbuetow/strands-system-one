"""A local client shaped like TypeSafe's Python SDK, so code written for Jev runs on local models.

TypeSafe SDK (https://docs.typesafe.ai/sdk/python):

    response = client.system_one(model="jev-latest", state=..., questions={"id": Noul(...), ...})
    response.answers["id"].noul / .choice / .score

LocalSystemOneClient.system_one takes the same arguments and returns answers with the same fields.
Questions are answered one after another, not in one parallel pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openjev.agents import SystemOneAgent
from openjev.config import Config, Method, load_config
from openjev.questions import Option, Question


@dataclass(frozen=True)
class NoulCriteria:
    """What a yes (true) and a no (false) mean."""

    true: str
    false: str


@dataclass(frozen=True)
class Noul:
    """A yes/no question; the answer is P(yes). Pass criteria=None to rely on the instructions alone."""

    instructions: str
    criteria: NoulCriteria | None


@dataclass(frozen=True)
class Choice:
    """Pick one option. criteria maps each option name to its description, or None."""

    instructions: str
    criteria: dict[str, str | None]


@dataclass(frozen=True)
class Score:
    """Rate the state on ordered levels. criteria describes each level, lowest first."""

    instructions: str
    criteria: list[str]


type SystemOneQuestion = Noul | Choice | Score


@dataclass(frozen=True)
class NoulAnswer:
    """Probability that the answer is yes."""

    noul: float

    @property
    def type(self) -> Literal["noul"]:
        """The question type, as in the TypeSafe SDK."""
        return "noul"


@dataclass(frozen=True)
class ChoiceAnswer:
    """The most probable option, its confidence, and a probability per option."""

    choice: str
    confidence: float
    probabilities: dict[str, float]

    @property
    def type(self) -> Literal["choice"]:
        """The question type, as in the TypeSafe SDK."""
        return "choice"


@dataclass(frozen=True)
class ScoreAnswer:
    """The expected level (probability-weighted average), its confidence, the levels, and a probability per level."""

    score: float
    confidence: float
    legend: dict[str, str]
    probabilities: dict[str, float]

    @property
    def type(self) -> Literal["score"]:
        """The question type, as in the TypeSafe SDK."""
        return "score"


type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class SystemOneResponse:
    """All answers of one call, keyed by the question ids you chose."""

    model: str
    answers: dict[str, Answer]

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        """The yes/no answers."""
        return {key: answer for key, answer in self.answers.items() if isinstance(answer, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """The choice answers."""
        return {key: answer for key, answer in self.answers.items() if isinstance(answer, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """The score answers."""
        return {key: answer for key, answer in self.answers.items() if isinstance(answer, ScoreAnswer)}


def confidence(probabilities: list[float]) -> float:
    """How concentrated a distribution is: 1 when one option has everything, 0 when it is uniform.

    (K · largest probability − 1) / (K − 1), the formula TypeSafe's confidence page uses to explain its
    confidence. Jev's exact computation is not published.
    """
    count = len(probabilities)
    return min(1.0, max(0.0, (count * max(probabilities) - 1) / (count - 1)))


def _state_text(state: str | dict[str, object]) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _question(state: str, question: SystemOneQuestion) -> Question:
    match question:
        case Noul(instructions=instructions, criteria=criteria):
            no = None if criteria is None else criteria.false
            yes = None if criteria is None else criteria.true
            return Question("noul", state, instructions, (Option("no", no), Option("yes", yes)))
        case Choice(instructions=instructions, criteria=criteria):
            return Question("choice", state, instructions, tuple(Option(name, text) for name, text in criteria.items()))
        case Score(instructions=instructions, criteria=levels):
            return Question("score", state, instructions, tuple(Option(str(index), text) for index, text in enumerate(levels)))


class LocalSystemOneClient:
    """Answers Jev-style questions with a local model through a SystemOneAgent."""

    def __init__(self, config: Config, method: Method) -> None:
        """Create a client.

        Args:
            config: The openjev configuration (provider and models).
            method: "readout" (probabilities from logprobs, no text) or "verbalized" (the model writes JSON).
        """
        self._config = config
        self._method: Method = method
        self._agents: dict[str, SystemOneAgent] = {}

    @classmethod
    def from_config(cls, path: Path, method: Method) -> LocalSystemOneClient:
        """Create a client from a config file."""
        return cls(load_config(path), method)

    def _agent(self, model: str) -> SystemOneAgent:
        if model not in self._agents:
            config = self._config
            self._agents[model] = SystemOneAgent(config.provider, config.model(model), self._method, config.readout, config.verbalized)
        return self._agents[model]

    def system_one(self, model: str, state: str | dict[str, object], questions: dict[str, SystemOneQuestion]) -> SystemOneResponse:
        """Answer every question about the state.

        Args:
            model: A model key from the config, e.g. "qwen3.5-4b".
            state: Text, or a JSON-serializable dict.
            questions: Your question ids mapped to Noul, Choice or Score questions.

        Raises:
            UnansweredQuestionError: If the model gives no valid answer to a question.
        """
        agent = self._agent(model)
        text = _state_text(state)
        answers: dict[str, Answer] = {}
        for key, question in questions.items():
            result = agent.answer(_question(text, question))
            probabilities = result.probabilities
            match question:
                case Noul():
                    answers[key] = NoulAnswer(noul=probabilities["yes"])
                case Choice():
                    if result.pick is None:
                        raise ValueError("A choice answer always has a pick")
                    answers[key] = ChoiceAnswer(
                        choice=result.pick,
                        confidence=confidence(list(probabilities.values())),
                        probabilities=probabilities,
                    )
                case Score(criteria=levels):
                    answers[key] = ScoreAnswer(
                        score=sum(int(level) * p for level, p in probabilities.items()),
                        confidence=confidence(list(probabilities.values())),
                        legend={str(index): text for index, text in enumerate(levels)},
                        probabilities=probabilities,
                    )
        return SystemOneResponse(model=agent.model.model_id, answers=answers)
