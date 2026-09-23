"""System One agents: Strands agents that answer typed questions the way Jev does.

A System One model such as Jev reads a state and answers one typed question with a probability
for every allowed answer: noul (P(yes)), choice (one of N options) or score (a rubric level).
SystemOneAgent emulates that interface on top of a local chat model, with one of two methods:

- readout: the Strands agent runs on LogprobReadoutModel, which reads the option probabilities
  from next-token logprobs in one forward pass per label digit and writes no text.
- verbalized: the Strands agent runs on PrefilledOpenAIModel and the model writes its
  probabilities as JSON, exactly as jevals asks LLMs to (the "jevals adapter").
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from strands import Agent
from strands.models.model import Model
from strands.types.exceptions import MaxTokensReachedException

from openjev.config import Method, ModelConfig, ProviderConfig, ReadoutConfig, VerbalizedConfig
from openjev.probabilities import Listing, ListingMode, MalformedReplyError, normalize, parse_reply, pick
from openjev.prompts import adapter_prompt, readout_labels, readout_prompt
from openjev.providers import LogprobReadoutModel, PrefilledOpenAIModel
from openjev.questions import Option, Question


class UnansweredQuestionError(RuntimeError):
    """The agent gave no valid probability map, even after retries."""


@dataclass(frozen=True)
class Decision:
    """One answered question, with everything the benchmark logs."""

    stated: dict[str, float] | None
    mode: ListingMode
    raw: str
    malformed: bool
    retries: int
    seconds: float
    first_token_seconds: float | None
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Answer:
    """A validated answer: a probability for every option, in option order."""

    probabilities: dict[str, float]
    pick: str | None
    decision: Decision


class _FirstTokenClock:
    """Strands callback handler that records when the first text arrives."""

    def __init__(self, start: float) -> None:
        self._start = start
        self.first_token_seconds: float | None = None

    def __call__(self, **kwargs: object) -> None:
        if "data" in kwargs and self.first_token_seconds is None:
            self.first_token_seconds = time.perf_counter() - self._start


class SystemOneAgent:
    """Answers noul, choice and score questions with probabilities, like Jev."""

    def __init__(
        self,
        provider: ProviderConfig,
        model: ModelConfig,
        method: Method,
        readout: ReadoutConfig,
        verbalized: VerbalizedConfig,
    ) -> None:
        self.model = model
        self.method = method
        self._provider = provider
        self._readout = readout
        self._verbalized = verbalized

    def _client_args(self) -> dict[str, Any]:
        return {
            "base_url": self._provider.base_url,
            "api_key": self._provider.api_key,
            "timeout": self._provider.timeout_seconds,
            "max_retries": 0,
        }

    def _mode(self, question: Question) -> ListingMode:
        if self.method == "verbalized" and len(question.options) > self._verbalized.full_listing_max_options:
            return "top"
        return "full"

    def _strands_model(self, question: Question) -> Model:
        if self.method == "readout":
            labels = readout_labels(len(question.options))
            return LogprobReadoutModel(
                labels=dict(zip(labels, question.option_names, strict=True)),
                top_logprobs=self._readout.top_logprobs,
                temperature=self._readout.temperature,
                prefill=self.model.assistant_prefill,
                client_args=self._client_args(),
                model_id=self.model.model_id,
            )
        return PrefilledOpenAIModel(
            prefill=self.model.assistant_prefill,
            client_args=self._client_args(),
            model_id=self.model.model_id,
            params={
                "temperature": self._verbalized.temperature,
                "top_p": self._verbalized.top_p,
                "max_tokens": self._verbalized.max_tokens,
            },
        )

    def _prompt(self, question: Question) -> str:
        if self.method == "readout":
            return readout_prompt(question)
        listed = self._verbalized.top_listed_options if self._mode(question) == "top" else None
        return adapter_prompt(question, listed)

    def decide(self, question: Question) -> Decision:
        """Ask the question; malformed replies are retried as configured (verbalized only)."""
        mode = self._mode(question)
        prompt = self._prompt(question)
        attempts = 1 + self._verbalized.malformed_retries if self.method == "verbalized" else 1
        start = time.perf_counter()
        clock = _FirstTokenClock(start)
        input_tokens = 0
        output_tokens = 0
        raw = ""
        for attempt in range(attempts):
            agent = Agent(model=self._strands_model(question), callback_handler=clock)
            try:
                result = agent(prompt)
            except MaxTokensReachedException:
                raw = "<max_tokens reached>"
                continue
            finally:
                usage = agent.event_loop_metrics.accumulated_usage
                input_tokens += usage["inputTokens"]
                output_tokens += usage["outputTokens"]
            raw = str(result).strip()
            try:
                listing = parse_reply(raw, question.option_names, mode, self._verbalized.top_listed_options)
            except MalformedReplyError:
                continue
            return Decision(
                stated=listing.probabilities,
                mode=mode,
                raw=raw,
                malformed=False,
                retries=attempt,
                seconds=time.perf_counter() - start,
                first_token_seconds=clock.first_token_seconds,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        return Decision(
            stated=None,
            mode=mode,
            raw=raw,
            malformed=True,
            retries=attempts - 1,
            seconds=time.perf_counter() - start,
            first_token_seconds=clock.first_token_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def answer(self, question: Question) -> Answer:
        """Ask the question and return the full distribution and pick.

        Raises:
            UnansweredQuestionError: If no valid probability map came back.
        """
        decision = self.decide(question)
        if decision.stated is None:
            raise UnansweredQuestionError(f"{self.model.display_name} ({self.method}) gave no valid answer: {decision.raw!r}")
        vector = normalize(Listing(probabilities=decision.stated), question.option_names, decision.mode)
        index = pick(question.primitive, vector, question.option_names, list(decision.stated))
        return Answer(
            probabilities=dict(zip(question.option_names, vector, strict=True)),
            pick=None if index is None else question.option_names[index],
            decision=decision,
        )

    def noul(self, state: str, instructions: str, yes: str, no: str) -> float:
        """Return P(yes) for a yes/no question. `yes` and `no` describe when each answer applies."""
        question = Question(
            primitive="noul",
            state=state,
            instructions=instructions,
            options=(Option("no", no), Option("yes", yes)),
        )
        return self.answer(question).probabilities["yes"]

    def choice(self, state: str, instructions: str, options: dict[str, str | None]) -> Answer:
        """Pick one option; `options` maps each name to its description (or None)."""
        question = Question(
            primitive="choice",
            state=state,
            instructions=instructions,
            options=tuple(Option(name, description) for name, description in options.items()),
        )
        return self.answer(question)

    def score(self, state: str, instructions: str, levels: list[str]) -> Answer:
        """Place the state on a rubric; `levels` describes each level, lowest first."""
        question = Question(
            primitive="score",
            state=state,
            instructions=instructions,
            options=tuple(Option(str(index), level) for index, level in enumerate(levels)),
        )
        return self.answer(question)
