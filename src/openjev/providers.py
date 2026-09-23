"""Strands model providers for OpenAI-compatible servers such as LM Studio.

PrefilledOpenAIModel opens the assistant turn with fixed text. With an empty think block as
the prefill, reasoning models answer directly instead of thinking first.

LogprobReadoutModel is the System One provider: it writes no text. It reads the probability of
every option label from next-token logprobs and returns them to the agent as a JSON probability map.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any, cast, override

from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam
from strands.models.openai import OpenAIModel
from strands.types.content import Messages
from strands.types.streaming import StreamEvent

from openjev.readout import TokenLogprob, read_label_masses


class PrefilledOpenAIModel(OpenAIModel):
    """OpenAIModel whose requests end with a partial assistant message the model continues."""

    def __init__(self, *, prefill: str, client_args: dict[str, Any], model_id: str, params: dict[str, Any]) -> None:
        super().__init__(client_args=client_args, model_id=model_id, params=params)
        self._prefill = prefill

    @override
    def format_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Format the request and append the assistant prefill."""
        request = super().format_request(*args, **kwargs)
        if self._prefill:
            request["messages"].append({"role": "assistant", "content": self._prefill})
        return request


class LogprobReadoutModel(PrefilledOpenAIModel):
    """Answers with option probabilities read from logprobs, without generating text."""

    def __init__(
        self,
        *,
        labels: dict[str, str],
        top_logprobs: int,
        temperature: float,
        prefill: str,
        client_args: dict[str, Any],
        model_id: str,
    ) -> None:
        """Create the provider.

        Args:
            labels: Maps each label the prompt shows (e.g. "07") to its option name.
            top_logprobs: How many top next tokens the server returns per request.
            temperature: Sampling temperature sent with each one-token request.
            prefill: Text that opens the assistant turn.
            client_args: Arguments for the OpenAI client (base_url, api_key, timeout).
            model_id: Model name on the server.
        """
        super().__init__(prefill=prefill, client_args=client_args, model_id=model_id, params={})
        self._model_id = model_id
        self._labels = labels
        self._top_logprobs = top_logprobs
        self._temperature = temperature

    async def _read(self, client: AsyncOpenAI, history: list[ChatCompletionMessageParam]) -> tuple[dict[str, float], CompletionUsage]:
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        async def next_tokens(prefix: str) -> list[TokenLogprob]:
            opening: ChatCompletionAssistantMessageParam = {"role": "assistant", "content": self._prefill + prefix}
            response = await client.chat.completions.create(
                model=self._model_id,
                messages=[*history, opening],
                # LM Studio generates max_tokens - 1 tokens; only the first token's logprobs are read.
                max_tokens=2,
                logprobs=True,
                top_logprobs=self._top_logprobs,
                temperature=self._temperature,
                stream=False,
            )
            if response.usage is None:
                raise RuntimeError("Server returned no token usage")
            usage.prompt_tokens += response.usage.prompt_tokens
            usage.completion_tokens += response.usage.completion_tokens
            usage.total_tokens += response.usage.total_tokens
            choice = response.choices[0]
            content = None if choice.logprobs is None else choice.logprobs.content
            if not content:
                if choice.finish_reason == "stop":
                    return []  # the model ended its reply here; the server sends no logprobs for that step
                raise RuntimeError(f"Server returned no logprobs for model {self._model_id}")
            return [TokenLogprob(token=top.token, logprob=top.logprob) for top in content[0].top_logprobs]

        masses = await read_label_masses(next_tokens, list(self._labels))
        return {self._labels[label]: mass for label, mass in masses.items()}, usage

    @override
    async def stream(self, messages: Messages, *args: Any, **kwargs: Any) -> AsyncGenerator[StreamEvent, None]:
        """Read the option probabilities and emit them as the assistant's text."""
        if any(arg is not None for arg in args):
            raise ValueError("The readout model takes no tools and no system prompt")
        history = cast(list[ChatCompletionMessageParam], self.format_request_messages(messages))
        async with self._get_client() as client:
            masses, usage = await self._read(client, history)
        total = sum(masses.values())
        stated = {name: mass / total for name, mass in masses.items() if mass > 0} if total > 0 else {}
        text = json.dumps({"probabilities": stated})
        yield self.format_chunk({"chunk_type": "message_start"})
        yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})
        yield self.format_chunk({"chunk_type": "content_delta", "data_type": "text", "data": text})
        yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})
        yield self.format_chunk({"chunk_type": "message_stop", "data": "stop"})
        yield self.format_chunk({"chunk_type": "metadata", "data": usage})
