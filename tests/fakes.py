"""A fake OpenAI-compatible client, so the Strands agents can be tested without a server."""

from __future__ import annotations

import math
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any


def _usage(prompt: int, completion: int) -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion, prompt_tokens_details=None)


class _Stream:
    """Streams one text reply the way the OpenAI SDK does: text chunk, finish chunk, usage chunk."""

    def __init__(self, text: str) -> None:
        def chunk(content: str | None, finish: str | None) -> SimpleNamespace:
            delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
            return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)

        self._events = [chunk(text, None), chunk(None, "stop"), SimpleNamespace(choices=[], usage=_usage(100, 10))]

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class FakeClient:
    """Answers streaming requests with queued replies and logprob requests from a prefix table."""

    def __init__(self, replies: list[str], logprobs: dict[str, list[tuple[str, float]]]) -> None:
        self.replies = replies
        self.logprobs = logprobs
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **request: Any) -> Any:
        self.requests.append(request)
        if request["stream"]:
            return _Stream(self.replies.pop(0))
        opening: str = request["messages"][-1]["content"]
        prefix = opening.split("</think>\n\n")[-1]
        tops = [SimpleNamespace(token=token, logprob=math.log(p)) for token, p in self.logprobs[prefix]]
        content = SimpleNamespace(token=tops[0].token, logprob=tops[0].logprob, top_logprobs=tops)
        choice = SimpleNamespace(logprobs=SimpleNamespace(content=[content]), finish_reason="length")
        return SimpleNamespace(choices=[choice], usage=_usage(100, 1))


def fake_get_client(client: FakeClient) -> Any:
    """Replacement for OpenAIModel._get_client that yields the fake client."""

    @asynccontextmanager
    async def get_client(self: object) -> AsyncGenerator[FakeClient]:
        yield client

    return get_client
