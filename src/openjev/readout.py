"""Read a probability for every option label from next-token logprobs.

Servers cap how many top logprobs they return (LM Studio: 10), so labels longer than one
token are read digit by digit: P("07") = P("0") * P("7" | "0"). Each distinct prefix costs
one request of one token; the shared prompt prefix is served from the server's prompt cache.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenLogprob:
    """One candidate next token and its log probability."""

    token: str
    logprob: float


type NextTokens = Callable[[str], Awaitable[list[TokenLogprob]]]


async def read_label_masses(next_tokens: NextTokens, labels: list[str]) -> dict[str, float]:
    """Return the probability mass the model puts on each label.

    Args:
        next_tokens: Returns the top next tokens after the assistant has written the given prefix.
        labels: Option labels of equal length, so that no label is a prefix of another.

    Returns:
        Unnormalized mass per label; labels outside the returned top tokens get 0.
    """
    if len({len(label) for label in labels}) != 1:
        raise ValueError(f"Labels must all have the same length: {labels}")
    masses = dict.fromkeys(labels, 0.0)
    frontier: dict[str, float] = {"": 1.0}
    while frontier:
        next_frontier: defaultdict[str, float] = defaultdict(float)
        for prefix, prefix_mass in frontier.items():
            for candidate in await next_tokens(prefix):
                text = prefix + candidate.token.strip()
                if text == prefix:
                    continue
                mass = prefix_mass * math.exp(candidate.logprob)
                if text in masses:
                    masses[text] += mass
                elif any(label.startswith(text) for label in labels):
                    next_frontier[text] += mass
        frontier = dict(next_frontier)
    return masses
