"""Probability replies: validation, normalization and picks, following the jevals rules.

See https://jevals.com/methodology/ ("Probability vectors").
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal, cast

from openjev.questions import Primitive

ListingMode = Literal["full", "top"]


class MalformedReplyError(ValueError):
    """The reply is not a valid probability map for the question."""


@dataclass(frozen=True)
class Listing:
    """Probabilities as the system stated them, in the order it listed them."""

    probabilities: dict[str, float]


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise MalformedReplyError(f"Duplicate keys in reply: {keys}")
    return dict(pairs)


def _extract_object(text: str) -> object:
    start = text.find("{")
    if start == -1:
        raise MalformedReplyError("Reply contains no JSON object")
    try:
        value, end = json.JSONDecoder(object_pairs_hook=_reject_duplicates).raw_decode(text, start)
    except json.JSONDecodeError as error:
        raise MalformedReplyError(f"Reply is not valid JSON: {error}") from error
    surrounding = text[:start] + text[end:]
    if "{" in surrounding or "}" in surrounding:
        raise MalformedReplyError("Reply has other text containing braces around the JSON object")
    return value


def _probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MalformedReplyError(f"Probability for {name!r} is not a number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0 or number > 1:
        raise MalformedReplyError(f"Probability for {name!r} is outside [0, 1]: {number}")
    return number


def parse_reply(text: str, option_names: list[str], mode: ListingMode, top_listed: int) -> Listing:
    """Validate a reply of the form {"probabilities": {"<option>": p}}.

    Args:
        text: The raw reply text.
        option_names: The allowed option names (exact match).
        mode: "full" when every option was requested, "top" when only the most likely ones were.
        top_listed: Maximum number of listed options in "top" mode.

    Raises:
        MalformedReplyError: If the reply breaks any jevals validity rule.
    """
    value = _extract_object(text)
    if not isinstance(value, dict):
        raise MalformedReplyError("Reply must be a JSON object")
    fields = cast(dict[str, object], value)
    if list(fields) != ["probabilities"]:
        raise MalformedReplyError("Reply must be an object whose only field is 'probabilities'")
    if not isinstance(fields["probabilities"], dict) or not fields["probabilities"]:
        raise MalformedReplyError("'probabilities' must be a non-empty object")
    stated = cast(dict[str, object], fields["probabilities"])
    allowed = set(option_names)
    probabilities: dict[str, float] = {}
    for name, raw in stated.items():
        if name not in allowed:
            raise MalformedReplyError(f"Unknown option {name!r}")
        probabilities[name] = _probability(name, raw)
    if mode == "top" and len(probabilities) > top_listed:
        raise MalformedReplyError(f"Listed {len(probabilities)} options, asked for at most {top_listed}")
    if sum(probabilities.values()) <= 0:
        raise MalformedReplyError("Probabilities sum to 0")
    return Listing(probabilities=probabilities)


def normalize(listing: Listing, option_names: list[str], mode: ListingMode) -> list[float]:
    """Turn stated probabilities into a distribution over all options, in option order.

    Sums above 1 are divided by the sum. Sums below 1 are divided by the sum in "full" mode;
    in "top" mode the remainder is spread evenly over the unlisted options.
    """
    total = sum(listing.probabilities.values())
    if total <= 0:
        raise MalformedReplyError("Probabilities sum to 0")
    unlisted = [name for name in option_names if name not in listing.probabilities]
    if mode == "top" and total < 1 and unlisted:
        spread = {**dict.fromkeys(unlisted, (1 - total) / len(unlisted)), **listing.probabilities}
        return [spread[name] for name in option_names]
    divided = {**dict.fromkeys(unlisted, 0.0), **{name: p / total for name, p in listing.probabilities.items()}}
    return [divided[name] for name in option_names]


def pick(primitive: Primitive, vector: list[float], option_names: list[str], listed_first: list[str]) -> int | None:
    """Index of the most probable option, or None for a yes/no answer of exactly 0.5.

    Ties: choice goes to the option the system listed first (then display order),
    score goes to the lower level.
    """
    best = max(vector)
    tied = [index for index, value in enumerate(vector) if value == best]
    match primitive:
        case "noul":
            return None if len(tied) > 1 else tied[0]
        case "score":
            return tied[0]
        case "choice":
            for name in listed_first:
                index = option_names.index(name)
                if index in tied:
                    return index
            return tied[0]
