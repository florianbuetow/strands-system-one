"""Reply validation and normalization follow the jevals rules."""

import pytest

from llm_system_one.probabilities import Listing, ListingMode, MalformedReplyError, normalize, parse_reply, pick

NAMES = ["a", "b", "c", "d"]


def parse(text: str, mode: ListingMode = "full", top: int = 5) -> dict[str, float]:
    return parse_reply(text, NAMES, mode, top).probabilities


def test_accepts_plain_and_fenced_json() -> None:
    assert parse('{"probabilities": {"a": 0.7, "b": 0.3}}') == {"a": 0.7, "b": 0.3}
    assert parse('```json\n{"probabilities": {"b": 1}}\n```') == {"b": 1.0}


def test_keeps_listing_order() -> None:
    assert list(parse('{"probabilities": {"c": 0.5, "a": 0.5}}')) == ["c", "a"]


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        '{"probabilities": {"a": 0.5, "a": 0.5}}',
        '{"probabilities": {"e": 1}}',
        '{"probabilities": {"a": 1.5}}',
        '{"probabilities": {"a": -0.1}}',
        '{"probabilities": {"a": "high"}}',
        '{"probabilities": {"a": true}}',
        '{"probabilities": {"a": 0, "b": 0}}',
        '{"probabilities": {}}',
        '{"probabilities": {"a": 1}, "reason": "x"}',
        'Note {x} {"probabilities": {"a": 1}}',
        '{"probabilities": {"a": 1',
        "[1, 2]",
    ],
)
def test_rejects_malformed_replies(text: str) -> None:
    with pytest.raises(MalformedReplyError):
        parse(text)


def test_top_mode_limits_listed_options() -> None:
    with pytest.raises(MalformedReplyError, match="at most 2"):
        parse('{"probabilities": {"a": 0.4, "b": 0.3, "c": 0.3}}', mode="top", top=2)


def test_full_mode_divides_by_sum() -> None:
    assert normalize(Listing({"a": 0.2, "b": 0.2}), NAMES, "full") == [0.5, 0.5, 0.0, 0.0]
    assert normalize(Listing({"a": 1.0, "b": 1.0}), NAMES, "full") == [0.5, 0.5, 0.0, 0.0]


def test_top_mode_spreads_remainder_over_unlisted() -> None:
    assert normalize(Listing({"a": 0.6, "b": 0.2}), NAMES, "top") == pytest.approx([0.6, 0.2, 0.1, 0.1])


def test_top_mode_divides_when_sum_exceeds_one() -> None:
    assert normalize(Listing({"a": 1.0, "b": 1.0}), NAMES, "top") == [0.5, 0.5, 0.0, 0.0]


def test_noul_exactly_half_has_no_pick() -> None:
    assert pick("noul", [0.5, 0.5], ["no", "yes"], []) is None
    assert pick("noul", [0.4, 0.6], ["no", "yes"], []) == 1


def test_choice_tie_goes_to_first_listed() -> None:
    assert pick("choice", [0.4, 0.4, 0.2, 0.0], NAMES, ["b", "a"]) == 1


def test_score_tie_goes_to_lower_level() -> None:
    assert pick("score", [0.1, 0.45, 0.45], ["0", "1", "2"], []) == 1
