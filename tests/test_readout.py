"""Label probabilities read digit by digit from next-token logprobs."""

import math

import pytest

from openjev.readout import NextTokens, TokenLogprob, read_label_masses


def fake_next_tokens(table: dict[str, list[tuple[str, float]]]) -> tuple[NextTokens, list[str]]:
    calls: list[str] = []

    async def next_tokens(prefix: str) -> list[TokenLogprob]:
        calls.append(prefix)
        return [TokenLogprob(token, math.log(p)) for token, p in table[prefix]]

    return next_tokens, calls


async def test_single_digit_labels_need_one_request() -> None:
    next_tokens, calls = fake_next_tokens({"": [("1", 0.7), (" 0", 0.2), ("Yes", 0.1)]})
    masses = await read_label_masses(next_tokens, ["0", "1"])
    assert masses == pytest.approx({"0": 0.2, "1": 0.7})
    assert calls == [""]


async def test_two_digit_labels_use_chain_rule() -> None:
    next_tokens, calls = fake_next_tokens(
        {
            "": [("0", 0.6), ("1", 0.3), ("<|im_end|>", 0.1)],
            "0": [("7", 0.5), ("3", 0.5)],
            "1": [("2", 1.0)],
        }
    )
    masses = await read_label_masses(next_tokens, [f"{i:02d}" for i in range(13)])
    assert masses["07"] == pytest.approx(0.3)
    assert masses["03"] == pytest.approx(0.3)
    assert masses["12"] == pytest.approx(0.3)
    assert sorted(calls) == ["", "0", "1"]


async def test_multi_digit_token_completes_label_directly() -> None:
    next_tokens, _ = fake_next_tokens({"": [("07", 0.9), ("1", 0.1)], "1": [("0", 1.0)]})
    masses = await read_label_masses(next_tokens, [f"{i:02d}" for i in range(11)])
    assert masses["07"] == pytest.approx(0.9)
    assert masses["10"] == pytest.approx(0.1)


async def test_prefix_that_leads_nowhere_is_not_expanded() -> None:
    next_tokens, calls = fake_next_tokens({"": [("9", 1.0)]})
    masses = await read_label_masses(next_tokens, ["00", "01"])
    assert sum(masses.values()) == 0
    assert calls == [""]


async def test_labels_must_share_one_length() -> None:
    next_tokens, _ = fake_next_tokens({})
    with pytest.raises(ValueError, match="same length"):
        await read_label_masses(next_tokens, ["0", "10"])
