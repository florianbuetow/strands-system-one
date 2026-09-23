"""The jevals option order: a seeded shuffle, bit-exact with the JavaScript in the jevals-data README."""

from __future__ import annotations

from collections.abc import Callable, Sequence


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return _u32(_u32(a) * _u32(b))


def fnv1a(text: str) -> int:
    """32-bit FNV-1a over UTF-16 code units, as JavaScript's charCodeAt yields them."""
    value = 0x811C9DC5
    encoded = text.encode("utf-16-le")
    for index in range(0, len(encoded), 2):
        value ^= encoded[index] | (encoded[index + 1] << 8)
        value = _imul(value, 0x01000193)
    return value


def mulberry32(seed: int) -> Callable[[], float]:
    """The mulberry32 generator; returns floats in [0, 1)."""
    state = _u32(seed)

    def next_value() -> float:
        nonlocal state
        state = _u32(state + 0x6D2B79F5)
        t = state
        t = _imul(t ^ (t >> 15), t | 1)
        t = _u32(t ^ _u32(t + _imul(t ^ (t >> 7), t | 61)))
        return _u32(t ^ (t >> 14)) / 4294967296

    return next_value


def shuffled[T](items: Sequence[T], seed: int) -> list[T]:
    """Fisher-Yates shuffle driven by mulberry32(seed)."""
    result = list(items)
    random = mulberry32(seed)
    for index in range(len(result) - 1, 0, -1):
        other = int(random() * (index + 1))
        result[index], result[other] = result[other], result[index]
    return result


def order_seed(epoch: int) -> int:
    """Repeats 0 and 1 share order seed 0; repeats 2, 3, 4 use seeds 1, 2, 3."""
    if epoch < 0:
        raise ValueError(f"Epoch must be >= 0, got {epoch}")
    return max(0, epoch - 1)


def option_order(item_id: str, options: Sequence[str], seed: int) -> list[str]:
    """Option order shown for an item: shuffled(options, fnv1a(f"{item_id}:{seed}"))."""
    return shuffled(options, fnv1a(f"{item_id}:{seed}"))
