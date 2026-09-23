"""Typed decision questions: the three primitives of a System One model such as Jev."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Primitive = Literal["noul", "choice", "score"]


@dataclass(frozen=True)
class Option:
    """One allowed answer. The description is the criterion text shown next to the name."""

    name: str
    description: str | None


@dataclass(frozen=True)
class Question:
    """A typed question about a state.

    noul: options are exactly "no" and "yes"; the answer is P(yes).
    choice: pick one of the options.
    score: options are ordered rubric levels, lowest first.
    """

    primitive: Primitive
    state: str
    instructions: str
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        """Validate the option set for the primitive."""
        names = [option.name for option in self.options]
        if len(names) < 2:
            raise ValueError("A question needs at least two options")
        if len(set(names)) != len(names):
            raise ValueError(f"Option names must be unique: {names}")
        if self.primitive == "noul" and sorted(names) != ["no", "yes"]:
            raise ValueError(f"A noul question has exactly the options 'no' and 'yes', got {names}")

    @property
    def option_names(self) -> list[str]:
        """Option names in display order."""
        return [option.name for option in self.options]
