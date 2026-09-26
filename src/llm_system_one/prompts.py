"""Prompts for the two answer methods.

The verbalized prompt follows the published jevals LLM adapter (prompt v1, hash 0383a0e3e592).
jevals publishes the template but not the text of its {options_heading} and {what}
placeholders, so those two lines are written here and the prompt hash will differ.
"""

from __future__ import annotations

from llm_system_one.questions import Option, Primitive, Question

ADAPTER_TEMPLATE = """You are answering one typed decision question about a state.

STATE (JSON):
{state}
QUESTION: {instructions}
{options_heading}
{options}

Give a probability for {what}
Reply with only this JSON object and nothing else:
{{"probabilities": {{"<option>": <probability between 0 and 1>}}}}"""

READOUT_TEMPLATE = """You are answering one typed decision question about a state.

STATE (JSON):
{state}
QUESTION: {instructions}
{options_heading}
{options}

Reply with only the number of the correct {noun}."""


def _heading(primitive: Primitive) -> str:
    match primitive:
        case "noul":
            return "ANSWERS:"
        case "choice":
            return "OPTIONS:"
        case "score":
            return "LEVELS (ordered from lowest to highest):"


def _noun(primitive: Primitive) -> str:
    match primitive:
        case "noul":
            return "answer"
        case "choice":
            return "option"
        case "score":
            return "level"


def _describe(option: Option) -> str:
    if option.description is None:
        return option.name
    return f"{option.name}: {option.description}"


def adapter_prompt(question: Question, listed_options: int | None) -> str:
    """Build the jevals adapter prompt.

    Args:
        question: The question to ask.
        listed_options: None to request a probability for every option, or N to request
            only the N most likely options (jevals does this above 10 options).
    """
    if listed_options is None:
        what = f"every {_noun(question.primitive)} listed above."
    else:
        what = f"the {listed_options} most likely options only."
    return ADAPTER_TEMPLATE.format(
        state=question.state,
        instructions=question.instructions,
        options_heading=_heading(question.primitive),
        options="\n".join(f"- {_describe(option)}" for option in question.options),
        what=what,
    )


def readout_labels(option_count: int) -> list[str]:
    """Zero-padded numeric labels, all the same width, so no label is a prefix of another."""
    width = len(str(option_count - 1))
    return [str(index).zfill(width) for index in range(option_count)]


def readout_prompt(question: Question) -> str:
    """Build the prompt whose next token is the label of the answer."""
    labels = readout_labels(len(question.options))
    return READOUT_TEMPLATE.format(
        state=question.state,
        instructions=question.instructions,
        options_heading=_heading(question.primitive),
        options="\n".join(f"{label}. {_describe(option)}" for label, option in zip(labels, question.options, strict=True)),
        noun=_noun(question.primitive),
    )
