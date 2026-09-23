"""Prompts, questions, benchmark question building and configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from openjev.benchmark import build_question
from openjev.config import Config, load_config
from openjev.jevals import SuiteItem, Task
from openjev.prompts import adapter_prompt, readout_labels, readout_prompt
from openjev.questions import Option, Primitive, Question
from openjev.sources import PreparedItem

CONFIG = Path(__file__).resolve().parent.parent / "config" / "openjev.toml"


def question() -> Question:
    return Question("choice", '{"text": "hi"}', "Which intent?", (Option("greeting", "Says hello"), Option("other", None)))


def test_adapter_prompt_lists_every_option() -> None:
    prompt = adapter_prompt(question(), None)
    assert 'STATE (JSON):\n{"text": "hi"}' in prompt
    assert "- greeting: Says hello\n- other" in prompt
    assert "every option listed above" in prompt
    assert prompt.endswith('{"probabilities": {"<option>": <probability between 0 and 1>}}')


def test_adapter_prompt_top_mode() -> None:
    assert "the 5 most likely options only" in adapter_prompt(question(), 5)


def test_readout_labels_share_one_width() -> None:
    assert readout_labels(3) == ["0", "1", "2"]
    assert readout_labels(77)[:2] == ["00", "01"]
    assert readout_labels(77)[-1] == "76"


def test_readout_prompt_numbers_options() -> None:
    prompt = readout_prompt(question())
    assert "0. greeting: Says hello\n1. other" in prompt
    assert prompt.endswith("Reply with only the number of the correct option.")


@pytest.mark.parametrize(
    ("primitive", "names"),
    [("noul", ["yes", "maybe"]), ("choice", ["a"]), ("choice", ["a", "a"])],
)
def test_question_rejects_invalid_options(primitive: Primitive, names: list[str]) -> None:
    with pytest.raises(ValueError):
        Question(primitive, "{}", "Q?", tuple(Option(name, None) for name in names))


def test_choice_options_are_shuffled_per_seed_but_noul_is_not() -> None:
    names = [f"o{i}" for i in range(6)]
    choice = Task("t", "choice", "d", "s", "r", "Q?", tuple(Option(n, None) for n in names), ("text",), {}, ())
    item = PreparedItem(SuiteItem("t-1", 0, "", 0), "{}", True)
    first, seed0 = build_question(choice, item, 0)
    again, _ = build_question(choice, item, 1)
    later, seed1 = build_question(choice, item, 2)
    assert (seed0, seed1) == (0, 1)
    assert first.options == again.options
    assert first.options != later.options
    noul = Task("n", "noul", "d", "s", "r", "Q?", (Option("no", None), Option("yes", None)), ("text",), {}, ())
    assert build_question(noul, item, 3)[0].options == noul.options


def test_config_loads_three_models() -> None:
    config = load_config(CONFIG)
    assert [model.key for model in config.models] == ["qwen3-0.6b", "minicpm5-2b", "qwen3.5-4b"]
    assert config.model("minicpm5-2b").model_id == "minicpm5-2b"
    with pytest.raises(KeyError, match="known keys"):
        config.model("gpt")


def test_config_rejects_missing_and_unknown_fields(tmp_path: Path) -> None:
    text = CONFIG.read_text()
    missing = tmp_path / "missing.toml"
    missing.write_text(text.replace("timeout_seconds = 300\n", ""))
    with pytest.raises(ValidationError, match="timeout_seconds"):
        load_config(missing)
    extra = tmp_path / "extra.toml"
    extra.write_text(text.replace("timeout_seconds = 300\n", "timeout_seconds = 300\nretries = 3\n"))
    with pytest.raises(ValidationError, match="retries"):
        load_config(extra)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.toml")


def test_config_type() -> None:
    assert isinstance(load_config(CONFIG), Config)
