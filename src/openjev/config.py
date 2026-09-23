"""Typed configuration loaded from a TOML file. Every field is required."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Method = Literal["readout", "verbalized"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConfig(_Strict):
    """OpenAI-compatible endpoint that serves the models."""

    base_url: str
    api_key: str
    timeout_seconds: float = Field(gt=0)


class ModelConfig(_Strict):
    """One model served by the provider."""

    key: str
    model_id: str
    display_name: str
    assistant_prefill: str


class ReadoutConfig(_Strict):
    """Settings for the logprob readout (System One emulation)."""

    top_logprobs: int = Field(ge=2)
    temperature: float = Field(ge=0)


class VerbalizedConfig(_Strict):
    """Settings for the JSON-writing LLM adapter."""

    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    max_tokens: int = Field(gt=0)
    malformed_retries: int = Field(ge=0)
    full_listing_max_options: int = Field(ge=2)
    top_listed_options: int = Field(ge=1)


class BenchmarkConfig(_Strict):
    """Pinned jevals release and run settings."""

    jevals_repository: str
    jevals_commit: str
    release: str
    suite: str
    tasks: list[str]
    methods: list[Method]
    epochs: int = Field(ge=1)
    reference_systems: list[str]
    input_dir: Path
    output_dir: Path
    report_dir: Path
    bootstrap_resamples: int = Field(ge=100)
    bootstrap_seed: int
    status_interval_seconds: float = Field(gt=0)


class SourceConfig(_Strict):
    """Where a task's item text lives inside its pinned Hugging Face dataset."""

    file: str
    label_field: str
    label_is_index: bool


class Config(_Strict):
    """Complete openjev configuration."""

    provider: ProviderConfig
    models: list[ModelConfig]
    readout: ReadoutConfig
    verbalized: VerbalizedConfig
    benchmark: BenchmarkConfig
    sources: dict[str, SourceConfig]

    def model(self, key: str) -> ModelConfig:
        """Return the model with the given key.

        Raises:
            KeyError: If no model has that key.
        """
        for model in self.models:
            if model.key == key:
                return model
        raise KeyError(f"No model with key {key!r} in config; known keys: {[m.key for m in self.models]}")


def load_config(path: Path) -> Config:
    """Load and validate the configuration file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as handle:
        return Config.model_validate(tomllib.load(handle))
