"""The jevals benchmark (https://github.com/Jevals/jevals-data, CC BY 4.0): suites, boards and run logs.

Attribution: Jevals (jevals.com), release 2026-09-18, suite 0.1.0. CC-BY-4.0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx

from openjev.config import BenchmarkConfig
from openjev.questions import Option, Primitive


@dataclass(frozen=True)
class SuiteItem:
    """One benchmark item: where its text lives upstream and the index of its true label."""

    item_id: str
    row_idx: int
    state_sha256: str
    target: int


@dataclass(frozen=True)
class Task:
    """One jevals task (one per primitive in suite 0.1.0)."""

    id: str
    primitive: Primitive
    dataset: str
    split: str
    hf_revision: str
    instructions: str
    options: tuple[Option, ...]
    state_fields: tuple[str, ...]
    class_counts: dict[str, int]
    items: tuple[SuiteItem, ...]

    @property
    def option_names(self) -> list[str]:
        """Option names in the task file's order (targets index into this list)."""
        return [option.name for option in self.options]


def jevals_dir(config: BenchmarkConfig) -> Path:
    """Local copy of the pinned jevals-data commit."""
    return config.input_dir / "jevals" / config.jevals_commit


def _files(config: BenchmarkConfig) -> list[str]:
    files = [f"suites/{config.suite}/{task}.json" for task in config.tasks]
    files.append(f"releases/{config.release}/board.json")
    files.extend(f"runs/{system}__{task}__{config.suite}.jsonl" for system in config.reference_systems for task in config.tasks)
    files.extend(["LICENSE", "NOTICE"])
    return files


def download_jevals(config: BenchmarkConfig) -> list[Path]:
    """Download the pinned suite files, board and reference run logs; skip files already present."""
    root = jevals_dir(config)
    paths: list[Path] = []
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for name in _files(config):
            path = root / name
            paths.append(path)
            if path.exists():
                continue
            url = f"https://raw.githubusercontent.com/{config.jevals_repository}/{config.jevals_commit}/{name}"
            response = client.get(url)
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
    return paths


def _descriptions(primitive: Primitive, options: list[str], criteria: object) -> list[str | None]:
    if isinstance(criteria, list):
        levels = cast(list[str], criteria)
        if len(levels) != len(options):
            raise ValueError(f"{len(levels)} criteria for {len(options)} options")
        return list(levels)
    if not isinstance(criteria, dict):
        raise ValueError(f"Unsupported criteria format: {type(criteria).__name__}")
    by_key = cast(dict[str, str | None], criteria)
    if primitive == "noul":
        return [by_key["true" if name == "yes" else "false"] for name in options]
    return [by_key[name] for name in options]


def load_task(config: BenchmarkConfig, task_id: str) -> Task:
    """Load a task's suite file from the local jevals copy."""
    path = jevals_dir(config) / "suites" / config.suite / f"{task_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Suite file missing: {path} (run the fetch step first)")
    data: dict[str, Any] = json.loads(path.read_text())
    primitive: Primitive = data["primitive"]
    names: list[str] = data["options"]
    descriptions = _descriptions(primitive, names, data["criteria"])
    return Task(
        id=data["id"],
        primitive=primitive,
        dataset=data["dataset"],
        split=data["split"],
        hf_revision=data["hf_revision"],
        instructions=data["instructions"],
        options=tuple(Option(name, description) for name, description in zip(names, descriptions, strict=True)),
        state_fields=tuple(data["state_fields"]),
        class_counts=dict(data["class_counts"]),
        items=tuple(
            SuiteItem(item_id=item["item_id"], row_idx=item["row_idx"], state_sha256=item["state_sha256"], target=item["target"])
            for item in data["items"]
        ),
    )


def load_board(config: BenchmarkConfig) -> list[dict[str, Any]]:
    """Rows of the published board for the pinned release."""
    path = jevals_dir(config) / "releases" / config.release / "board.json"
    if not path.is_file():
        raise FileNotFoundError(f"Board missing: {path} (run the fetch step first)")
    rows: list[dict[str, Any]] = json.loads(path.read_text())["rows"]
    return rows


def reference_run_path(config: BenchmarkConfig, system: str, task_id: str) -> Path:
    """Local path of a published jevals run log."""
    return jevals_dir(config) / "runs" / f"{system}__{task_id}__{config.suite}.jsonl"
