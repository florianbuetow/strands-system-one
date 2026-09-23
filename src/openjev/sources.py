"""Item text from the public Hugging Face datasets, pinned to the revision the jevals suite names."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from openjev.config import SourceConfig
from openjev.jevals import SuiteItem, Task


@dataclass(frozen=True)
class PreparedItem:
    """A suite item with its state text."""

    item: SuiteItem
    state: str
    hash_matches: bool


def source_path(task: Task, source: SourceConfig, input_dir: Path) -> Path:
    """Download (once) the task's source file at the pinned revision and return its local path."""
    path = input_dir / "sources" / task.dataset.replace("/", "__") / task.hf_revision / source.file
    if path.exists():
        return path
    url = f"https://huggingface.co/datasets/{task.dataset}/resolve/{task.hf_revision}/{source.file}"
    response = httpx.get(url, follow_redirects=True, timeout=300)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pl.read_parquet(path).to_dicts()
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError(f"Unsupported source file type: {path}")


def _state(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Whitelisted fields only; a dotted field keeps its nesting ("context.contexts")."""
    state: dict[str, Any] = {}
    for field in fields:
        value: Any = row
        target = state
        parts = field.split(".")
        for part in parts:
            value = value[part]
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return state


def _label_index(task: Task, row: dict[str, Any], source: SourceConfig) -> int:
    value = row[source.label_field]
    if source.label_is_index:
        return int(value)
    return task.option_names.index(str(value))


def prepare_items(task: Task, source: SourceConfig, input_dir: Path) -> list[PreparedItem]:
    """Build every item's state and check its label against the suite.

    Raises:
        ValueError: If any source row's label differs from the suite's target.
    """
    rows = _rows(source_path(task, source, input_dir))
    prepared: list[PreparedItem] = []
    for item in task.items:
        row = rows[item.row_idx]
        label = _label_index(task, row, source)
        if label != item.target:
            raise ValueError(f"{item.item_id}: source label {label} differs from suite target {item.target}")
        state = json.dumps(_state(row, task.state_fields), ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(state.encode("utf-8")).hexdigest()
        prepared.append(PreparedItem(item=item, state=state, hash_matches=digest == item.state_sha256))
    return prepared
