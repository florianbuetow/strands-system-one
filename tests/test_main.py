"""Command line interface."""

from pathlib import Path

import pytest

from src.main import main


def test_requires_config_and_command() -> None:
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["--config", "config/openjev.toml"])


def test_missing_config_file_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["--config", str(tmp_path / "absent.toml"), "report"])
