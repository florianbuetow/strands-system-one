"""Architecture test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytestarch import EvaluableArchitecture, get_evaluable_architecture

# Resolve paths relative to this file:
#   tests/architecture/conftest.py -> tests/ -> project root
_TESTS_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _TESTS_DIR.parent
_PACKAGE_DIR = _PROJECT_ROOT / "src" / "llm_system_one"


@pytest.fixture(scope="session")
def evaluable() -> EvaluableArchitecture:
    """Build the evaluable architecture graph for llm-system-one.

    Uses llm_system_one package as both root and module path
    so module names resolve cleanly (e.g. 'llm_system_one.module').
    """
    return get_evaluable_architecture(str(_PACKAGE_DIR), str(_PACKAGE_DIR))
