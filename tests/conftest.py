"""
tests/conftest.py
=================

Pytest shared fixtures, configuration, and test infrastructure for the
Static Security Scanner test suite.

Phase 1 — Foundation: basic fixtures only.
Scanner-specific fixtures are added per phase.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the src layout is importable when running pytest from the project root
# without installing the package.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def src_dir() -> Path:
    """Return the absolute path to the src/ directory."""
    return _SRC_DIR


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the absolute path to the fixtures/ directory."""
    return _PROJECT_ROOT / "fixtures"


# ---------------------------------------------------------------------------
# Function-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_model_dir(tmp_path: Path) -> Path:
    """Return a fresh temporary directory for test model artifacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
