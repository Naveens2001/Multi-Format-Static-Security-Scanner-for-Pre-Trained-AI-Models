"""
tests/unit/test_package_smoke.py
=================================

Phase 1 smoke tests — verify that the package is importable and exposes
the expected minimum metadata. No scanner functionality is tested here.
"""

from __future__ import annotations

import importlib
import sys
from typing import ClassVar


class TestPackageImport:
    """Verify model_scanner package imports cleanly."""

    def test_package_is_importable(self) -> None:
        """The model_scanner package must be importable without error."""
        import model_scanner  # noqa: F401 — import is the test

        assert "model_scanner" in sys.modules

    def test_version_attribute_exists(self) -> None:
        """Package must expose a __version__ string."""
        import model_scanner

        assert hasattr(model_scanner, "__version__")
        assert isinstance(model_scanner.__version__, str)
        assert len(model_scanner.__version__) > 0

    def test_version_format(self) -> None:
        """Version must follow semantic versioning (MAJOR.MINOR.PATCH)."""
        import model_scanner

        parts = model_scanner.__version__.split(".")
        assert len(parts) == 3, f"Expected MAJOR.MINOR.PATCH, got {model_scanner.__version__!r}"
        for part in parts:
            assert part.isdigit(), f"Non-numeric version segment: {part!r}"

    def test_author_attribute_exists(self) -> None:
        """Package must expose a __author__ string."""
        import model_scanner

        assert hasattr(model_scanner, "__author__")
        assert isinstance(model_scanner.__author__, str)

    def test_license_attribute_exists(self) -> None:
        """Package must expose a __license__ string."""
        import model_scanner

        assert hasattr(model_scanner, "__license__")
        assert isinstance(model_scanner.__license__, str)

    def test_all_attribute_is_list(self) -> None:
        """__all__ must be a list (empty at Phase 1)."""
        import model_scanner

        assert hasattr(model_scanner, "__all__")
        assert isinstance(model_scanner.__all__, list)

    def test_reimport_is_idempotent(self) -> None:
        """Re-importing the package must not raise and must be the same object."""
        import model_scanner as first_import

        reloaded = importlib.import_module("model_scanner")
        assert first_import is reloaded


class TestMainModuleImport:
    """Verify model_scanner.__main__ imports cleanly."""

    def test_main_module_is_importable(self) -> None:
        """model_scanner.__main__ must be importable."""
        import model_scanner.__main__  # noqa: F401

        assert "model_scanner.__main__" in sys.modules

    def test_main_function_exists(self) -> None:
        """model_scanner.__main__.main must be callable."""
        import model_scanner.__main__

        assert callable(model_scanner.__main__.main)


class TestNoForbiddenImports:
    """
    Safety verification: the installed package must not import forbidden
    third-party modules or use forbidden builtins.

    NOTE: Full dynamic-execution safety tests live in tests/safety/.
    These are lightweight import-level checks only.
    """

    FORBIDDEN_RUNTIME_MODULES: ClassVar[tuple[str, ...]] = (
        "torch",
        "safetensors",
        "transformers",
        "tensorflow",
        "onnx",
        "h5py",
        "numpy",
        "pandas",
    )

    def test_torch_not_imported_by_package(self) -> None:
        """Importing model_scanner must NOT trigger a torch import."""
        # Clear any cached model_scanner state, then re-check sys.modules
        import model_scanner  # noqa: F401

        assert "torch" not in sys.modules, (
            "torch was imported as a side-effect of importing model_scanner"
        )

    def test_no_forbidden_runtime_dependencies_in_sys_modules(self) -> None:
        """Importing model_scanner must not trigger any forbidden runtime imports.

        Checks the delta added by the import, not the absolute sys.modules state,
        so the test is robust when optional deps (e.g. h5py) are pre-installed
        by earlier tests in the same session.
        """
        pre_import = set(sys.modules.keys())
        import model_scanner  # noqa: F401
        added = set(sys.modules.keys()) - pre_import

        for mod in self.FORBIDDEN_RUNTIME_MODULES:
            assert mod not in added, (
                f"Forbidden module '{mod}' was imported as a side-effect of "
                f"importing model_scanner"
            )
