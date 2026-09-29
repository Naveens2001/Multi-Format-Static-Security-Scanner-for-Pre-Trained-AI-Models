"""
tests.unit.test_pickle_catalog
==============================

Unit tests for GlobalCategory and classify_global.
"""

from __future__ import annotations

from model_scanner.analyzers.pickle.catalog import (
    GlobalCategory,
    classify_global,
)


class TestPickleCatalog:
    """Test static global classification and allowlist/blocklist evaluation."""

    def test_dangerous_globals_classified(self) -> None:
        dangerous_samples = [
            ("os", "system"),
            ("subprocess", "Popen"),
            ("subprocess", "run"),
            ("builtins", "eval"),
            ("builtins", "exec"),
            ("posix", "system"),
            ("nt", "system"),
            ("pty", "spawn"),
            ("socket", "socket"),
            ("urllib.request", "urlopen"),
            ("requests", "get"),
        ]
        for mod, name in dangerous_samples:
            cat, rationale = classify_global(mod, name)
            assert cat == GlobalCategory.DANGEROUS, f"Expected {mod}.{name} to be DANGEROUS"
            assert "dangerous" in rationale.lower()

    def test_suspicious_globals_classified(self) -> None:
        suspicious_samples = [
            ("importlib", "import_module"),
            ("sys", "modules"),
            ("builtins", "getattr"),
            ("types", "FunctionType"),
            ("ctypes", "CDLL"),
            ("shutil", "rmtree"),
        ]
        for mod, name in suspicious_samples:
            cat, _ = classify_global(mod, name)
            assert cat == GlobalCategory.SUSPICIOUS, f"Expected {mod}.{name} to be SUSPICIOUS"

    def test_safe_globals_classified(self) -> None:
        safe_samples = [
            ("torch._utils", "_rebuild_tensor"),
            ("torch._utils", "_rebuild_tensor_v2"),
            ("torch", "FloatStorage"),
            ("collections", "OrderedDict"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"),
            ("builtins", "dict"),
            ("builtins", "int"),
        ]
        for mod, name in safe_samples:
            cat, _ = classify_global(mod, name)
            assert cat == GlobalCategory.SAFE, f"Expected {mod}.{name} to be SAFE"

    def test_custom_blocklist_overrides(self) -> None:
        # Normally safe module blocked by user
        cat, rationale = classify_global(
            "collections", "OrderedDict", blocklist=["collections"]
        )
        assert cat == GlobalCategory.DANGEROUS
        assert "blocklist" in rationale

    def test_custom_allowlist_overrides(self) -> None:
        # Unclassified module allowed by user
        cat, rationale = classify_global(
            "my_custom_package", "ModelClass", allowlist=["my_custom_package"]
        )
        assert cat == GlobalCategory.SAFE
        assert "allowlist" in rationale

    def test_unclassified_global_is_unknown(self) -> None:
        cat, rationale = classify_global("unknown_pkg", "SpecialLayer")
        assert cat == GlobalCategory.UNKNOWN
        assert "Unclassified" in rationale
