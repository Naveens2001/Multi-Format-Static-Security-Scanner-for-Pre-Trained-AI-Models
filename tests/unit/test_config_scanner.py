"""
tests.unit.test_config_scanner
================================

Unit tests for model_scanner.analyzers.repository.config_scanner.
Covers SC101/SC102 detection paths including auto_map, trust_remote_code,
custom_pipelines, and dotted-path walk.
"""

from __future__ import annotations

import json
from pathlib import Path

from model_scanner.analyzers.repository.config_scanner import (
    _check_for_referenced_py,
    _extract_auto_map_module,
    _read_json_bounded,
    _scan_value_for_dotted_path,
    _walk_for_dotted_paths,
    scan_config_file,
)

# ---------------------------------------------------------------------------
# _read_json_bounded
# ---------------------------------------------------------------------------


class TestReadJsonBounded:
    def test_reads_valid_object(self, tmp_path: Path) -> None:
        f = tmp_path / "config.json"
        f.write_text('{"model_type": "llama"}', encoding="utf-8")
        obj, err = _read_json_bounded(f)
        assert obj == {"model_type": "llama"}
        assert err is None

    def test_returns_error_for_array(self, tmp_path: Path) -> None:
        f = tmp_path / "config.json"
        f.write_text("[1, 2, 3]", encoding="utf-8")
        obj, err = _read_json_bounded(f)
        assert obj is None
        assert err is not None

    def test_returns_error_for_invalid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "config.json"
        f.write_bytes(b"not json {{{")
        obj, err = _read_json_bounded(f)
        assert obj is None
        assert err is not None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        obj, err = _read_json_bounded(tmp_path / "nonexistent.json")
        assert obj is None
        assert err is not None


# ---------------------------------------------------------------------------
# _extract_auto_map_module
# ---------------------------------------------------------------------------


class TestExtractAutoMapModule:
    def test_simple_module_class(self) -> None:
        result = _extract_auto_map_module("modeling_custom.MyModel")
        assert result == "modeling_custom"

    def test_dotted_module(self) -> None:
        result = _extract_auto_map_module("pkg.subpkg.MyClass")
        assert result == "pkg.subpkg"

    def test_strips_leading_slashes(self) -> None:
        result = _extract_auto_map_module("./modeling_custom.MyModel")
        assert result == "modeling_custom"

    def test_no_dot_returns_none(self) -> None:
        result = _extract_auto_map_module("NoClass")
        assert result is None


# ---------------------------------------------------------------------------
# _check_for_referenced_py
# ---------------------------------------------------------------------------


class TestCheckForReferencedPy:
    def test_finds_matching_py_file(self, tmp_path: Path) -> None:
        py_file = tmp_path / "modeling_custom.py"
        py_file.write_text("# custom", encoding="utf-8")
        result = _check_for_referenced_py("modeling_custom", tmp_path)
        assert result == "modeling_custom.py"

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        result = _check_for_referenced_py("nonexistent_module", tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# _scan_value_for_dotted_path
# ---------------------------------------------------------------------------


class TestScanValueForDottedPath:
    def test_dotted_string_matches(self) -> None:
        result = _scan_value_for_dotted_path("module.ClassName", "field")
        assert result == "module.ClassName"

    def test_plain_string_no_match(self) -> None:
        result = _scan_value_for_dotted_path("llama", "model_type")
        assert result is None

    def test_non_string_no_match(self) -> None:
        result = _scan_value_for_dotted_path(42, "field")
        assert result is None


# ---------------------------------------------------------------------------
# _walk_for_dotted_paths
# ---------------------------------------------------------------------------


class TestWalkForDottedPaths:
    def test_detects_dotted_value(self) -> None:
        findings: list = []
        _walk_for_dotted_paths(
            {"processor": "custom.MyProcessor"},
            "root",
            findings,
            "config.json",
        )
        assert len(findings) >= 1

    def test_nested_dict_traversal(self) -> None:
        findings: list = []
        _walk_for_dotted_paths(
            {"outer": {"inner": "pkg.ClassName"}},
            "root",
            findings,
            "config.json",
        )
        assert len(findings) >= 1

    def test_list_traversal(self) -> None:
        findings: list = []
        _walk_for_dotted_paths(
            ["pkg.ClassName", "other"],
            "root",
            findings,
            "config.json",
        )
        assert len(findings) >= 1

    def test_depth_limit_prevents_overflow(self) -> None:
        # Deeply nested structure — should not recurse infinitely
        findings: list = []
        nested: dict = {}
        d = nested
        for _ in range(20):
            d["child"] = {}
            d = d["child"]
        d["val"] = "pkg.Class"
        _walk_for_dotted_paths(nested, "root", findings, "config.json", depth=0)
        # Just verify it doesn't crash; may or may not find the deep value
        assert isinstance(findings, list)


# ---------------------------------------------------------------------------
# scan_config_file — integration
# ---------------------------------------------------------------------------


class TestScanConfigFile:
    def test_auto_map_emits_sc101(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({
                "auto_map": {
                    "AutoModelForCausalLM": "modeling_custom.CustomModel"
                }
            }),
            encoding="utf-8",
        )
        findings = scan_config_file(cfg, tmp_path)
        rule_ids = {f.rule_id for f in findings}
        assert "SC101" in rule_ids

    def test_trust_remote_code_true_emits_sc102(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"trust_remote_code": True}),
            encoding="utf-8",
        )
        findings = scan_config_file(cfg, tmp_path)
        rule_ids = {f.rule_id for f in findings}
        assert "SC102" in rule_ids

    def test_trust_remote_code_false_no_sc102(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"trust_remote_code": False}),
            encoding="utf-8",
        )
        findings = scan_config_file(cfg, tmp_path)
        sc102 = [f for f in findings if f.rule_id == "SC102"]
        assert sc102 == []

    def test_custom_pipelines_emits_sc101(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({
                "custom_pipelines": {
                    "text-generation": {"impl": "my_pkg.MyPipeline"}
                }
            }),
            encoding="utf-8",
        )
        findings = scan_config_file(cfg, tmp_path)
        rule_ids = {f.rule_id for f in findings}
        assert "SC101" in rule_ids

    def test_clean_config_no_findings(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"model_type": "llama", "hidden_size": 4096}),
            encoding="utf-8",
        )
        findings = scan_config_file(cfg, tmp_path)
        assert findings == []

    def test_unreadable_file_returns_empty(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"
        findings = scan_config_file(missing, tmp_path)
        assert findings == []

    def test_dotted_path_in_other_field_detected(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"processor_class": "custom_module.Processor"}),
            encoding="utf-8",
        )
        findings = scan_config_file(cfg, tmp_path)
        sc101 = [f for f in findings if f.rule_id == "SC101"]
        assert len(sc101) >= 1

    def test_auto_map_with_referenced_py_file(self, tmp_path: Path) -> None:
        py_file = tmp_path / "modeling_custom.py"
        py_file.write_text("# custom", encoding="utf-8")
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({
                "auto_map": {"AutoModel": "modeling_custom.MyModel"}
            }),
            encoding="utf-8",
        )
        findings = scan_config_file(cfg, tmp_path)
        sc101 = [f for f in findings if f.rule_id == "SC101"]
        assert len(sc101) >= 1
        assert sc101[0].referenced_py_file is not None
