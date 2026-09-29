"""
tests.unit.test_shard_checker
================================

Unit tests for model_scanner.analyzers.repository.shard_checker, covering
G005, G006, G007, and G008 paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from model_scanner.analyzers.repository.shard_checker import (
    check_shard_integrity,
    group_shard_files,
)


def _make_weight(tmp_path: Path, name: str, content: bytes = b"FAKE") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# group_shard_files
# ---------------------------------------------------------------------------


class TestGroupShardFiles:
    def test_basic_grouping(self, tmp_path: Path) -> None:
        files = [
            _make_weight(tmp_path, "model-00001-of-00002.safetensors"),
            _make_weight(tmp_path, "model-00002-of-00002.safetensors"),
        ]
        groups = group_shard_files(files)
        assert len(groups) == 1
        assert groups[0].claimed_total == 2
        assert groups[0].base_name == "model"

    def test_non_shard_files_ignored(self, tmp_path: Path) -> None:
        files = [_make_weight(tmp_path, "model.safetensors")]
        groups = group_shard_files(files)
        assert groups == []

    def test_inconsistent_totals_gives_zero(self, tmp_path: Path) -> None:
        files = [
            _make_weight(tmp_path, "model-00001-of-00002.safetensors"),
            _make_weight(tmp_path, "model-00002-of-00003.safetensors"),
        ]
        groups = group_shard_files(files)
        assert len(groups) == 1
        assert groups[0].claimed_total == 0


# ---------------------------------------------------------------------------
# check_shard_integrity — G005 (missing declared shard)
# ---------------------------------------------------------------------------


class TestG005MissingShard:
    def test_g005_fires_for_missing_shard(self, tmp_path: Path) -> None:
        index = {
            "weight_map": {
                "layer.weight": "model-00001-of-00002.safetensors",
                "other.weight": "model-00002-of-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        # Only shard 1 is present
        weight_files = [_make_weight(tmp_path, "model-00001-of-00002.safetensors")]
        findings = check_shard_integrity(tmp_path, weight_files)
        rule_ids = {f.rule_id for f in findings}
        assert "G005" in rule_ids


# ---------------------------------------------------------------------------
# check_shard_integrity — G006 (undeclared shard on disk)
# ---------------------------------------------------------------------------


class TestG006UndeclaredShard:
    def test_g006_fires_for_extra_shard(self, tmp_path: Path) -> None:
        index = {
            "weight_map": {
                "layer.weight": "model-00001-of-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        weight_files = [
            _make_weight(tmp_path, "model-00001-of-00002.safetensors"),
            _make_weight(tmp_path, "model-00002-of-00002.safetensors"),
        ]
        findings = check_shard_integrity(tmp_path, weight_files)
        rule_ids = {f.rule_id for f in findings}
        assert "G006" in rule_ids


# ---------------------------------------------------------------------------
# check_shard_integrity — G007 (size mismatch)
# ---------------------------------------------------------------------------


class TestG007SizeMismatch:
    def test_g007_fires_on_size_mismatch(self, tmp_path: Path) -> None:
        content = b"X" * 100  # actual size = 100 bytes
        shard = _make_weight(tmp_path, "model-00001-of-00001.safetensors", content)
        index = {
            "weight_map": {
                "layer.weight": "model-00001-of-00001.safetensors",
            },
            "file_sizes": {
                "model-00001-of-00001.safetensors": 999,  # declared != actual
            },
        }
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        findings = check_shard_integrity(tmp_path, [shard])
        rule_ids = {f.rule_id for f in findings}
        assert "G007" in rule_ids

    def test_g007_no_finding_when_sizes_match(self, tmp_path: Path) -> None:
        content = b"X" * 100
        shard = _make_weight(tmp_path, "model-00001-of-00001.safetensors", content)
        index = {
            "weight_map": {
                "layer.weight": "model-00001-of-00001.safetensors",
            },
            "file_sizes": {
                "model-00001-of-00001.safetensors": 100,  # matches
            },
        }
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        findings = check_shard_integrity(tmp_path, [shard])
        g007 = [f for f in findings if f.rule_id == "G007"]
        assert g007 == []

    def test_g007_uses_shard_sizes_key(self, tmp_path: Path) -> None:
        content = b"Y" * 50
        shard = _make_weight(tmp_path, "model-00001-of-00001.safetensors", content)
        index = {
            "weight_map": {"w": "model-00001-of-00001.safetensors"},
            "shard_sizes": {"model-00001-of-00001.safetensors": 200},
        }
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        findings = check_shard_integrity(tmp_path, [shard])
        assert any(f.rule_id == "G007" for f in findings)


# ---------------------------------------------------------------------------
# check_shard_integrity — G008 (filename count mismatch)
# ---------------------------------------------------------------------------


class TestG008CountMismatch:
    def test_g008_fires_when_count_wrong(self, tmp_path: Path) -> None:
        # 3-of-5 means 5 expected but only 1 file
        shard = _make_weight(tmp_path, "model-00003-of-00005.safetensors")
        findings = check_shard_integrity(tmp_path, [shard])
        rule_ids = {f.rule_id for f in findings}
        assert "G008" in rule_ids

    def test_g008_fires_for_inconsistent_totals(self, tmp_path: Path) -> None:
        # Shards claim different totals — group_shard_files gives claimed_total=0
        shards = [
            _make_weight(tmp_path, "model-00001-of-00002.safetensors"),
            _make_weight(tmp_path, "model-00002-of-00003.safetensors"),
        ]
        findings = check_shard_integrity(tmp_path, shards)
        rule_ids = {f.rule_id for f in findings}
        assert "G008" in rule_ids

    def test_g008_no_finding_when_complete(self, tmp_path: Path) -> None:
        shards = [
            _make_weight(tmp_path, "model-00001-of-00002.safetensors"),
            _make_weight(tmp_path, "model-00002-of-00002.safetensors"),
        ]
        findings = check_shard_integrity(tmp_path, shards)
        g008 = [f for f in findings if f.rule_id == "G008"]
        assert g008 == []

    def test_no_findings_without_weight_files(self, tmp_path: Path) -> None:
        findings = check_shard_integrity(tmp_path, [])
        assert findings == []

    def test_invalid_index_json_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors.index.json").write_bytes(b"not json {{{")
        shard = _make_weight(tmp_path, "model.safetensors")
        findings = check_shard_integrity(tmp_path, [shard])
        # No crash, possibly no findings (or G008)
        g005 = [f for f in findings if f.rule_id == "G005"]
        assert g005 == []
