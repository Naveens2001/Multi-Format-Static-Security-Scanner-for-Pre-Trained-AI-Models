"""
tests.unit.test_provenance
============================

Unit tests for model_scanner.analyzers.repository.provenance.
Covers SC106, SC107, SC108, SC109 paths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from model_scanner.analyzers.repository.provenance import (
    check_config_format_consistency,
    check_index_hash_consistency,
    check_lfs_pointer_stubs,
    check_signature_artifacts,
)

# ---------------------------------------------------------------------------
# SC106 — Signature artifact presence/absence
# ---------------------------------------------------------------------------


class TestSC106SignatureArtifacts:
    def test_unsigned_weight_emits_sc106(self, tmp_path: Path) -> None:
        wf = tmp_path / "model.safetensors"
        wf.write_bytes(b"FAKE")
        findings = check_signature_artifacts(tmp_path, [wf])
        rule_ids = {f.rule_id for f in findings}
        assert "SC106" in rule_ids

    def test_signed_weight_no_sc106(self, tmp_path: Path) -> None:
        wf = tmp_path / "model.safetensors"
        wf.write_bytes(b"FAKE")
        sig = tmp_path / "model.safetensors.sig"
        sig.write_bytes(b"SIGNATURE")
        findings = check_signature_artifacts(tmp_path, [wf])
        sc106 = [f for f in findings if f.rule_id == "SC106"]
        assert sc106 == []

    def test_repo_level_sig_suppresses_sc106(self, tmp_path: Path) -> None:
        wf = tmp_path / "model.safetensors"
        wf.write_bytes(b"FAKE")
        # checksums.txt is a recognized repo-level signature file
        (tmp_path / "checksums.txt").write_bytes(b"abc123  model.safetensors\n")
        findings = check_signature_artifacts(tmp_path, [wf])
        sc106 = [f for f in findings if f.rule_id == "SC106"]
        assert sc106 == []

    def test_empty_weight_list_no_findings(self, tmp_path: Path) -> None:
        findings = check_signature_artifacts(tmp_path, [])
        assert findings == []

    def test_weight_outside_repo_root_fallback(self, tmp_path: Path) -> None:
        # weight file not under repo_root — triggers ValueError in relative_to
        import tempfile
        with tempfile.TemporaryDirectory() as other_dir:
            wf = Path(other_dir) / "model.safetensors"
            wf.write_bytes(b"FAKE")
            findings = check_signature_artifacts(tmp_path, [wf])
        # SC106 should still be emitted using absolute path as fallback
        assert any(f.rule_id == "SC106" for f in findings)

    def test_sigstore_ext_recognized(self, tmp_path: Path) -> None:
        wf = tmp_path / "model.safetensors"
        wf.write_bytes(b"FAKE")
        sig = tmp_path / "model.safetensors.sigstore"
        sig.write_bytes(b"SIGSTORE")
        findings = check_signature_artifacts(tmp_path, [wf])
        sc106 = [f for f in findings if f.rule_id == "SC106"]
        assert sc106 == []


# ---------------------------------------------------------------------------
# SC107 — Hash self-consistency
# ---------------------------------------------------------------------------


class TestSC107IndexHashConsistency:
    def test_sc107_fires_on_mismatch(self, tmp_path: Path) -> None:
        wf = tmp_path / "model-00001-of-00001.safetensors"
        wf.write_bytes(b"REAL_CONTENT")
        bad_hash = "a" * 64  # deliberately wrong hash

        index = {"checksums": {"model-00001-of-00001.safetensors": bad_hash}}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        findings = check_index_hash_consistency(tmp_path, [wf])
        rule_ids = {f.rule_id for f in findings}
        assert "SC107" in rule_ids

    def test_no_sc107_when_hashes_match(self, tmp_path: Path) -> None:
        content = b"REAL_CONTENT"
        wf = tmp_path / "model-00001-of-00001.safetensors"
        wf.write_bytes(content)
        actual_hash = hashlib.sha256(content).hexdigest()

        index = {"checksums": {"model-00001-of-00001.safetensors": actual_hash}}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        findings = check_index_hash_consistency(tmp_path, [wf])
        sc107 = [f for f in findings if f.rule_id == "SC107"]
        assert sc107 == []

    def test_no_index_returns_empty(self, tmp_path: Path) -> None:
        wf = tmp_path / "model.safetensors"
        wf.write_bytes(b"FAKE")
        findings = check_index_hash_consistency(tmp_path, [wf])
        assert findings == []

    def test_index_without_hash_section_returns_empty(self, tmp_path: Path) -> None:
        wf = tmp_path / "model.safetensors"
        wf.write_bytes(b"FAKE")
        index = {"weight_map": {"layer.weight": "model.safetensors"}}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        findings = check_index_hash_consistency(tmp_path, [wf])
        sc107 = [f for f in findings if f.rule_id == "SC107"]
        assert sc107 == []

    def test_sha256_key_also_recognized(self, tmp_path: Path) -> None:
        content = b"CONTENT"
        wf = tmp_path / "model.safetensors"
        wf.write_bytes(content)
        bad_hash = "b" * 64

        index = {"sha256": {"model.safetensors": bad_hash}}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        findings = check_index_hash_consistency(tmp_path, [wf])
        assert any(f.rule_id == "SC107" for f in findings)

    def test_invalid_json_index_skipped(self, tmp_path: Path) -> None:
        wf = tmp_path / "model.safetensors"
        wf.write_bytes(b"X")
        (tmp_path / "model.safetensors.index.json").write_bytes(b"not json{{")
        findings = check_index_hash_consistency(tmp_path, [wf])
        assert findings == []


# ---------------------------------------------------------------------------
# SC108 — LFS pointer stubs
# ---------------------------------------------------------------------------

_LFS_STUB = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890\n"
    b"size 1234\n"
)


class TestSC108LFSPointerStubs:
    def test_lfs_pointer_stub_emits_sc108(self, tmp_path: Path) -> None:
        wf = tmp_path / "model.safetensors"
        wf.write_bytes(_LFS_STUB)
        findings = check_lfs_pointer_stubs(tmp_path, [wf])
        assert any(f.rule_id == "SC108" for f in findings)

    def test_real_binary_not_sc108(self, tmp_path: Path) -> None:
        wf = tmp_path / "model.safetensors"
        wf.write_bytes(b"\x00\x01\x02\x03" * 1000)
        findings = check_lfs_pointer_stubs(tmp_path, [wf])
        assert findings == []

    def test_large_file_skipped_for_lfs_check(self, tmp_path: Path) -> None:
        wf = tmp_path / "model.safetensors"
        # Write LFS preamble but make it exceed the pointer max size
        wf.write_bytes(_LFS_STUB + b"X" * 5000)
        findings = check_lfs_pointer_stubs(tmp_path, [wf])
        # File is too large for pointer check → should not flag SC108
        assert findings == []

    def test_gitattributes_lfs_pattern_used(self, tmp_path: Path) -> None:
        wf = tmp_path / "model.safetensors"
        wf.write_bytes(_LFS_STUB)
        (tmp_path / ".gitattributes").write_text(
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
            encoding="utf-8",
        )
        findings = check_lfs_pointer_stubs(tmp_path, [wf])
        assert any(f.rule_id == "SC108" for f in findings)


# ---------------------------------------------------------------------------
# SC109 — Declared library vs detected format
# ---------------------------------------------------------------------------


class TestSC109ConfigFormatConsistency:
    def test_sc109_transformers_with_gguf(self) -> None:
        findings = check_config_format_consistency(
            declared_model_type="llama",
            declared_library="transformers",
            detected_formats={"GGUF"},
        )
        assert any(f.rule_id == "SC109" for f in findings)

    def test_sc109_gguf_library_with_safetensors(self) -> None:
        findings = check_config_format_consistency(
            declared_model_type=None,
            declared_library="gguf",
            detected_formats={"SAFETENSORS"},
        )
        assert any(f.rule_id == "SC109" for f in findings)

    def test_no_sc109_when_consistent(self) -> None:
        findings = check_config_format_consistency(
            declared_model_type="llama",
            declared_library="transformers",
            detected_formats={"SAFETENSORS"},
        )
        sc109 = [f for f in findings if f.rule_id == "SC109"]
        assert sc109 == []

    def test_no_sc109_when_library_is_none(self) -> None:
        findings = check_config_format_consistency(
            declared_model_type=None,
            declared_library=None,
            detected_formats={"GGUF"},
        )
        assert findings == []

    def test_no_sc109_when_formats_empty(self) -> None:
        findings = check_config_format_consistency(
            declared_model_type="llama",
            declared_library="transformers",
            detected_formats=set(),
        )
        assert findings == []

    def test_sc109_llamacpp_library_with_safetensors(self) -> None:
        findings = check_config_format_consistency(
            declared_model_type=None,
            declared_library="llamacpp",
            detected_formats={"SAFETENSORS"},
        )
        assert any(f.rule_id == "SC109" for f in findings)
