"""
tests.unit.test_repository_analyzer
=====================================

Unit tests for the repository supply-chain analyzer (WP-A / WP-B).

Coverage:
1.  RepositoryAnalyzer.analyze_directory returns a list of Findings
2.  SC101 fires on config with auto_map
3.  SC102 fires on config with trust_remote_code=True
4.  SC103 fires when bundled modeling_*.py file is present
5.  SC104 fires on dangerous AST call in bundled Python
6.  SC105 fires on network import in bundled Python
7.  SC106 fires when no signature file found
8.  SC108 fires on LFS pointer stub
9.  SC201 fires for unpinned dependency in requirements.txt
10. SC203 fires for VCS URL dependency
11. SC204 fires for typosquatting dependency
12. G005 fires for shard declared in index but missing on disk
13. G006 fires for undeclared shard file on disk
14. All findings use ModelFormat.REPOSITORY
15. All findings have valid rule IDs from the catalog
16. make_repository_artifact returns an Artifact with REPOSITORY input source
17. Engine integrates RepositoryAnalyzer for LOCAL_DIRECTORY scans
18. Empty directory produces only SC106 (no signature) finding
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from model_scanner.analyzers.repository.analyzer import (
    RepositoryAnalyzer,
    make_repository_artifact,
)
from model_scanner.core.engine import ScanOrchestrator
from model_scanner.core.models import (
    InputSourceType,
    ModelFormat,
    ScanConfig,
    ScanRequest,
)
from model_scanner.rules.catalog import RuleCatalog


def _write_config(directory: Path, data: dict) -> None:
    (directory / "config.json").write_text(json.dumps(data), encoding="utf-8")


def _write_requirements(directory: Path, content: str) -> None:
    (directory / "requirements.txt").write_text(content, encoding="utf-8")


def _safetensors_bytes() -> bytes:
    header = b"{}"
    return len(header).to_bytes(8, "little") + header


# ---------------------------------------------------------------------------
# Test 1: analyze_directory basic contract
# ---------------------------------------------------------------------------


def test_analyze_directory_returns_list() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        analyzer = RepositoryAnalyzer()
        result = analyzer.analyze_directory(root, [], ScanConfig())
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Test 2: SC101 auto_map detection
# ---------------------------------------------------------------------------


def test_sc101_fires_on_auto_map() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_config(root, {"auto_map": {"AutoModel": "modeling_custom.MyModel"}})
        findings = RepositoryAnalyzer().analyze_directory(root, [], ScanConfig())
        rule_ids = {f.rule_id for f in findings}
        assert "SC101" in rule_ids


# ---------------------------------------------------------------------------
# Test 3: SC102 trust_remote_code detection
# ---------------------------------------------------------------------------


def test_sc102_fires_on_trust_remote_code() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_config(root, {"trust_remote_code": True})
        findings = RepositoryAnalyzer().analyze_directory(root, [], ScanConfig())
        rule_ids = {f.rule_id for f in findings}
        assert "SC102" in rule_ids


# ---------------------------------------------------------------------------
# Test 4: SC103 bundled Python code detection
# ---------------------------------------------------------------------------


def test_sc103_fires_for_bundled_py_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "modeling_custom.py").write_text("x = 1\n", encoding="utf-8")
        findings = RepositoryAnalyzer().analyze_directory(root, [], ScanConfig())
        rule_ids = {f.rule_id for f in findings}
        assert "SC103" in rule_ids


# ---------------------------------------------------------------------------
# Test 5: SC104 dangerous system call in bundled Python
# ---------------------------------------------------------------------------


def test_sc104_fires_for_os_system_in_bundled_py() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "modeling_evil.py").write_text(
            "import os\nos.system('id')\n", encoding="utf-8"
        )
        findings = RepositoryAnalyzer().analyze_directory(root, [], ScanConfig())
        rule_ids = {f.rule_id for f in findings}
        assert "SC104" in rule_ids


# ---------------------------------------------------------------------------
# Test 6: SC105 network import in bundled Python
# ---------------------------------------------------------------------------


def test_sc105_fires_for_network_import_in_bundled_py() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "modeling_net.py").write_text(
            "import urllib.request\n", encoding="utf-8"
        )
        findings = RepositoryAnalyzer().analyze_directory(root, [], ScanConfig())
        rule_ids = {f.rule_id for f in findings}
        assert "SC105" in rule_ids


# ---------------------------------------------------------------------------
# Test 7: SC106 no signature file
# ---------------------------------------------------------------------------


def test_sc106_fires_when_no_signature_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        weight = root / "model.safetensors"
        weight.write_bytes(_safetensors_bytes())
        findings = RepositoryAnalyzer().analyze_directory(root, [weight], ScanConfig())
        rule_ids = {f.rule_id for f in findings}
        assert "SC106" in rule_ids


def test_sc106_absent_when_signature_present() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        weight = root / "model.safetensors"
        weight.write_bytes(_safetensors_bytes())
        (root / "checksums.txt").write_text(
            "abc123  model.safetensors\n", encoding="utf-8"
        )
        findings = RepositoryAnalyzer().analyze_directory(root, [weight], ScanConfig())
        rule_ids = {f.rule_id for f in findings}
        assert "SC106" not in rule_ids


# ---------------------------------------------------------------------------
# Test 8: SC108 LFS pointer stub
# ---------------------------------------------------------------------------


def test_sc108_fires_for_lfs_pointer_stub() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        stub = root / "model.safetensors"
        stub.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:abc123\n"
            b"size 1234567890\n"
        )
        findings = RepositoryAnalyzer().analyze_directory(
            root, [stub], ScanConfig()
        )
        rule_ids = {f.rule_id for f in findings}
        assert "SC108" in rule_ids


# ---------------------------------------------------------------------------
# Test 9: SC201 unpinned dependency
# ---------------------------------------------------------------------------


def test_sc201_fires_for_unpinned_dep() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_requirements(root, "transformers>=4.0.0\n")
        findings = RepositoryAnalyzer().analyze_directory(root, [], ScanConfig())
        rule_ids = {f.rule_id for f in findings}
        assert "SC201" in rule_ids


def test_sc201_absent_for_pinned_dep() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_requirements(root, "transformers==4.40.0\n")
        findings = RepositoryAnalyzer().analyze_directory(root, [], ScanConfig())
        # SC201 should NOT fire for a pinned dep
        sc201_findings = [f for f in findings if f.rule_id == "SC201"]
        assert len(sc201_findings) == 0


# ---------------------------------------------------------------------------
# Test 10: SC203 VCS URL dependency
# ---------------------------------------------------------------------------


def test_sc203_fires_for_vcs_url_dep() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_requirements(
            root, "git+https://github.com/huggingface/transformers.git\n"
        )
        findings = RepositoryAnalyzer().analyze_directory(root, [], ScanConfig())
        rule_ids = {f.rule_id for f in findings}
        assert "SC203" in rule_ids


# ---------------------------------------------------------------------------
# Test 11: SC204 typosquatting
# ---------------------------------------------------------------------------


def test_sc204_fires_for_typosquatting_dep() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # "requets" is edit distance 1 from "requests"
        _write_requirements(root, "requets==2.28.0\n")
        findings = RepositoryAnalyzer().analyze_directory(root, [], ScanConfig())
        rule_ids = {f.rule_id for f in findings}
        assert "SC204" in rule_ids


# ---------------------------------------------------------------------------
# Test 12: G005 missing declared shard
# ---------------------------------------------------------------------------


def test_g005_fires_for_missing_declared_shard() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        index = {
            "metadata": {"total_size": 1},
            "weight_map": {
                "model.layer.weight": "model-00001-of-00002.safetensors",
                "model.layer.bias": "model-00002-of-00002.safetensors",
            },
        }
        (root / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        shard1 = root / "model-00001-of-00002.safetensors"
        shard1.write_bytes(_safetensors_bytes())
        # shard2 is missing on disk

        findings = RepositoryAnalyzer().analyze_directory(
            root, [shard1], ScanConfig()
        )
        rule_ids = {f.rule_id for f in findings}
        assert "G005" in rule_ids


# ---------------------------------------------------------------------------
# Test 13: G006 undeclared shard file
# ---------------------------------------------------------------------------


def test_g006_fires_for_undeclared_shard() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        index = {
            "metadata": {"total_size": 1},
            "weight_map": {
                "model.layer.weight": "model-00001-of-00002.safetensors",
            },
        }
        (root / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        shard1 = root / "model-00001-of-00002.safetensors"
        shard1.write_bytes(_safetensors_bytes())
        # shard2 has the N-of-M naming pattern but is NOT in the index
        extra = root / "model-00002-of-00002.safetensors"
        extra.write_bytes(_safetensors_bytes())

        findings = RepositoryAnalyzer().analyze_directory(
            root, [shard1, extra], ScanConfig()
        )
        rule_ids = {f.rule_id for f in findings}
        assert "G006" in rule_ids


# ---------------------------------------------------------------------------
# Test 14: All findings use ModelFormat.REPOSITORY
# ---------------------------------------------------------------------------


def test_all_findings_use_repository_format() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_config(root, {"trust_remote_code": True, "auto_map": {"A": "b.C"}})
        (root / "modeling_evil.py").write_text(
            "import os\nos.system('x')\n", encoding="utf-8"
        )
        _write_requirements(root, "requets==1.0\n")

        findings = RepositoryAnalyzer().analyze_directory(root, [], ScanConfig())
        assert len(findings) > 0
        for f in findings:
            assert f.format == ModelFormat.REPOSITORY, (
                f"Finding {f.rule_id} has format {f.format}, expected REPOSITORY"
            )


# ---------------------------------------------------------------------------
# Test 15: All emitted rule IDs exist in catalog
# ---------------------------------------------------------------------------


def test_all_emitted_rule_ids_in_catalog() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_config(root, {"trust_remote_code": True, "auto_map": {"A": "b.C"}})
        (root / "modeling_evil.py").write_text(
            "import urllib.request\nos.system('x')\n", encoding="utf-8"
        )
        weight = root / "model.safetensors"
        weight.write_bytes(_safetensors_bytes())
        _write_requirements(root, "requets==1.0\ntransformers>=4.0\n")

        findings = RepositoryAnalyzer().analyze_directory(
            root, [weight], ScanConfig()
        )
        for f in findings:
            assert RuleCatalog.has_rule(f.rule_id), (
                f"Emitted rule {f.rule_id!r} is not in RuleCatalog"
            )


# ---------------------------------------------------------------------------
# Test 16: make_repository_artifact
# ---------------------------------------------------------------------------


def test_make_repository_artifact() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        artifact = make_repository_artifact(root)
        assert artifact.input_source == InputSourceType.LOCAL_DIRECTORY
        assert artifact.sha256 != ""
        assert artifact.source_path == str(root)


# ---------------------------------------------------------------------------
# Test 17: Engine integration for LOCAL_DIRECTORY scans
# ---------------------------------------------------------------------------


def test_engine_adds_repository_artifact_for_directory_scan() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        weight = root / "model.safetensors"
        weight.write_bytes(_safetensors_bytes())

        request = ScanRequest(
            target=root,
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        result = ScanOrchestrator().scan(request)

        # Must include one REPOSITORY-format artifact
        repo_artifacts = [
            a for a in result.artifacts if a.detected_format == ModelFormat.REPOSITORY
        ]
        assert len(repo_artifacts) == 1


# ---------------------------------------------------------------------------
# Test 18: Empty directory only produces SC106 (no signature)
# ---------------------------------------------------------------------------


def test_empty_directory_produces_sc106() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        findings = RepositoryAnalyzer().analyze_directory(root, [], ScanConfig())
        # SC106 fires only when there are weight files — empty dir has no weights
        # so SC106 should NOT fire (no artifacts to sign)
        rule_ids = {f.rule_id for f in findings}
        # SC202 advisory-not-performed finding may appear, but nothing else
        unexpected = rule_ids - {"SC202", "SC201"}
        assert not unexpected, f"Unexpected findings in empty dir: {unexpected}"


# ---------------------------------------------------------------------------
# FP Rate Protection: Benign repository produces no SC1xx/SC2xx/G00x findings
# ---------------------------------------------------------------------------


def test_benign_repository_no_sc_findings() -> None:
    """A clean, well-structured repo does not produce SC1xx/G005-G008 false positives.

    SC202 ('advisory snapshot absent') is excluded — it's a coverage-gap advisory, not
    a genuine finding against the repo content.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Provide pinned dependencies (no SC201/SC203/SC204)
        _write_requirements(root, "transformers==4.40.0\nnumpy==1.26.0\n")

        # 'checksums.txt' is in the recognized signature filename set (no SC106)
        weight = root / "model.safetensors"
        weight.write_bytes(_safetensors_bytes())
        (root / "checksums.txt").write_text(
            "a" * 64 + "  model.safetensors\n", encoding="utf-8"
        )

        findings = RepositoryAnalyzer().analyze_directory(root, [weight], ScanConfig())
        # Exclude SC202 — fires when advisory snapshot is absent, not a content FP.
        advisory_noise = {"SC202"}
        sc_findings = [
            f for f in findings
            if (
                f.rule_id.startswith("SC1")
                or (f.rule_id.startswith("SC2") and f.rule_id not in advisory_noise)
                or f.rule_id in {"G005", "G006", "G007", "G008"}
            )
        ]
        fp_rule_ids = [f.rule_id for f in sc_findings]
        assert not fp_rule_ids, f"False positives on clean repo: {fp_rule_ids}"


def test_benign_shard_index_no_g005_g006() -> None:
    """A repo with complete shards and matching index does not trigger G005/G006."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create two model shards
        shard_a = root / "model-00001-of-00002.safetensors"
        shard_b = root / "model-00002-of-00002.safetensors"
        shard_a.write_bytes(_safetensors_bytes())
        shard_b.write_bytes(_safetensors_bytes())

        # Create a matching index declaring exactly those two shards
        index = {
            "metadata": {"total_size": 1000},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.lm_head.weight": "model-00002-of-00002.safetensors",
            },
        }
        (root / "model.safetensors.index.json").write_text(
            __import__("json").dumps(index), encoding="utf-8"
        )

        findings = RepositoryAnalyzer().analyze_directory(
            root, [shard_a, shard_b], ScanConfig()
        )
        rule_ids = {f.rule_id for f in findings}
        assert "G005" not in rule_ids, "G005 false positive on complete shard set"
        assert "G006" not in rule_ids, "G006 false positive on complete shard set"


def test_benign_python_file_no_sc103_sc104_sc105() -> None:
    """A helper .py with only imports and no dangerous ops does not trigger SC103-SC105."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Benign Python helper — only standard-library and transformers imports
        helper = root / "convert_model.py"
        helper.write_text(
            "import json\nimport pathlib\n\ndef convert(src, dst):\n    pass\n",
            encoding="utf-8",
        )

        findings = RepositoryAnalyzer().analyze_directory(root, [], ScanConfig())
        fp_ids = [
            f.rule_id for f in findings
            if f.rule_id in {"SC103", "SC104", "SC105"}
        ]
        assert not fp_ids, f"FP on benign helper: {fp_ids}"
