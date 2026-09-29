"""
tests.integration.test_full_system_regression
=============================================

End-to-end integration and regression suite verifying the complete system lifecycle
across all model formats, acquisition channels, API surfaces, and CLI commands.

Phase 12 System Verification.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import pytest

from model_scanner import __version__
from model_scanner.__main__ import main as cli_main
from model_scanner.api import export_json, export_json_dict, scan
from model_scanner.core.models import (
    ModelFormat,
    ScanOutcome,
)


def _make_safetensors(header: dict, data: bytes = b"") -> bytes:
    raw = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + data


def _make_pytorch_zip(pickle_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", pickle_bytes)
    return buf.getvalue()


class TestFullSystemRegression:
    """End-to-end regression validation for all supported formats and entry points."""

    def test_clean_raw_pickle_scan_and_export(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.pkl"
        f.write_bytes(b"\x80\x02}q\x00.")
        result = scan(f)

        assert result.summary.overall_outcome == ScanOutcome.COMPLETE
        assert result.summary.max_risk_score == 0
        assert result.artifacts[0].detected_format == ModelFormat.RAW_PICKLE

        json_str = export_json(result)
        doc = json.loads(json_str)
        assert doc["summary"]["max_risk_score"] == 0
        assert doc["summary"]["overall_outcome"] == "COMPLETE"

    def test_malicious_raw_pickle_scan(self, tmp_path: Path) -> None:
        f = tmp_path / "malicious.pkl"
        f.write_bytes(b"cos\nsystem\n(S'id'\ntR.")
        result = scan(f)

        assert result.summary.max_risk_score > 0
        rule_ids = [f.rule_id for f in result.artifacts[0].findings]
        assert "P101" in rule_ids

    def test_modern_pytorch_zip_scan(self, tmp_path: Path) -> None:
        f = tmp_path / "model.pt"
        tensor_payload = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00."
        f.write_bytes(_make_pytorch_zip(tensor_payload))
        result = scan(f)

        assert result.artifacts[0].detected_format == ModelFormat.PYTORCH_ZIP
        assert result.summary.max_risk_score == 0

    def test_legacy_non_zip_pytorch_scan(self, tmp_path: Path) -> None:
        f = tmp_path / "legacy.bin"
        f.write_bytes(b"\x80\x02}q\x00.RAW_WEIGHT_DATA")
        result = scan(f)

        assert result.summary.overall_outcome == ScanOutcome.COMPLETE
        assert result.artifacts[0].detected_format == ModelFormat.LEGACY_PYTORCH

    def test_huggingface_legacy_pytorch_tmp_artifact_routes_to_analyzer(
        self, tmp_path: Path
    ) -> None:
        """Verify HF legacy PyTorch reaches the Pickle analyzer.

        The extension comes from the source URL, not the temporary download
        name, so a downloaded pytorch_model.bin must be treated exactly as a
        local one.
        """
        from model_scanner.core.engine import ScanOrchestrator
        from model_scanner.core.models import Artifact, InputSourceType, ScanConfig

        tmp_file = tmp_path / "tmpr9jqur11.tmp_artifact"
        tmp_file.write_bytes(b"\x80\x02}q\x00.RAW_LEGACY_PYTORCH_FLOAT_WEIGHTS")

        artifact = Artifact(
            artifact_id="test-hf-legacy",
            source_path=(
                "https://huggingface.co/sshleifer/"
                "tiny-distilbert-base-cased-distilled-squad/resolve/main/pytorch_model.bin"
            ),
            input_source=InputSourceType.PUBLIC_URL,
            size_bytes=tmp_file.stat().st_size,
            sha256="0" * 64,
            local_path=tmp_file,
        )

        orchestrator = ScanOrchestrator(ScanConfig())
        art_result = orchestrator.scan_artifact(artifact)

        assert art_result.detected_format == ModelFormat.LEGACY_PYTORCH
        assert art_result.scan_outcome == ScanOutcome.COMPLETE
        assert art_result.risk_score.final_score == 0
        assert len(art_result.findings) == 0
        assert len(art_result.errors) == 0

    def test_clean_safetensors_scan(self, tmp_path: Path) -> None:
        f = tmp_path / "model.safetensors"
        f.write_bytes(
            _make_safetensors(
                {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
                data=bytes(4),
            )
        )
        result = scan(f)

        assert result.artifacts[0].detected_format == ModelFormat.SAFETENSORS
        assert result.summary.overall_outcome == ScanOutcome.COMPLETE
        assert result.summary.max_risk_score == 0

    def test_malformed_safetensors_scan(self, tmp_path: Path) -> None:
        f = tmp_path / "corrupted.safetensors"
        f.write_bytes(b"\x00\x00\x00\x00")
        result = scan(f)

        assert result.summary.overall_outcome == ScanOutcome.ERROR

    def test_clean_gguf_scan(self, tmp_path: Path) -> None:
        f = tmp_path / "model.gguf"
        # Magic GGUF + version 3 + 0 tensors + 0 KV
        f.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 0))
        result = scan(f)

        assert result.artifacts[0].detected_format == ModelFormat.GGUF
        assert result.summary.overall_outcome == ScanOutcome.COMPLETE

    def test_malformed_gguf_scan(self, tmp_path: Path) -> None:
        f = tmp_path / "corrupted.gguf"
        f.write_bytes(b"GGUFCORRUPTED_TRUNCATED")
        result = scan(f)

        # GGUF parser handles corrupted headers gracefully with security finding GU202
        assert result.summary.overall_outcome == ScanOutcome.WARNING
        rule_ids = [f.rule_id for f in result.artifacts[0].findings]
        assert "GU202" in rule_ids


class TestCLIEntryPoints:
    """Verify standard command-line interface subcommands."""

    def test_cli_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_cli_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Static Security Scanner" in captured.out

    def test_cli_scan_clean(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.pkl"
        f.write_bytes(b"\x80\x02}q\x00.")
        exit_code = cli_main(["scan", str(f)])
        assert exit_code == 0

    def test_cli_scan_malicious(self, tmp_path: Path) -> None:
        f = tmp_path / "malicious.pkl"
        f.write_bytes(b"cos\nsystem\n(S'id'\ntR.")
        exit_code = cli_main(["scan", str(f)])
        assert exit_code == 1


class TestPythonAPISurface:
    """Verify programmatic Python API entry points."""

    def test_export_json_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.pkl"
        f.write_bytes(b"\x80\x02}q\x00.")
        res = scan(f)
        doc = export_json_dict(res)
        assert isinstance(doc, dict)
        assert "$schema" not in doc, "$schema removed — URL pointed to non-existent repo"
        assert doc["scanner_version"] == __version__
        assert "artifacts" in doc
