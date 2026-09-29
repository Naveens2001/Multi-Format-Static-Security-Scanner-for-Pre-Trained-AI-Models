"""
tests.unit.test_api
===================

Unit tests for Phase 10 public Python API (`model_scanner.api` and top-level package exports).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import model_scanner
from model_scanner import export_json, export_json_dict, scan, serve
from model_scanner.core.models import (
    InputSourceType,
    ModelFormat,
    ScanConfig,
    ScanOutcome,
    ScanResult,
)

# Established benign standard pickle payload (PROTO 2 empty dict)
_BENIGN_PICKLE = b"\x80\x02}q\x00."


class TestPublicAPI:
    """Test public programmatic interface."""

    def test_top_level_package_exports(self) -> None:
        """Verify model_scanner exports key entrypoints and data models."""
        assert callable(model_scanner.scan)
        assert callable(model_scanner.serve)
        assert callable(model_scanner.export_json)
        assert callable(model_scanner.export_json_dict)
        assert hasattr(model_scanner, "__version__")

    def test_scan_local_file(self, tmp_path: Path) -> None:
        """Verify scan() on a local file returns valid ScanResult."""
        clean_file = tmp_path / "model.pkl"
        clean_file.write_bytes(_BENIGN_PICKLE)

        result = scan(clean_file)
        assert isinstance(result, ScanResult)
        assert len(result.artifacts) == 1
        assert result.artifacts[0].detected_format == ModelFormat.RAW_PICKLE
        assert result.artifacts[0].scan_outcome == ScanOutcome.COMPLETE
        assert result.artifacts[0].risk_score.final_score == 0
        assert result.summary.exit_code == 0
        assert result.summary.overall_outcome == ScanOutcome.COMPLETE

    def test_scan_local_directory(self, tmp_path: Path) -> None:
        """Verify scan() on a local directory scans contained model files.

        Directory scans produce a REPOSITORY artifact (supply-chain checks) plus
        one artifact per discovered model file.
        """
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(_BENIGN_PICKLE)

        result = scan(tmp_path)
        assert isinstance(result, ScanResult)
        # Directory scans yield a REPOSITORY artifact + per-file artifacts.
        formats = {a.detected_format for a in result.artifacts}
        assert ModelFormat.REPOSITORY in formats
        assert ModelFormat.RAW_PICKLE in formats
        # The RAW_PICKLE artifact itself must be clean.
        pickle_artifact = next(
            a for a in result.artifacts if a.detected_format == ModelFormat.RAW_PICKLE
        )
        assert pickle_artifact.scan_outcome == ScanOutcome.COMPLETE
        assert pickle_artifact.risk_score.final_score == 0

    @patch("model_scanner.core.engine.ScanOrchestrator.scan")
    def test_scan_public_url_dispatch(self, mock_scan: MagicMock) -> None:
        """Verify scan() with a URL correctly configures PUBLIC_URL scan request."""
        mock_scan.return_value = MagicMock(spec=ScanResult)
        url = "https://huggingface.co/bert-base-uncased"

        scan(url)
        assert mock_scan.call_count == 1
        req = mock_scan.call_args[0][0]
        assert req.input_source_type == InputSourceType.PUBLIC_URL
        assert req.target == url

    def test_scan_with_custom_config_strict_mode(self, tmp_path: Path) -> None:
        """Verify scan() propagates custom ScanConfig and strict mode."""
        clean_file = tmp_path / "model.pkl"
        clean_file.write_bytes(_BENIGN_PICKLE)

        config = ScanConfig(strict_mode=True)
        result = scan(clean_file, config=config, strict=True)
        assert isinstance(result, ScanResult)
        assert result.strict_mode is True
        assert result.summary.exit_code == 0
        assert result.summary.overall_outcome == ScanOutcome.COMPLETE

    def test_export_json(self, tmp_path: Path) -> None:
        """Verify export_json() returns valid canonical JSON string."""
        clean_file = tmp_path / "model.pkl"
        clean_file.write_bytes(_BENIGN_PICKLE)

        result = scan(clean_file)
        json_str = export_json(result)

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert "$schema" not in data, "$schema removed — URL pointed to non-existent repo"
        assert "scanner_version" in data
        assert "summary" in data
        assert "artifacts" in data

    def test_export_json_dict(self, tmp_path: Path) -> None:
        """Verify export_json_dict() returns valid dictionary representation."""
        clean_file = tmp_path / "model.pkl"
        clean_file.write_bytes(_BENIGN_PICKLE)

        result = scan(clean_file)
        data_dict = export_json_dict(result)

        assert isinstance(data_dict, dict)
        assert data_dict["summary"]["total_artifacts"] == 1
        assert data_dict["summary"]["exit_code"] == 0

    def test_serve_background(self) -> None:
        """Verify serve(background=True) starts WebServer on ephemeral port."""
        server = serve(host="127.0.0.1", port=0, background=True)
        try:
            host, port = server.server_address
            assert host == "127.0.0.1"
            assert port > 0
        finally:
            server.stop()
