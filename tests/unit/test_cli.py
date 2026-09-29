"""
tests.unit.test_cli
===================

Unit tests for Phase 10 CLI entry point (`__main__.py`).
Tests --help, --version, serve command dispatch, scan command execution,
and ClamAV exit code propagation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model_scanner import __version__
from model_scanner.__main__ import _create_parser, main


class TestCLIParsing:
    """Test CLI argument parsing and dispatch."""

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify --version prints the version string and exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        ver_str = f"model-scanner v{__version__}"
        assert ver_str in captured.out or ver_str in captured.err

    def test_help_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify --help displays commands and exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "serve" in captured.out
        assert "scan" in captured.out

    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify running with no arguments prints help and returns 0."""
        exit_code = main([])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Commands" in captured.out
        assert "serve" in captured.out

    def test_invalid_command(self) -> None:
        """Verify invalid commands raise SystemExit 2."""
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent-command"])
        assert exc_info.value.code == 2

    def test_parser_serve_defaults(self) -> None:
        """Verify serve subparser default arguments."""
        parser = _create_parser()
        args = parser.parse_args(["serve"])
        assert args.command == "serve"
        assert args.host == "127.0.0.1"
        assert args.port == 8000

    def test_parser_serve_custom_host_port(self) -> None:
        """Verify serve subparser custom arguments."""
        parser = _create_parser()
        args = parser.parse_args(["serve", "--host", "0.0.0.0", "-p", "9090"])
        assert args.command == "serve"
        assert args.host == "0.0.0.0"
        assert args.port == 9090

    def test_parser_scan_arguments(self) -> None:
        """Verify scan subparser arguments."""
        parser = _create_parser()
        args = parser.parse_args(
            ["scan", "weights.safetensors", "--strict", "-o", "report.json", "--json"]
        )
        assert args.command == "scan"
        assert args.target == "weights.safetensors"
        assert args.strict is True
        assert args.output == "report.json"
        assert args.json is True


class TestCLIExecution:
    """Test CLI command execution workflows."""

    @patch("model_scanner.__main__.serve")
    def test_serve_command_dispatch(
        self, mock_serve: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify serve command initializes WebServer, prints banner, and handles Ctrl+C."""
        mock_server = MagicMock()
        mock_server.server_address = ("127.0.0.1", 8000)
        mock_server.start.side_effect = KeyboardInterrupt
        mock_serve.return_value = mock_server

        exit_code = main(["serve", "--host", "127.0.0.1", "--port", "8000"])
        assert exit_code == 0

        mock_serve.assert_called_once_with(
            host="127.0.0.1", port=8000, background=False
        )
        mock_server.start.assert_called_once()
        mock_server.stop.assert_called_once()

        captured = capsys.readouterr()
        assert "Model Scanner Web UI" in captured.out
        assert "http://127.0.0.1:8000" in captured.out
        assert "Server stopped cleanly." in captured.out

    def test_scan_command_clean_model(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify scan command on a clean pickle exits with ClamAV code 0."""
        clean_file = tmp_path / "clean_model.pkl"
        clean_file.write_bytes(b"\x80\x02}q\x00.")

        exit_code = main(["scan", str(clean_file)])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Total Artifacts" in captured.err
        assert "Exit Code           : 0" in captured.err

    def test_scan_command_malicious_model(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify scan command on a malicious pickle exits with scanner exit code 1."""
        malicious_file = tmp_path / "malicious.pkl"
        malicious_file.write_bytes(b"cos\nsystem\n(S'id'\ntR.")

        exit_code = main(["scan", str(malicious_file)])
        assert exit_code == 1

        captured = capsys.readouterr()
        assert "Exit Code           : 1" in captured.err
        assert "P101" in captured.err

    def test_scan_command_with_output_json_file(self, tmp_path: Path) -> None:
        """Verify scan command saves canonical JSON output to specified file."""
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"\x80\x02}q\x00.")
        report_file = tmp_path / "output_report.json"

        exit_code = main(["scan", str(model_file), "-o", str(report_file)])
        assert exit_code == 0
        assert report_file.exists()

        data = json.loads(report_file.read_text(encoding="utf-8"))
        assert "summary" in data
        assert "artifacts" in data
        assert data["summary"]["exit_code"] == 0

    def test_scan_command_with_raw_json_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify scan command --json outputs valid JSON string to stdout."""
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"\x80\x02}q\x00.")

        exit_code = main(["scan", str(model_file), "--json"])
        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["summary"]["total_artifacts"] == 1


class TestCLINewFlags:
    """Test new CLI flags: --include, --exclude, --max-size, --timeout, --format, --config."""

    def test_include_flag_parses(self) -> None:
        """--include wires to ScanConfig.include_pattern."""
        parser = _create_parser()
        args = parser.parse_args(["scan", "model.pt", "--include", r"\.safetensors$"])
        assert args.include == r"\.safetensors$"

    def test_exclude_flag_parses(self) -> None:
        """--exclude wires to ScanConfig.exclude_pattern."""
        parser = _create_parser()
        args = parser.parse_args(["scan", "model.pt", "--exclude", r"^tmp/"])
        assert args.exclude == r"^tmp/"

    def test_max_size_flag_parses(self) -> None:
        """--max-size parses as int."""
        parser = _create_parser()
        args = parser.parse_args(["scan", "model.pt", "--max-size", "512"])
        assert args.max_size == 512

    def test_timeout_flag_parses(self) -> None:
        """--timeout parses as int."""
        parser = _create_parser()
        args = parser.parse_args(["scan", "model.pt", "--timeout", "30"])
        assert args.timeout == 30

    def test_format_json_flag(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """--format json produces JSON output (equivalent to --json)."""
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"\x80\x02}q\x00.")

        exit_code = main(["scan", str(model_file), "--format", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "summary" in data

    def test_config_flag_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """--config with a JSON file is loaded and applied."""
        import json as _json

        cfg_file = tmp_path / "scanner.json"
        cfg_file.write_text(_json.dumps({"strict_mode": False}), encoding="utf-8")

        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"\x80\x02}q\x00.")

        exit_code = main(["scan", str(model_file), "--config", str(cfg_file), "--json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "summary" in data

    def test_config_flag_missing_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--config with a missing file exits with code 2 and prints an error."""
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"\x80\x02}q\x00.")

        exit_code = main(["scan", str(model_file), "--config", str(tmp_path / "nonexistent.json")])
        assert exit_code == 2
        captured = capsys.readouterr()
        assert "Error" in captured.err or "Error" in captured.out


class TestConfigLoader:
    """Unit tests for model_scanner.config.loader."""

    def test_load_json_config(self, tmp_path: Path) -> None:
        """load_config parses a valid JSON config and returns ScanConfig."""
        import json as _json

        from model_scanner.config.loader import load_config

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(
            _json.dumps({"strict_mode": True, "download_timeout_sec": 120}),
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.strict_mode is True
        assert cfg.download_timeout_sec == 120

    def test_load_toml_config(self, tmp_path: Path) -> None:
        """load_config parses a TOML config with [scanner] table."""
        from model_scanner.config.loader import load_config

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            "[scanner]\nstrict_mode = true\ndownload_timeout_sec = 45\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.strict_mode is True
        assert cfg.download_timeout_sec == 45

    def test_load_config_ignores_unknown_keys(self, tmp_path: Path) -> None:
        """load_config ignores unrecognized keys without error."""
        import json as _json

        from model_scanner.config.loader import load_config

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(
            _json.dumps({"strict_mode": False, "unknown_option": "xyz"}),
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.strict_mode is False

    def test_load_config_missing_file_raises(self, tmp_path: Path) -> None:
        """load_config raises FileNotFoundError for missing files."""
        from model_scanner.config.loader import load_config

        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.json")

    def test_load_config_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """load_config raises ValueError for unsupported file extensions."""
        from model_scanner.config.loader import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("strict_mode: true\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config(cfg_file)


class TestJSONExportCWEOWASP:
    """Verify that cwe_ids, owasp_category, mitre_technique appear in JSON export."""

    def test_finding_export_includes_cwe_and_owasp(self, tmp_path: Path) -> None:
        """JSON export includes cwe_ids, owasp_category, mitre_technique per finding."""
        from model_scanner import scan
        from model_scanner.api import export_json_dict

        model_file = tmp_path / "malicious.pkl"
        model_file.write_bytes(b"cos\nsystem\n(S'id'\ntR.")

        result = scan(model_file)
        data = export_json_dict(result)

        for art in data.get("artifacts", []):
            for f in art.get("findings", []):
                assert "cwe_ids" in f, f"cwe_ids missing from finding {f['rule_id']}"
                assert "owasp_category" in f, f"owasp_category missing from {f['rule_id']}"
                assert "mitre_technique" in f, f"mitre_technique missing from {f['rule_id']}"
                assert "confidence" in f, f"confidence missing from {f['rule_id']}"
