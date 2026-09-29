"""
tests.unit.test_py_ast_scanner
================================

Unit tests for model_scanner.analyzers.repository.py_ast_scanner.
Covers SC103/SC104/SC105/SC106 detection paths.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from model_scanner.analyzers.repository.py_ast_scanner import (
    is_bundled_code_file,
    scan_python_file,
    scan_python_files_in_repo,
)


def _write_py(tmp_path: Path, name: str, code: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(code), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# is_bundled_code_file
# ---------------------------------------------------------------------------


class TestIsBundledCodeFile:
    def test_modeling_py_is_bundled(self, tmp_path: Path) -> None:
        f = tmp_path / "modeling_llama.py"
        assert is_bundled_code_file(f) is True

    def test_configuration_py_is_bundled(self, tmp_path: Path) -> None:
        f = tmp_path / "configuration_gpt2.py"
        assert is_bundled_code_file(f) is True

    def test_tokenization_py_is_bundled(self, tmp_path: Path) -> None:
        f = tmp_path / "tokenization_bert.py"
        assert is_bundled_code_file(f) is True

    def test_random_py_not_bundled(self, tmp_path: Path) -> None:
        f = tmp_path / "utils.py"
        assert is_bundled_code_file(f) is False


# ---------------------------------------------------------------------------
# scan_python_file — dangerous builtins (SC104)
# ---------------------------------------------------------------------------


class TestSC104DangerousBuiltins:
    def test_eval_emits_sc104(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "modeling.py", "x = eval('1+1')\n")
        findings = scan_python_file(f)
        assert any(fi.rule_id == "SC104" for fi in findings)

    def test_exec_emits_sc104(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "modeling.py", "exec('print(1)')\n")
        findings = scan_python_file(f)
        assert any(fi.rule_id == "SC104" for fi in findings)

    def test_os_system_emits_sc104(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "modeling.py", "import os\nos.system('ls')\n")
        findings = scan_python_file(f)
        assert any(fi.rule_id == "SC104" for fi in findings)

    def test_subprocess_popen_emits_sc104(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "modeling.py", "import subprocess\nsubprocess.Popen(['ls'])\n")
        findings = scan_python_file(f)
        assert any(fi.rule_id == "SC104" for fi in findings)

    def test_clean_file_no_findings(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "modeling.py", "x = 1 + 2\n")
        findings = scan_python_file(f)
        assert findings == []


# ---------------------------------------------------------------------------
# scan_python_file — network imports (SC105)
# ---------------------------------------------------------------------------


class TestSC105NetworkImports:
    def test_import_socket_emits_sc105(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "modeling.py", "import socket\n")
        findings = scan_python_file(f)
        assert any(fi.rule_id == "SC105" for fi in findings)

    def test_import_urllib_emits_sc105(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "modeling.py", "import urllib.request\n")
        findings = scan_python_file(f)
        assert any(fi.rule_id == "SC105" for fi in findings)

    def test_from_import_requests_emits_sc105(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "modeling.py", "from requests import get\n")
        findings = scan_python_file(f)
        assert any(fi.rule_id == "SC105" for fi in findings)


# ---------------------------------------------------------------------------
# scan_python_file — base64 decode (SC106)
# ---------------------------------------------------------------------------


class TestSC106Base64:
    def test_base64_b64decode_import_emits_sc106(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "modeling.py", "from base64 import b64decode\n")
        findings = scan_python_file(f)
        # SC106 rule
        assert any(fi.rule_id == "SC106" for fi in findings)

    def test_base64_b64decode_call_emits_sc106(self, tmp_path: Path) -> None:
        f = _write_py(
            tmp_path, "modeling.py",
            "import base64\nbase64.b64decode('abc')\n"
        )
        findings = scan_python_file(f)
        assert any(fi.rule_id == "SC106" for fi in findings)


# ---------------------------------------------------------------------------
# scan_python_file — error handling
# ---------------------------------------------------------------------------


class TestScanPythonFileErrors:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        findings = scan_python_file(tmp_path / "nonexistent.py")
        assert findings == []

    def test_syntax_error_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.py"
        f.write_bytes(b"def broken(\n")
        findings = scan_python_file(f)
        assert findings == []


# ---------------------------------------------------------------------------
# scan_python_files_in_repo
# ---------------------------------------------------------------------------


class TestScanPythonFilesInRepo:
    def test_finds_dangerous_modeling_file(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "modeling_custom.py", "import os\nos.system('ls')\n")
        results = scan_python_files_in_repo(tmp_path)
        assert len(results) >= 1

    def test_skips_non_bundled_files_by_default(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "utils.py", "exec('x')\n")
        results = scan_python_files_in_repo(tmp_path)
        assert results == {}

    def test_scan_all_py_finds_utils(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "utils.py", "exec('x')\n")
        results = scan_python_files_in_repo(tmp_path, scan_all_py=True)
        assert len(results) >= 1

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".git"
        hidden.mkdir()
        _write_py(hidden, "modeling_custom.py", "exec('x')\n")
        results = scan_python_files_in_repo(tmp_path)
        assert results == {}

    def test_empty_repo_returns_empty(self, tmp_path: Path) -> None:
        results = scan_python_files_in_repo(tmp_path)
        assert results == {}
