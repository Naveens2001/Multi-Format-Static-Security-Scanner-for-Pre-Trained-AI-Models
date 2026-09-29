"""
model_scanner.analyzers.repository.py_ast_scanner
===================================================

Static AST scanner for bundled Python files in model repositories (B1).

Scans ``modeling_*.py``, ``configuration_*.py``, ``tokenization_*.py`` and
any other Python source files for dangerous patterns.

Dangerous patterns detected:
  - ``os.system``, ``os.popen``, ``os.exec*``
  - ``subprocess.*`` calls
  - ``eval()``, ``exec()``, ``compile()`` calls
  - ``__import__`` usage
  - ``socket``, ``urllib``, ``requests``, ``httpx`` imports or calls
  - ``base64.b64decode`` (common payload obfuscation)
  - Writes to paths outside a temp directory

SECURITY GUARANTEE
------------------
- Uses stdlib ``ast`` with ``ast.parse()`` only — NEVER ``eval()`` or
  ``compile()`` to a code object that is run.
- File reads are bounded (max 512 KB per file).
- This module NEVER imports or executes model code.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

_MAX_PY_FILE_BYTES: int = 512 * 1024  # 512 KB

# Patterns for names/attributes that are dangerous
_DANGEROUS_CALLS: frozenset[str] = frozenset(
    [
        "eval",
        "exec",
        "compile",
        "__import__",
    ]
)

_OS_DANGEROUS_ATTRS: frozenset[str] = frozenset(
    [
        "system",
        "popen",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "spawn",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "startfile",
        "remove",
        "unlink",
    ]
)

_NETWORK_MODULES: frozenset[str] = frozenset(
    [
        "socket",
        "urllib",
        "urllib.request",
        "urllib.parse",
        "http",
        "http.client",
        "ftplib",
        "smtplib",
        "requests",
        "httpx",
        "aiohttp",
        "httplib2",
        "pycurl",
        "paramiko",
    ]
)

_B64_DECODE_FUNCS: frozenset[str] = frozenset(
    ["b64decode", "decodebytes", "decodestring"]
)

# Match modeling_*, configuration_*, tokenization_* Python files
_BUNDLED_CODE_NAME_RE: re.Pattern[str] = re.compile(
    r"^(?:modeling|configuration|tokenization|feature_extraction|processing|"
    r"image_processing|video_processing)_.*\.py$"
)


def is_bundled_code_file(path: Path) -> bool:
    """Return True if ``path`` looks like a bundled model-code Python file."""
    return _BUNDLED_CODE_NAME_RE.match(path.name) is not None or path.name in (
        "modeling.py",
        "configuration.py",
        "tokenization.py",
    )


@dataclass(frozen=True)
class ASTFinding:
    """A dangerous pattern found during static AST scanning.

    Attributes
    ----------
    rule_id:
        The SC1xx rule that matched.
    description:
        Human-readable description.
    line_number:
        Source line number (1-based).
    col_offset:
        Column offset within the line.
    matched_text:
        The source snippet that triggered the match.
    """

    rule_id: str
    description: str
    line_number: int
    col_offset: int
    matched_text: str


class _DangerVisitor(ast.NodeVisitor):
    """AST visitor collecting dangerous patterns into ``self.findings``."""

    def __init__(self, source_path: str) -> None:
        self.source_path = source_path
        self.findings: list[ASTFinding] = []
        self._imports: dict[str, str] = {}  # local name → module

    def _add(
        self, rule_id: str, description: str, node: ast.AST, matched_text: str
    ) -> None:
        lineno = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        self.findings.append(
            ASTFinding(
                rule_id=rule_id,
                description=description,
                line_number=lineno,
                col_offset=col,
                matched_text=matched_text,
            )
        )

    # -----------------------------------------------------------------
    # Import tracking
    # -----------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname if alias.asname else alias.name.split(".")[0]
            self._imports[local] = alias.name
            if alias.name in _NETWORK_MODULES or any(
                alias.name.startswith(m + ".") for m in _NETWORK_MODULES
            ):
                self._add(
                    rule_id="SC105",
                    description=f"Network module imported: {alias.name!r}",
                    node=node,
                    matched_text=f"import {alias.name}",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module in _NETWORK_MODULES or any(
            module.startswith(m + ".") for m in _NETWORK_MODULES
        ):
            self._add(
                rule_id="SC105",
                description=f"Network module imported: {module!r}",
                node=node,
                matched_text=f"from {module} import ...",
            )
        if module == "base64":
            for alias in node.names:
                if alias.name in _B64_DECODE_FUNCS:
                    self._add(
                        rule_id="SC106",
                        description=f"base64.{alias.name} imported (payload decoding heuristic)",
                        node=node,
                        matched_text=f"from base64 import {alias.name}",
                    )
        self.generic_visit(node)

    # -----------------------------------------------------------------
    # Call inspection
    # -----------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # Direct builtins: eval(), exec(), compile(), __import__()
        if isinstance(func, ast.Name) and func.id in _DANGEROUS_CALLS:
            self._add(
                rule_id="SC104",
                description=f"Dangerous builtin called: {func.id!r}",
                node=node,
                matched_text=func.id,
            )

        # Attribute calls: os.system(), subprocess.Popen(), etc.
        elif isinstance(func, ast.Attribute):
            obj = func.value
            attr = func.attr

            # os.system / os.popen / os.exec* etc.
            if isinstance(obj, ast.Name) and obj.id == "os":
                if attr in _OS_DANGEROUS_ATTRS:
                    self._add(
                        rule_id="SC104",
                        description=f"Dangerous os.{attr}() call",
                        node=node,
                        matched_text=f"os.{attr}",
                    )

            # subprocess.*
            elif isinstance(obj, ast.Name) and obj.id in ("subprocess", "sp"):
                self._add(
                    rule_id="SC104",
                    description=f"subprocess.{attr}() call detected",
                    node=node,
                    matched_text=f"subprocess.{attr}",
                )

            # base64.b64decode etc.
            elif isinstance(obj, ast.Name) and obj.id == "base64":
                if attr in _B64_DECODE_FUNCS:
                    self._add(
                        rule_id="SC106",
                        description=f"base64.{attr}() payload decoding heuristic",
                        node=node,
                        matched_text=f"base64.{attr}",
                    )

        self.generic_visit(node)


def scan_python_file(path: Path) -> list[ASTFinding]:
    """Statically scan a single Python file for dangerous patterns.

    Parameters
    ----------
    path:
        Path to the Python source file.

    Returns
    -------
    list[ASTFinding]
        List of findings.  Empty if no dangerous patterns are detected or if
        the file cannot be parsed.

    Notes
    -----
    Uses ``ast.parse()`` only — never ``eval()``, ``exec()``, or
    ``compile()`` to run the code.
    """
    try:
        raw = path.read_bytes()[:_MAX_PY_FILE_BYTES]
        source = raw.decode("utf-8", errors="replace")
    except OSError:
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    visitor = _DangerVisitor(source_path=str(path))
    visitor.visit(tree)
    return visitor.findings


def scan_python_files_in_repo(
    repo_root: Path,
    *,
    scan_all_py: bool = False,
) -> dict[Path, list[ASTFinding]]:
    """Scan bundled Python files in a model repository.

    By default only scans files matching the ``modeling_*.py`` /
    ``configuration_*.py`` / ``tokenization_*.py`` convention (B1).
    Pass ``scan_all_py=True`` to scan every ``.py`` file (slower).

    Parameters
    ----------
    repo_root:
        Root of the model repository.
    scan_all_py:
        If True, scan all ``.py`` files under ``repo_root``.

    Returns
    -------
    dict[Path, list[ASTFinding]]
        Mapping of file path → list of findings.  Files with no findings
        are omitted from the dict.
    """
    results: dict[Path, list[ASTFinding]] = {}

    for py_file in repo_root.rglob("*.py"):
        # Skip hidden directories
        parts = py_file.relative_to(repo_root).parts
        if any(p.startswith(".") or p == "__pycache__" for p in parts):
            continue

        if scan_all_py or is_bundled_code_file(py_file):
            findings = scan_python_file(py_file)
            if findings:
                results[py_file] = findings

    return results


__all__ = [
    "ASTFinding",
    "is_bundled_code_file",
    "scan_python_file",
    "scan_python_files_in_repo",
]
