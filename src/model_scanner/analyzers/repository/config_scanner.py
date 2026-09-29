"""
model_scanner.analyzers.repository.config_scanner
===================================================

Supply chain — remote code execution vector detection in model configs (B1).

Scans ``config.json`` and ``tokenizer_config.json`` for fields that enable
arbitrary Python code execution when a model is loaded:

  - ``auto_map`` — maps model classes to bundled Python modules.
  - ``trust_remote_code: true`` — Hugging Face flag that executes bundled code.
  - ``custom_pipelines`` — arbitrary pipeline classes.
  - Dotted ``module.ClassName`` patterns in any string value.
  - Presence of ``modeling_*.py`` / ``configuration_*.py`` / ``tokenization_*.py``
    alongside weights (bundled executable code).

Design Reference
----------------
Inspired by how modelaudit (MIT, https://github.com/promptfoo/modelaudit)
detects trust_remote_code and auto_map in config files (modelaudit/scanner.py).
No code was copied; only the conceptual check is adopted.

SECURITY GUARANTEE
------------------
- Reads bounded JSON with stdlib ``json`` only.
- No eval / exec / pickle.loads / torch.load.
- No network calls.
- YAML is not used anywhere in this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_CONFIG_BYTES: int = 256 * 1024  # 256 KB

# Regex matching a Python dotted module.ClassName path (e.g. modeling_custom.MyModel)
# Must have at least one dot separating a module from a class name.
_DOTTED_PATH_RE: re.Pattern[str] = re.compile(
    r"^[a-zA-Z_][a-zA-Z0-9_.]*\.[A-Z][a-zA-Z0-9_]*$"
)

# Keys that are known to hold remote-code execution vectors
_AUTO_MAP_KEY: str = "auto_map"
_TRUST_REMOTE_CODE_KEY: str = "trust_remote_code"
_CUSTOM_PIPELINES_KEY: str = "custom_pipelines"


@dataclass(frozen=True)
class ConfigFinding:
    """A supply chain finding from repository config analysis (B1).

    Attributes
    ----------
    rule_id:
        The SC1xx rule that matched.
    config_file:
        The config file where the finding was made (relative path from repo root).
    field_path:
        JSON path of the triggering field (e.g. ``auto_map.AutoModel``).
    matched_value:
        The value that triggered the finding.
    description:
        Human-readable description.
    referenced_py_file:
        If the auto_map value references a Python file, the path if it exists
        in the repository; ``None`` otherwise.
    """

    rule_id: str
    config_file: str
    field_path: str
    matched_value: str
    description: str
    referenced_py_file: str | None = None


def _read_json_bounded(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read and parse a JSON file, capped at _MAX_CONFIG_BYTES.

    Returns ``(parsed_dict, error_string)``.
    """
    try:
        raw = path.read_bytes()[:_MAX_CONFIG_BYTES]
        text = raw.decode("utf-8", errors="replace")
    except OSError as exc:
        return None, f"Cannot read {path.name}: {exc}"

    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None, f"{path.name}: JSON root is not an object"
        return obj, None
    except json.JSONDecodeError as exc:
        return None, f"{path.name}: JSON parse error: {exc}"


def _extract_auto_map_module(value: str) -> str | None:
    """Extract the Python module name from an auto_map value string.

    Auto_map values look like ``"modeling_custom.MyModel"`` or
    ``"./modeling_custom.MyModel"``.  Returns the module part (before the dot
    that separates module from class), or ``None`` if not recognisable.
    """
    # Strip leading './' or '../'
    cleaned = value.lstrip("./")
    # Module is everything before the last dotted segment
    parts = cleaned.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:-1])
    return None


def _check_for_referenced_py(module_name: str, repo_root: Path) -> str | None:
    """Return relative path of the Python file referenced by an auto_map entry."""
    # Try both with and without a leading path separator
    for candidate in (
        f"{module_name}.py",
        f"{module_name.replace('.', '/')}.py",
    ):
        candidate_path = repo_root / candidate
        if candidate_path.is_file():
            return str(candidate_path.relative_to(repo_root))
    return None


def _scan_value_for_dotted_path(
    value: object, field_path: str
) -> str | None:
    """Return the value string if it matches a dotted module.ClassName pattern."""
    if not isinstance(value, str):
        return None
    if _DOTTED_PATH_RE.match(value):
        return value
    return None


def _walk_for_dotted_paths(
    obj: object,
    path: str,
    findings: list[ConfigFinding],
    config_file: str,
    *,
    depth: int = 0,
) -> None:
    """Recursively walk a JSON object looking for dotted module.ClassName values."""
    if depth > 5:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            child_path = f"{path}.{k}" if path else k
            _walk_for_dotted_paths(v, child_path, findings, config_file, depth=depth + 1)
    elif isinstance(obj, list):
        for i, item in enumerate(obj[:50]):  # bound the iteration
            _walk_for_dotted_paths(item, f"{path}[{i}]", findings, config_file, depth=depth + 1)
    elif isinstance(obj, str):
        dotted = _scan_value_for_dotted_path(obj, path)
        if dotted:
            findings.append(
                ConfigFinding(
                    rule_id="SC101",
                    config_file=config_file,
                    field_path=path,
                    matched_value=dotted,
                    description=(
                        f"Config field {path!r} contains a dotted Python module.ClassName "
                        f"reference ({dotted!r}) that may load remote code."
                    ),
                )
            )


def scan_config_file(
    path: Path,
    repo_root: Path,
    *,
    config_file_name: str | None = None,
) -> list[ConfigFinding]:
    """Scan a single config JSON file for remote code execution vectors.

    Parameters
    ----------
    path:
        Absolute path to the config file.
    repo_root:
        Absolute root of the model repository.
    config_file_name:
        Display name to use in findings (defaults to ``path.name``).

    Returns
    -------
    list[ConfigFinding]
        List of supply chain config findings.
    """
    label = config_file_name or path.name
    parsed, _err = _read_json_bounded(path)
    if parsed is None:
        return []  # unreadable / not JSON — not our problem to report here

    findings: list[ConfigFinding] = []

    # --- B1.1: auto_map field ---
    if _AUTO_MAP_KEY in parsed:
        auto_map = parsed[_AUTO_MAP_KEY]
        if isinstance(auto_map, dict):
            for model_class, target in auto_map.items():
                if not isinstance(target, str):
                    continue
                module_name = _extract_auto_map_module(target)
                ref_file = None
                if module_name:
                    ref_file = _check_for_referenced_py(module_name, repo_root)

                findings.append(
                    ConfigFinding(
                        rule_id="SC101",
                        config_file=label,
                        field_path=f"auto_map.{model_class}",
                        matched_value=target,
                        description=(
                            f"auto_map[{model_class!r}] = {target!r} — "
                            f"loads bundled Python module at model-load time."
                        ),
                        referenced_py_file=ref_file,
                    )
                )

    # --- B1.2: trust_remote_code ---
    if parsed.get(_TRUST_REMOTE_CODE_KEY) is True:
        findings.append(
            ConfigFinding(
                rule_id="SC102",
                config_file=label,
                field_path=_TRUST_REMOTE_CODE_KEY,
                matched_value="true",
                description=(
                    "trust_remote_code=true allows arbitrary Python code in the "
                    "repository to execute when the model is loaded via Hugging Face."
                ),
            )
        )

    # --- B1.3: custom_pipelines ---
    if _CUSTOM_PIPELINES_KEY in parsed:
        cp = parsed[_CUSTOM_PIPELINES_KEY]
        if isinstance(cp, dict):
            for pipeline_name, pipeline_def in cp.items():
                if isinstance(pipeline_def, dict):
                    impl = pipeline_def.get("impl", "")
                    if isinstance(impl, str) and impl:
                        findings.append(
                            ConfigFinding(
                                rule_id="SC101",
                                config_file=label,
                                field_path=f"custom_pipelines.{pipeline_name}.impl",
                                matched_value=impl,
                                description=(
                                    f"custom_pipelines[{pipeline_name!r}].impl = {impl!r} "
                                    f"references a custom Python pipeline class."
                                ),
                            )
                        )

    # --- B1.4: Dotted module.ClassName values anywhere in the config ---
    # Walk the top-level keys only (skip auto_map and custom_pipelines already handled)
    skip_keys = {_AUTO_MAP_KEY, _CUSTOM_PIPELINES_KEY, _TRUST_REMOTE_CODE_KEY}
    for top_key, top_val in parsed.items():
        if top_key in skip_keys:
            continue
        _walk_for_dotted_paths(top_val, top_key, findings, label)

    return findings


def scan_repo_configs(repo_root: Path) -> list[ConfigFinding]:
    """Scan all relevant config JSON files in a repository.

    Scans ``config.json`` and ``tokenizer_config.json``.

    Parameters
    ----------
    repo_root:
        Root directory of the model repository.

    Returns
    -------
    list[ConfigFinding]
        Combined findings from all scanned config files.
    """
    findings: list[ConfigFinding] = []

    for config_name in ("config.json", "tokenizer_config.json"):
        config_path = repo_root / config_name
        if config_path.is_file():
            findings.extend(
                scan_config_file(config_path, repo_root, config_file_name=config_name)
            )

    return findings


def find_bundled_code_files(repo_root: Path) -> list[Path]:
    """Return paths to bundled Python executable code files (B1).

    Finds files matching: modeling_*.py, configuration_*.py,
    tokenization_*.py and similar patterns alongside weight files.

    Parameters
    ----------
    repo_root:
        Root directory of the model repository.

    Returns
    -------
    list[Path]
        List of bundled Python file paths (absolute).
    """
    from model_scanner.analyzers.repository.py_ast_scanner import is_bundled_code_file

    bundled: list[Path] = []
    for py_file in repo_root.glob("*.py"):
        if is_bundled_code_file(py_file):
            bundled.append(py_file)
    return bundled


__all__ = [
    "ConfigFinding",
    "find_bundled_code_files",
    "scan_config_file",
    "scan_repo_configs",
]
