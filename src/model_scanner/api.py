"""
model_scanner.api
=================

High-level public programmatic Python API for the Static Security Scanner.

Provides simple top-level entry points for scanning AI models, exporting reports,
and running the Web UI server without manual pipeline orchestration.

SECURITY GUARANTEE
------------------
Zero dynamic execution: never imports ML frameworks or executes model code.
Delegates strictly to the frozen ScanOrchestrator and standard-library WebServer.

Design Reference
----------------
Section 29 of docs/implementation_design.md (Phase 10)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_scanner.core.engine import ScanOrchestrator
from model_scanner.core.models import (
    InputSourceType,
    ScanConfig,
    ScanRequest,
    ScanResult,
)
from model_scanner.presentation.json_export import JSONReportExporter
from model_scanner.presentation.web.server import WebServer


def _infer_input_source_type(target: str | Path) -> InputSourceType:
    """Infer the appropriate InputSourceType for a scan target."""
    if isinstance(target, str) and (
        target.startswith("http://")
        or target.startswith("https://")
        or target.startswith("hf://")
    ):
        return InputSourceType.PUBLIC_URL

    path = Path(target)
    if path.is_dir():
        return InputSourceType.LOCAL_DIRECTORY

    return InputSourceType.LOCAL_FILE


def scan(
    target: str | Path,
    config: ScanConfig | None = None,
    strict: bool = False,
) -> ScanResult:
    """Perform a static security scan on a target model file, directory, or public URL.

    Parameters
    ----------
    target:
        Path to a local model file or directory (str or Path), or a public/Hugging Face URL string.
    config:
        Optional ScanConfig instance for custom limits and thresholds.
    strict:
        Enable strict mode validation.

    Returns
    -------
    ScanResult
        Comprehensive scan result including per-artifact findings, risk scores, and summary.
    """
    input_source_type = _infer_input_source_type(target)
    effective_config = config or ScanConfig(strict_mode=strict)

    target_param: Path | str = (
        str(target)
        if input_source_type == InputSourceType.PUBLIC_URL
        else Path(target)
    )

    request = ScanRequest(
        target=target_param,
        input_source_type=input_source_type,
        config=effective_config,
        strict=strict,
    )

    orchestrator = ScanOrchestrator(config=effective_config)
    return orchestrator.scan(request)


def export_json(result: ScanResult, indent: int | None = 2) -> str:
    """Export a ScanResult into canonical JSON report format.

    Parameters
    ----------
    result:
        The ScanResult to export.
    indent:
        JSON indentation level (defaults to 2).

    Returns
    -------
    str
        JSON-formatted string representation of the scan report.
    """
    return JSONReportExporter.to_json(result, indent=indent)


def export_json_dict(result: ScanResult) -> dict[str, Any]:
    """Export a ScanResult into a Python dictionary.

    Parameters
    ----------
    result:
        The ScanResult to export.

    Returns
    -------
    dict[str, Any]
        Dictionary matching the standard JSON report schema.
    """
    return JSONReportExporter.to_dict(result)


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    background: bool = False,
) -> WebServer:
    """Initialize and optionally start the Web UI server.

    Parameters
    ----------
    host:
        Bind address (default: "127.0.0.1").
    port:
        Bind port (default: 8000). Use 0 for an ephemeral port.
    background:
        If True, starts the server in a background daemon thread and returns immediately.

    Returns
    -------
    WebServer
        The WebServer instance.
    """
    server = WebServer(host=host, port=port)
    if background:
        server.start(background=True)
    return server


__all__ = [
    "export_json",
    "export_json_dict",
    "scan",
    "serve",
]
