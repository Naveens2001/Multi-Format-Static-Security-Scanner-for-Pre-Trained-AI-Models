"""
model_scanner.__main__
======================

Command-line application entry point for ``python -m model_scanner`` and
the ``model-scanner`` console script.

Commands
--------
serve:
    Start the Model Scanner Web UI (Phase 9 embedded server).
    Default: http://127.0.0.1:8000

scan:
    Perform a static security scan on a file, directory, or URL.
    Outputs structured scan results and exits with standard ClamAV exit codes.

--version, -v:
    Display installed model-scanner version.

--help, -h:
    Display usage information and help.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from model_scanner import __version__
from model_scanner.api import export_json, scan, serve
from model_scanner.config.loader import load_config
from model_scanner.core.models import ScanConfig


def _create_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="model-scanner",
        description=(
            f"Static Security Scanner for Pre-Trained AI Models (v{__version__})\n"
            "Zero dynamic execution static security analysis for AI weights."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"model-scanner v{__version__}",
        help="Show program version and exit.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        description="Run 'model-scanner <command> --help' for details on each command.",
    )

    # Command 1: serve (Web UI)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the Model Scanner Web UI dashboard (Primary Interface).",
        description="Launch the embedded Model Scanner Web UI and REST API server.",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind the web server (default: 127.0.0.1).",
    )
    serve_parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port number to bind the web server (default: 8000).",
    )

    # Command 2: scan (CLI Scanner)
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan a local file, directory, or Hugging Face URL.",
        description="Execute static security inspection on AI model artifacts.",
    )
    scan_parser.add_argument(
        "target",
        type=str,
        help="Path to local file/directory or Hugging Face URL.",
    )
    scan_parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Enable strict mode (stricter security validation thresholds).",
    )
    scan_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Save canonical JSON report to specified file path.",
    )
    scan_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print raw JSON report to stdout (alias for --format json).",
    )
    scan_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default=None,
        help="Output format for stdout: 'text' (default) or 'json'.",
    )
    scan_parser.add_argument(
        "--include",
        type=str,
        default=None,
        metavar="PATTERN",
        help="Only scan files whose paths match this regex pattern.",
    )
    scan_parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        metavar="PATTERN",
        help="Skip files whose paths match this regex pattern.",
    )
    scan_parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        metavar="MB",
        help="Maximum file size to scan in megabytes (default: 1024 MB).",
    )
    scan_parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Download timeout in seconds for remote targets (default: 60).",
    )
    scan_parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to a JSON or TOML configuration file (overrides CLI flags).",
    )

    return parser


def _run_serve(host: str, port: int) -> int:
    """Launch the WebServer and block until KeyboardInterrupt."""
    server = serve(host=host, port=port, background=False)

    bound_host, bound_port = server.server_address
    url = f"http://{bound_host}:{bound_port}"

    print(
        f"\n============================================================\n"
        f" Model Scanner Web UI\n"
        f" Version: {__version__}\n"
        f" Dashboard URL: {url}\n"
        f"\n"
        f" Primary Interface is active.\n"
        f" Press Ctrl+C to stop the server.\n"
        f"============================================================\n",
        flush=True,
    )

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nStopping Model Scanner server...", flush=True)
    finally:
        server.stop()
        print("Server stopped cleanly.", flush=True)

    return 0


def _run_scan(
    target: str,
    strict: bool,
    output_path: str | None,
    print_json: bool,
    output_format: str | None = None,
    include_pattern: str | None = None,
    exclude_pattern: str | None = None,
    max_size_mb: int | None = None,
    timeout_sec: int | None = None,
    config_path: str | None = None,
) -> int:
    """Execute a scan and output findings."""
    if config_path is not None:
        try:
            cfg = load_config(config_path)
        except Exception as exc:
            print(f"Error loading config file {config_path!r}: {exc}", file=sys.stderr)
            return 2
    else:
        _defaults = ScanConfig()
        cfg = ScanConfig(
            strict_mode=strict,
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
            max_download_file_bytes=(
                max_size_mb * 1024 * 1024
                if max_size_mb is not None
                else _defaults.max_download_file_bytes
            ),
            download_timeout_sec=(
                timeout_sec
                if timeout_sec is not None
                else _defaults.download_timeout_sec
            ),
        )

    use_json = print_json or output_format == "json"
    print(f"Scanning target: {target} (strict_mode={cfg.strict_mode})...", file=sys.stderr)

    result = scan(target=target, config=cfg)
    json_output = export_json(result, indent=2)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json_output, encoding="utf-8")
        print(f"Report saved to: {out_p.resolve()}", file=sys.stderr)

    if use_json:
        print(json_output)
    else:
        # Print human-readable summary
        summary = result.summary
        print(
            f"\n--- Scan Summary ---\n"
            f"Total Artifacts     : {summary.total_artifacts}\n"
            f"Scanned Successfully: {summary.scanned_successfully}\n"
            f"Failed / Errors     : {summary.failed_artifacts}\n"
            f"Coverage Gaps       : {summary.coverage_gap_artifacts}\n"
            f"Unsupported         : {summary.unsupported_artifacts}\n"
            f"Max Risk Score      : {summary.max_risk_score}\n"
            f"Overall Outcome     : {summary.overall_outcome.value}\n"
            f"Exit Code           : {summary.exit_code}\n",
            file=sys.stderr,
        )

        for art in result.artifacts:
            band_val = art.risk_score.risk_band.value
            score_display = (
                "N/A" if band_val == "UNANALYZED" else str(art.risk_score.final_score)
            )
            print(
                f"Artifact: {art.name} [{art.detected_format.value}] -> "
                f"Outcome: {art.scan_outcome.value}, "
                f"Risk: {score_display} ({band_val}), "
                f"Findings: {len(art.findings)}",
                file=sys.stderr,
            )
            for f in art.findings:
                print(
                    f"  - [{f.severity.value}] {f.rule_id}: {f.rule_name} "
                    f"({f.description})",
                    file=sys.stderr,
                )

    return result.summary.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = _create_parser()

    if not args_list:
        parser.print_help()
        return 0

    args = parser.parse_args(args_list)

    if args.command == "serve":
        return _run_serve(host=args.host, port=args.port)

    if args.command == "scan":
        return _run_scan(
            target=args.target,
            strict=args.strict,
            output_path=args.output,
            print_json=args.json,
            output_format=args.format,
            include_pattern=args.include,
            exclude_pattern=args.exclude,
            max_size_mb=args.max_size,
            timeout_sec=args.timeout,
            config_path=args.config,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
