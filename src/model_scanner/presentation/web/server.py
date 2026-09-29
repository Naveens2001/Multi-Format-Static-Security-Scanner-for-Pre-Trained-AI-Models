"""
model_scanner.presentation.web.server
=====================================

Embedded HTTP server and REST API dispatcher for the Web UI (Phase 9).
Uses purely Python standard library (http.server) with zero third-party dependencies.
"""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from model_scanner.core.engine import ScanOrchestrator
from model_scanner.core.models import InputSourceType, ScanConfig, ScanRequest, ScanResult
from model_scanner.presentation.html_report import HTMLReportRenderer
from model_scanner.presentation.report_model import ScanReportBuilder
from model_scanner.presentation.web.executor import (
    BoundedScanExecutor,
    ScanJobStatus,
    ScanNotCancellableError,
    ScanQueueFullError,
)
from model_scanner.presentation.web.security import (
    PayloadTooLargeError,
    WebSecurityError,
    create_isolated_upload_dir,
    parse_multipart_upload,
)
from model_scanner.rules.catalog import RuleCatalog
from model_scanner.rules.remediation import RemediationRegistry

_STATIC_DIR = Path(__file__).parent / "static"


def _single_root(files: list[Path], upload_dir: Path) -> Path:
    """Return the folder to scan for a multi-file upload.

    When every uploaded file sits under one top-level directory - the usual
    case, because ``webkitRelativePath`` is prefixed with the folder the user
    picked - that directory is the scan target, so the artifact carries the
    real repository name instead of the scratch directory's random one.

    Parameters
    ----------
    files:
        Paths of the saved upload, all under ``upload_dir``.
    upload_dir:
        The isolated scratch directory for this upload.

    Returns
    -------
    Path
        The single common root if there is one, otherwise ``upload_dir``.
    """
    roots = set()
    for f in files:
        try:
            rel = f.relative_to(upload_dir)
        except ValueError:
            return upload_dir
        if len(rel.parts) < 2:
            return upload_dir
        roots.add(rel.parts[0])
    if len(roots) == 1:
        candidate = upload_dir / roots.pop()
        if candidate.is_dir():
            return candidate
    return upload_dir


class ModelScannerHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Model Scanner Web UI and REST API."""

    executor: BoundedScanExecutor
    orchestrator: ScanOrchestrator

    # Security headers
    server_version = "ModelScanner/2.0.0"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default stdout request logging during execution."""

    def _send_json_response(
        self,
        data: object,
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Send JSON response with UTF-8 encoding and security headers."""
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(
        self,
        message: str,
        error_code: str = "BAD_REQUEST",
        status: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> None:
        """Send structured JSON error."""
        self._send_json_response(
            {"error": error_code, "message": message},
            status=status,
        )

    def _send_queue_full(self, exc: ScanQueueFullError) -> None:
        """Send a 429 that tells the client how to recover."""
        self._send_json_response(
            {
                "error": "SCAN_QUEUE_FULL",
                "message": str(exc),
                "waiting": exc.waiting,
                "running": exc.running,
                "max_queued": exc.max_queued,
                "max_active": exc.max_active,
                "recovery": "cancel-waiting",
            },
            status=HTTPStatus.TOO_MANY_REQUESTS,
        )

    def do_DELETE(self) -> None:
        """Route DELETE requests. Cancelling a scan is the only one."""
        path = urlparse(self.path).path
        m = re.match(r"^/api/scans/([a-f0-9\-]+)$", path)
        if m:
            self._handle_cancel(m.group(1))
            return
        self._send_error_json(
            f"Endpoint {path!r} not found",
            error_code="ENDPOINT_NOT_FOUND",
            status=HTTPStatus.NOT_FOUND,
        )

    def _handle_cancel(self, scan_id: str) -> None:
        """Cancel one scan and report whether its worker slot was freed."""
        try:
            outcome = self.executor.cancel_scan(scan_id)
        except KeyError:
            self._send_error_json(
                f"Scan ID {scan_id!r} not found or expired",
                error_code="SCAN_NOT_FOUND",
                status=HTTPStatus.NOT_FOUND,
            )
            return
        except ScanNotCancellableError as exc:
            self._send_error_json(
                str(exc),
                error_code=exc.error_code,
                status=HTTPStatus.CONFLICT,
            )
            return

        outcome["capacity"] = self.executor.capacity()
        outcome["message"] = (
            "Scan detached. It was already running, so the worker keeps unwinding in "
            "the background and its report will be discarded; the slot frees when it "
            "stops."
            if outcome.get("was_running")
            else "Scan cancelled before it started. Its slot is free now."
        )
        self._send_json_response(outcome)

    def do_GET(self) -> None:
        """Route GET requests for static UI assets and REST APIs."""
        parsed = urlparse(self.path)
        path = parsed.path

        # 0. API: queue capacity snapshot
        if path == "/api/scans":
            cap = self.executor.capacity()
            cap["jobs"] = [
                j.to_summary()
                for j in sorted(self.executor.list_jobs(), key=lambda j: j.created_at)
            ]
            self._send_json_response(cap)
            return

        # 1. API: List rules
        if path == "/api/rules":
            rules = [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "severity": r.severity.value,
                    "category": r.category,
                    "description": r.description,
                    "remediation": r.remediation,
                    "affected_format": r.affected_format,
                    "cve_ids": list(r.cve_ids),
                    "cwe_ids": list(r.cwe_ids),
                    "owasp_category": r.owasp_category,
                    "mitre_technique": r.mitre_technique,
                }
                for r in RuleCatalog.list_all_rules()
            ]
            self._send_json_response({"rules": rules, "total": len(rules)})
            return

        # 2. API: Get specific rule
        m_rule = re.match(r"^/api/rules/([a-zA-Z0-9]+)$", path)
        if m_rule:
            rule_id = m_rule.group(1).upper()
            if RuleCatalog.has_rule(rule_id):
                r = RuleCatalog.get_rule(rule_id)
                self._send_json_response(
                    {
                        "rule_id": r.rule_id,
                        "name": r.name,
                        "severity": r.severity.value,
                        "category": r.category,
                        "description": r.description,
                        "detection_condition": r.detection_condition,
                        "remediation": r.remediation,
                        "affected_format": r.affected_format,
                        "cve_ids": list(r.cve_ids),
                        "cwe_ids": list(r.cwe_ids),
                        "owasp_category": r.owasp_category,
                        "mitre_technique": r.mitre_technique,
                        "migration_advice": RemediationRegistry.get_migration_advice(
                            r.affected_format.split(",")[0].strip()
                        ),
                    }
                )
            else:
                self._send_error_json(
                    f"Rule ID {rule_id!r} not found",
                    error_code="RULE_NOT_FOUND",
                    status=HTTPStatus.NOT_FOUND,
                )
            return

        # 3. API: Poll scan job status
        m_scan = re.match(r"^/api/scans/([a-f0-9\-]+)$", path)
        if m_scan:
            scan_id = m_scan.group(1)
            job = self.executor.get_job(scan_id)
            if job:
                self._send_json_response(job.to_dict())
            else:
                self._send_error_json(
                    f"Scan ID {scan_id!r} not found or expired",
                    error_code="SCAN_NOT_FOUND",
                    status=HTTPStatus.NOT_FOUND,
                )
            return

        # 4. API: Download canonical JSON report
        m_report = re.match(r"^/api/scans/([a-f0-9\-]+)/report$", path)
        if m_report:
            scan_id = m_report.group(1)
            job = self.executor.get_job(scan_id)
            if not job:
                self._send_error_json(
                    f"Scan ID {scan_id!r} not found",
                    error_code="SCAN_NOT_FOUND",
                    status=HTTPStatus.NOT_FOUND,
                )
                return

            if job.status != ScanJobStatus.COMPLETED or not job.result:
                self._send_error_json(
                    f"Scan {scan_id!r} has not completed successfully",
                    error_code="SCAN_NOT_READY",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            headers = {
                "Content-Disposition": f'attachment; filename="scan-report-{scan_id}.json"'
            }
            self._send_json_response(job.result, headers=headers)
            return

        # 5. API: Download HTML report
        m_html = re.match(r"^/api/scans/([a-f0-9\-]+)/report/html$", path)
        if m_html:
            scan_id = m_html.group(1)
            job = self.executor.get_job(scan_id)
            if not job:
                self._send_error_json(
                    f"Scan ID {scan_id!r} not found",
                    error_code="SCAN_NOT_FOUND",
                    status=HTTPStatus.NOT_FOUND,
                )
                return

            if job.status != ScanJobStatus.COMPLETED or not job.scan_result_obj:
                self._send_error_json(
                    f"Scan {scan_id!r} has not completed successfully",
                    error_code="SCAN_NOT_READY",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            try:
                scan_report = ScanReportBuilder.build(job.scan_result_obj)
                html_content = HTMLReportRenderer.render_scan(scan_report)
            except Exception as exc:
                self._send_error_json(
                    f"Failed to generate HTML report: {exc}",
                    error_code="HTML_RENDER_ERROR",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return

            body_bytes = html_content.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="scan-report-{scan_id}.html"',
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body_bytes)
            return

        # 6. Static asset delivery
        self._serve_static(path)

    def do_POST(self) -> None:
        """Route POST requests for local upload and Hugging Face scans."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/scans/upload":
            self._handle_local_upload()
        elif path == "/api/scans/hf":
            self._handle_hf_scan()
        elif path == "/api/scans/cancel-all":
            outcome = self.executor.cancel_all_waiting()
            outcome["message"] = (
                f"Cancelled {outcome['cancelled_count']} waiting "
                f"{'scan' if outcome['cancelled_count'] == 1 else 'scans'}."
            )
            self._send_json_response(outcome)
        elif re.match(r"^/api/scans/[a-f0-9\-]+/cancel$", path):
            self._handle_cancel(path.split("/")[3])
        else:
            self._send_error_json(
                f"Endpoint {path!r} not found",
                error_code="ENDPOINT_NOT_FOUND",
                status=HTTPStatus.NOT_FOUND,
            )

    def _handle_local_upload(self) -> None:
        """Handle multipart file upload for static scanning."""
        content_type = self.headers.get("Content-Type", "")
        content_length_str = self.headers.get("Content-Length", "0")

        try:
            content_length = int(content_length_str)
        except ValueError:
            self._send_error_json("Invalid Content-Length header")
            return

        if content_length <= 0:
            self._send_error_json("Empty upload request body")
            return

        upload_dir = create_isolated_upload_dir()

        try:
            files, fields = parse_multipart_upload(
                self.rfile,
                content_type,
                content_length,
                upload_dir,
            )

            if not files:
                upload_dir.rmdir()
                self._send_error_json("No model file uploaded in multipart request")
                return

            strict_mode = fields.get("strict_mode", "").lower() in ("true", "1", "yes")

            # A folder upload arrives as many parts whose filenames carry the
            # browser's webkitRelativePath, already rebuilt under upload_dir by
            # the multipart parser. Scanning the directory (rather than each
            # file alone) is what activates the repository analyzer, and with
            # it the SC-series supply-chain rules and the C001-C003
            # correlations - none of which a single-file scan can ever reach.
            if len(files) > 1:
                target_path = _single_root(files, upload_dir)
                target_name = target_path.name
                source_type = InputSourceType.LOCAL_DIRECTORY
            else:
                target_path = files[0]
                target_name = target_path.name
                source_type = InputSourceType.LOCAL_FILE

            job_scan_id = str(uuid.uuid4())

            def scan_task() -> ScanResult:
                req = ScanRequest(
                    target=target_path,
                    input_source_type=source_type,
                    config=ScanConfig(strict_mode=strict_mode),
                )
                return self.orchestrator.scan(req, scan_id=job_scan_id)

            scan_id = self.executor.submit_scan(
                target_name=target_name,
                scan_fn=scan_task,
                cleanup_paths=[upload_dir],
                preset_scan_id=job_scan_id,
            )

            self._send_json_response(
                {
                    "scan_id": scan_id,
                    "target_name": target_name,
                    "status": ScanJobStatus.QUEUED.value,
                    "strict_mode": strict_mode,
                    "file_count": len(files),
                    "mode": "directory" if len(files) > 1 else "file",
                },
                status=HTTPStatus.ACCEPTED,
            )

        except PayloadTooLargeError as exc:
            self._remove_dir_safe(upload_dir)
            self._send_error_json(
                str(exc),
                error_code="PAYLOAD_TOO_LARGE",
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        except ScanQueueFullError as exc:
            self._remove_dir_safe(upload_dir)
            self._send_queue_full(exc)
        except WebSecurityError as exc:
            self._remove_dir_safe(upload_dir)
            self._send_error_json(
                str(exc),
                error_code=exc.error_code,
                status=HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:
            self._remove_dir_safe(upload_dir)
            self._send_error_json(
                f"Upload failed: {exc}",
                error_code="UPLOAD_ERROR",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_hf_scan(self) -> None:
        """Handle Hugging Face URL scan submission."""
        content_length_str = self.headers.get("Content-Length", "0")
        try:
            content_length = int(content_length_str)
        except ValueError:
            self._send_error_json("Invalid Content-Length header")
            return

        if content_length <= 0:
            self._send_error_json("Empty request body")
            return

        try:
            body_bytes = self.rfile.read(content_length)
            payload: dict[str, object] = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            self._send_error_json("Malformed JSON payload")
            return

        raw_url = payload.get("url", "")
        url = str(raw_url).strip()
        if not url:
            self._send_error_json("Field 'url' is required")
            return

        if not (
            url.startswith("https://huggingface.co/")
            or url.startswith("http://huggingface.co/")
        ):
            self._send_error_json(
                "Only public Hugging Face URLs are supported (e.g. "
                "'https://huggingface.co/org/model')",
                error_code="INVALID_URL_PROVIDER",
            )
            return

        strict_mode = bool(payload.get("strict_mode", False))
        job_scan_id = str(uuid.uuid4())

        def scan_task() -> ScanResult:
            config = ScanConfig(strict_mode=strict_mode)
            req = ScanRequest(
                target=url,
                input_source_type=InputSourceType.PUBLIC_URL,
                config=config,
            )

            # Run the full orchestrator pipeline. This handler previously drove
            # acquisition and scan_artifact() by hand, which skipped Step 2b
            # (repository analysis) and Step 2c (cross-artifact correlation):
            # C001-C004 could never fire on a Hugging Face target, so the same
            # artifact produced different findings depending on how it arrived.
            # ScanOrchestrator._acquire_artifacts already routes PUBLIC_URL to
            # HuggingFaceAcquisition, so one call covers acquisition and scan.
            #
            # raise_on_acquisition_error keeps the pre-existing web contract: a
            # network failure, a gated repository or an oversized file must reach
            # the client as a job in ERROR state carrying the real error_code and
            # error_message, not as a COMPLETED job holding an empty ERROR report.
            return self.orchestrator.scan(req, raise_on_acquisition_error=True, scan_id=job_scan_id)

        try:
            scan_id = self.executor.submit_scan(
                target_name=url,
                scan_fn=scan_task,
                initial_status=ScanJobStatus.ACQUIRING,
                preset_scan_id=job_scan_id,
            )

            self._send_json_response(
                {
                    "scan_id": scan_id,
                    "target_name": url,
                    "status": ScanJobStatus.QUEUED.value,
                    "strict_mode": strict_mode,
                },
                status=HTTPStatus.ACCEPTED,
            )
        except ScanQueueFullError as exc:
            self._send_queue_full(exc)
        except Exception as exc:
            self._send_error_json(
                f"Failed to enqueue scan: {exc}",
                error_code="ENQUEUE_ERROR",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _serve_static(self, path: str) -> None:
        """Serve static files safely from the static/ directory."""
        if path in ("/", "/index.html"):
            target = _STATIC_DIR / "index.html"
        elif path.startswith("/static/"):
            clean_rel = unquote(path[len("/static/"):]).lstrip("/").replace("\\", "/")
            if ".." in clean_rel:
                self._send_error_json("Invalid static path", status=HTTPStatus.FORBIDDEN)
                return
            target = _STATIC_DIR / clean_rel
        else:
            clean_rel = unquote(path).lstrip("/").replace("\\", "/")
            if ".." in clean_rel:
                self._send_error_json("Invalid static path", status=HTTPStatus.FORBIDDEN)
                return
            target = _STATIC_DIR / clean_rel

        if not target.is_file():
            self._send_error_json(
                "Resource not found",
                error_code="NOT_FOUND",
                status=HTTPStatus.NOT_FOUND,
            )
            return

        content_type, _ = mimetypes.guess_type(str(target))
        if not content_type:
            if target.suffix == ".css":
                content_type = "text/css"
            elif target.suffix == ".js":
                content_type = "application/javascript"
            elif target.suffix == ".html":
                content_type = "text/html"
            else:
                content_type = "application/octet-stream"

        try:
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            self._send_error_json(
                f"Failed to read static asset: {exc}",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    @staticmethod
    def _remove_dir_safe(dir_path: Path) -> None:
        """Helper to remove upload directory safely on error."""
        try:
            shutil.rmtree(dir_path, ignore_errors=True)
        except OSError:
            pass


class WebServer:
    """Embedded standard-library HTTP Web Server for Model Scanner."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        executor: BoundedScanExecutor | None = None,
        orchestrator: ScanOrchestrator | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.executor = executor or BoundedScanExecutor()
        self.orchestrator = orchestrator or ScanOrchestrator()

        # Bind dependencies to handler class
        class CustomHandler(ModelScannerHTTPHandler):
            executor = self.executor
            orchestrator = self.orchestrator

        self._server = ThreadingHTTPServer((self.host, self.port), CustomHandler)
        self._thread: threading.Thread | None = None

    def start(self, background: bool = False) -> None:
        """Start the HTTP server."""
        if background:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="ModelScannerHTTPServer",
                daemon=True,
            )
            self._thread.start()
        else:
            self._server.serve_forever()

    def stop(self) -> None:
        """Stop the HTTP server and shut down background executor."""
        self._server.shutdown()
        self._server.server_close()
        self.executor.shutdown(wait=False)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def server_address(self) -> tuple[str, int]:
        """Return (host, port) tuple."""
        addr = self._server.server_address
        if isinstance(addr, tuple) and len(addr) >= 2:
            host = (
                addr[0].decode("utf-8")
                if isinstance(addr[0], (bytes, bytearray))
                else str(addr[0])
            )
            port = int(addr[1])
            return host, port
        return str(self.host), int(self.port)


__all__ = [
    "ModelScannerHTTPHandler",
    "WebServer",
]
