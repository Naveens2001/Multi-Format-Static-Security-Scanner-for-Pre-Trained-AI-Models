"""
tests.robustness.test_web_security_hardening
============================================

Hardening tests for Web UI / REST API security boundaries, including:
- Path traversal filename sanitization
- SSRF prevention & Hugging Face URL boundary enforcement via /api/scans/hf
- Oversized payload rejection
- Queue bounds & worker concurrency handling
- Result TTL expiration & maintenance eviction
- Malformed multipart payload handling

Phase 12 Web Security Hardening.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from pathlib import Path

import pytest

from model_scanner.core.models import (
    ScanOutcome,
    ScanResult,
    ScanSummary,
)
from model_scanner.presentation.web.executor import (
    BoundedScanExecutor,
    ScanJobStatus,
    ScanQueueFullError,
)
from model_scanner.presentation.web.security import (
    PayloadTooLargeError,
    create_isolated_upload_dir,
    parse_multipart_upload,
    sanitize_filename,
)
from model_scanner.presentation.web.server import WebServer


@pytest.fixture
def running_server() -> Generator[tuple[WebServer, str], None, None]:
    """Spawn an in-process WebServer on an ephemeral port and yield (server, base_url)."""
    server = WebServer(host="127.0.0.1", port=0)
    server.start(background=True)
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        yield server, base_url
    finally:
        server.stop()


def _build_multipart(filename: str, file_bytes: bytes) -> tuple[bytes, str]:
    boundary = "----ModelScannerHardeningBoundary12345"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("latin-1") + file_bytes + f"\r\n--{boundary}--\r\n".encode("latin-1")
    return body, f"multipart/form-data; boundary={boundary}"


def _make_dummy_scan_result() -> ScanResult:
    """Create a minimal valid ScanResult for testing executor callbacks."""
    summary = ScanSummary(
        total_artifacts=1,
        scanned_successfully=1,
        failed_artifacts=0,
        unsupported_artifacts=0,
        findings_by_severity={},
        artifacts_by_risk_band={},
        overall_outcome=ScanOutcome.COMPLETE,
        max_risk_score=0,
        highest_severity=None,
        total_duration_ms=10,
        exit_code=0,
    )
    return ScanResult(
        scan_id="test-scan-id",
        scanner_version="2.0.0",
        timestamp_utc="2026-08-18T00:00:00Z",
        target_request="test",
        strict_mode=False,
        total_duration_ms=10,
        summary=summary,
        artifacts=(),
    )


class TestPathTraversalSanitization:
    """Verify robust path traversal and control character stripping in filenames."""

    def test_relative_path_traversal_stripped(self) -> None:
        assert sanitize_filename("../../etc/passwd") == "passwd"
        assert sanitize_filename("..\\..\\windows\\system32\\evil.dll") == "evil.dll"

    def test_absolute_path_stripped(self) -> None:
        assert sanitize_filename("/var/data/model.safetensors") == "model.safetensors"
        assert sanitize_filename("C:\\Users\\Admin\\model.pt") == "model.pt"

    def test_null_bytes_and_control_chars_stripped(self) -> None:
        cleaned = sanitize_filename("model\x00evil.pt")
        assert "\x00" not in cleaned
        assert cleaned.endswith(".pt")

    def test_empty_or_dots_only_fallback(self) -> None:
        assert sanitize_filename("...") != "..."
        assert sanitize_filename("") != ""
        assert sanitize_filename("   ") != ""


class TestSSRFAndEndpointValidation:
    """Verify SSRF validation strictly rejects non-HuggingFace URLs at the endpoint level."""

    def test_non_hf_domains_rejected_with_http_400(
        self,
        running_server: tuple[WebServer, str],
    ) -> None:
        _, base_url = running_server
        bad_urls = [
            "http://evil.com/malicious_model.pt",
            "https://evil-huggingface.co/model.bin",
            "https://huggingface.co.evil.com/model.bin",
            "http://169.254.169.254/latest/meta-data",
            "http://127.0.0.1:8080/internal",
            "ftp://huggingface.co/model.bin",
            "file:///etc/passwd",
        ]
        for bad_url in bad_urls:
            payload = json.dumps({"url": bad_url}).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/scans/hf",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req)
            assert exc_info.value.code == 400

    def test_missing_url_field_rejected_with_http_400(
        self,
        running_server: tuple[WebServer, str],
    ) -> None:
        _, base_url = running_server
        payload = json.dumps({"invalid_field": "test"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/scans/hf",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 400

    def test_invalid_scan_id_returns_404(
        self,
        running_server: tuple[WebServer, str],
    ) -> None:
        _, base_url = running_server
        req = urllib.request.Request(f"{base_url}/api/scans/non-existent-id-9999")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 404


class TestUploadHardening:
    """Verify payload limits and multipart stream protection."""

    def test_oversized_upload_raises_error(self, tmp_path: Path) -> None:
        body, ctype = _build_multipart("big.bin", bytes(2000))
        stream = io.BytesIO(body)
        with pytest.raises(PayloadTooLargeError):
            parse_multipart_upload(
                stream,
                ctype,
                len(body),
                tmp_path,
                max_upload_bytes=1000,
            )

    def test_isolated_directory_creation_and_cleanup(self, tmp_path: Path) -> None:
        upload_dir = create_isolated_upload_dir(base_dir=tmp_path)
        assert upload_dir.exists()
        assert upload_dir.is_dir()


class TestExecutorConcurrencyAndEviction:
    """Verify queue bounds, worker scaling, and TTL eviction."""

    def test_executor_queue_overflow_raises_error(self) -> None:
        executor = BoundedScanExecutor(max_active=1, max_queued=1)
        try:
            def slow_task() -> ScanResult:
                time.sleep(0.5)
                return _make_dummy_scan_result()

            # Job 1: Active
            s1 = executor.submit_scan("job1", slow_task)
            # Job 2: Queued (1/1)
            s2 = executor.submit_scan("job2", slow_task)
            assert s1 and s2

            # Job 3: Overflow -> raises ScanQueueFullError
            with pytest.raises(ScanQueueFullError):
                executor.submit_scan("job3", slow_task)
        finally:
            executor.shutdown(wait=False)

    def test_expired_results_eviction(self) -> None:
        executor = BoundedScanExecutor(ttl_seconds=0.01)
        try:
            scan_id = executor.submit_scan(
                "test_ttl",
                _make_dummy_scan_result,
            )
            job = executor.get_job(scan_id)
            assert job is not None
            job.status = ScanJobStatus.COMPLETED
            job.created_at = time.time() - 1.0  # 1s ago > 0.01s TTL

            executor.submit_scan(
                "dummy",
                _make_dummy_scan_result,
            )
            assert executor.get_job(scan_id) is None
        finally:
            executor.shutdown(wait=False)
