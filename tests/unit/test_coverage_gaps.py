"""
tests.unit.test_coverage_gaps
==============================

Targeted tests covering uncovered lines in executor.py, security.py,
professional_report.py, and report_model.py to bring total coverage to >=90%.
"""

from __future__ import annotations

import io
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from model_scanner.core.models import (
    ArtifactScanResult,
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    InputSourceType,
    ModelFormat,
    RiskScore,
    ScanOutcome,
    ScanResult,
    ScanSummary,
    Severity,
    risk_band_for_score,
)
from model_scanner.presentation.web.executor import (
    BoundedScanExecutor,
    ScanJob,
    ScanJobStatus,
    ScanNotCancellableError,
)
from model_scanner.presentation.web.security import (
    PayloadTooLargeError,
    WebSecurityError,
    parse_multipart_upload,
    sanitize_relative_path,
)

# ---------------------------------------------------------------------------
# shared factories
# ---------------------------------------------------------------------------

_ZERO_SHA256 = "a" * 64
_VALID_UUID = str(uuid.uuid4())


def _make_risk_score(final_score: int = 0) -> RiskScore:
    return RiskScore(
        raw_score=float(final_score),
        final_score=final_score,
        risk_band=risk_band_for_score(final_score),
        format_multiplier=1.0,
        obfuscation_multiplier=1.0,
        findings_weighted_sum=0.0,
    )


def _make_evidence(matched_value: str = "os.system") -> Evidence:
    return Evidence(
        coordinate_space=CoordinateSpace.ARTIFACT,
        byte_offset=0,
        byte_length=None,
        archive_member=None,
        matched_value=matched_value,
        opcode=None,
    )


def _make_finding(
    rule_id: str = "P101",
    confidence: Confidence = Confidence.HIGH,
) -> Finding:
    return Finding(
        finding_id=str(uuid.uuid4()),
        rule_id=rule_id,
        rule_name="Test Rule",
        severity=Severity.CRITICAL,
        confidence=confidence,
        category="code_execution",
        format=ModelFormat.RAW_PICKLE,
        artifact_sha256=_ZERO_SHA256,
        evidence=_make_evidence(),
        description="A dangerous global import was detected.",
        remediation="Do not deserialize this artifact.",
    )


def _make_summary(outcome: ScanOutcome = ScanOutcome.COMPLETE) -> ScanSummary:
    return ScanSummary(
        total_artifacts=1,
        scanned_successfully=1,
        failed_artifacts=0,
        unsupported_artifacts=0,
        findings_by_severity={"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0},
        artifacts_by_risk_band={"CLEAN": 1},
        max_risk_score=0,
        overall_outcome=outcome,
        exit_code=0,
    )


def _make_artifact(findings: tuple[Finding, ...] = ()) -> ArtifactScanResult:
    return ArtifactScanResult(
        artifact_id=str(uuid.uuid4()),
        name="model.pkl",
        source_origin="/data/model.pkl",
        input_source=InputSourceType.LOCAL_FILE,
        detected_format=ModelFormat.RAW_PICKLE,
        size_bytes=1024,
        sha256=_ZERO_SHA256,
        scan_outcome=ScanOutcome.COMPLETE,
        risk_score=_make_risk_score(0),
        findings=findings,
        warnings=(),
        errors=(),
        duration_ms=42,
    )


def _make_scan_result(findings: tuple[Finding, ...] = ()) -> ScanResult:
    return ScanResult(
        scan_id=_VALID_UUID,
        scanner_version="2.0.0",
        timestamp_utc="2026-08-21T12:00:00+00:00",
        target_request="/data/model.pkl",
        strict_mode=False,
        artifacts=(_make_artifact(findings),),
        summary=_make_summary(),
        total_duration_ms=100,
    )


def _noop_scan() -> ScanResult:
    return _make_scan_result()


def _make_boundary_payload(
    filename: str,
    file_bytes: bytes,
    fields: dict[str, str] | None = None,
) -> tuple[bytes, str]:
    boundary = "----TestBoundary"
    lines: list[bytes] = []
    if fields:
        for k, v in fields.items():
            lines.append(f"--{boundary}\r\n".encode())
            lines.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
            lines.append(f"{v}\r\n".encode())
    lines.append(f"--{boundary}\r\n".encode())
    lines.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    lines.append(b"Content-Type: application/octet-stream\r\n\r\n")
    lines.append(file_bytes)
    lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode())
    body = b"".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


# ---------------------------------------------------------------------------
# BoundedScanExecutor — executor.py
# ---------------------------------------------------------------------------

class TestScanNotCancellableError:
    def test_default_message(self) -> None:
        exc = ScanNotCancellableError()
        assert "already finished" in str(exc)
        assert exc.error_code == "SCAN_NOT_CANCELLABLE"

    def test_custom_message(self) -> None:
        exc = ScanNotCancellableError("custom msg")
        assert "custom msg" in str(exc)


class TestScanJobToSummary:
    def test_to_summary_keys(self) -> None:
        job = ScanJob(scan_id="s1", target_name="m.pkl", status=ScanJobStatus.QUEUED)
        summary = job.to_summary()
        assert summary["scan_id"] == "s1"
        assert summary["target_name"] == "m.pkl"
        assert summary["status"] == ScanJobStatus.QUEUED.value
        assert "created_at" in summary
        assert "cancellable" in summary
        assert summary["cancellable"] is True

    def test_to_summary_terminal_not_cancellable(self) -> None:
        job = ScanJob(scan_id="s2", target_name="m.pkl", status=ScanJobStatus.COMPLETED)
        assert job.to_summary()["cancellable"] is False


class TestBoundedScanExecutorCapacity:
    def test_capacity_empty_executor(self) -> None:
        ex = BoundedScanExecutor(max_active=2, max_queued=5)
        cap = ex.capacity()
        assert cap["waiting"] == 0
        assert cap["running"] == 0
        assert cap["max_queued"] == 5
        assert cap["max_active"] == 2
        assert cap["accepting"] is True
        assert "stored_jobs" in cap
        ex.shutdown(wait=False)


class TestBoundedScanExecutorListJobs:
    def test_list_jobs_empty(self) -> None:
        ex = BoundedScanExecutor()
        assert ex.list_jobs() == []
        ex.shutdown(wait=False)

    def test_list_jobs_returns_submitted(self) -> None:
        ex = BoundedScanExecutor()
        ex.submit_scan("target", _noop_scan)
        jobs = ex.list_jobs()
        assert len(jobs) >= 1
        ex.shutdown(wait=False)


class TestBoundedScanExecutorShutdown:
    def test_shutdown_wait_true(self) -> None:
        ex = BoundedScanExecutor()
        ex.shutdown(wait=True)

    def test_shutdown_no_wait(self) -> None:
        ex = BoundedScanExecutor()
        ex.submit_scan("target", _noop_scan)
        ex.shutdown(wait=False)


class TestBoundedScanExecutorCancelScan:
    def test_cancel_unknown_scan_raises_key_error(self) -> None:
        ex = BoundedScanExecutor()
        with pytest.raises(KeyError):
            ex.cancel_scan("nonexistent-id")
        ex.shutdown(wait=False)

    def test_cancel_terminal_raises_not_cancellable(self) -> None:
        ex = BoundedScanExecutor()
        scan_id = ex.submit_scan("target", _noop_scan)
        for _ in range(50):
            time.sleep(0.05)
            job = ex.get_job(scan_id)
            if job and job.is_terminal:
                break
        with pytest.raises(ScanNotCancellableError):
            ex.cancel_scan(scan_id)
        ex.shutdown(wait=False)

    def test_cancel_queued_job_frees_slot(self) -> None:
        ex = BoundedScanExecutor(max_active=1, max_queued=5)

        def slow() -> ScanResult:
            time.sleep(2.0)
            return _noop_scan()

        ex.submit_scan("slow_job", slow)
        queued_id = ex.submit_scan("queued_job", _noop_scan)
        time.sleep(0.1)

        result = ex.cancel_scan(queued_id)
        assert result["scan_id"] == queued_id
        assert result["freed_slot"] is True
        ex.shutdown(wait=False)

    def test_cancel_all_waiting(self) -> None:
        ex = BoundedScanExecutor(max_active=1, max_queued=5)

        def slow() -> ScanResult:
            time.sleep(2.0)
            return _noop_scan()

        ex.submit_scan("slow_active", slow)
        ex.submit_scan("queued_1", _noop_scan)
        ex.submit_scan("queued_2", _noop_scan)
        time.sleep(0.1)

        result = ex.cancel_all_waiting()
        assert "cancelled" in result
        assert "cancelled_count" in result
        assert "capacity" in result
        ex.shutdown(wait=False)


class TestBoundedScanExecutorReapLocked:
    def test_queue_timeout_reaps_waiting_job(self) -> None:
        ex = BoundedScanExecutor(max_active=1, max_queued=5, max_queue_wait_seconds=1)

        def slow() -> ScanResult:
            time.sleep(2.0)
            return _noop_scan()

        ex.submit_scan("slow_active", slow)
        queued_id = ex.submit_scan("will_timeout", _noop_scan)

        job = ex.get_job(queued_id)
        assert job is not None
        job.created_at = time.time() - 10.0  # backdate beyond TTL

        ex.capacity()  # triggers _reap_locked

        job = ex.get_job(queued_id)
        assert job is not None
        assert job.status == ScanJobStatus.ERROR
        assert job.error_code == "SCAN_QUEUE_TIMEOUT"
        ex.shutdown(wait=False)

    def test_scan_timeout_marks_running_job_error(self) -> None:
        ex = BoundedScanExecutor(max_active=1, max_queued=5, max_scan_seconds=1)

        def slow() -> ScanResult:
            time.sleep(5.0)
            return _noop_scan()

        scan_id = ex.submit_scan("slow_running", slow)
        for _ in range(30):
            time.sleep(0.05)
            job = ex.get_job(scan_id)
            if job and job.status == ScanJobStatus.SCANNING:
                break

        job = ex.get_job(scan_id)
        assert job is not None
        assert job.started_at is not None
        job.started_at = time.time() - 100.0  # backdate

        ex.capacity()  # triggers _reap_locked

        job = ex.get_job(scan_id)
        assert job is not None
        assert job.status == ScanJobStatus.ERROR
        assert job.error_code == "SCAN_TIMEOUT"
        ex.shutdown(wait=False)


class TestBoundedScanExecutorCleanupExpired:
    def test_overflow_eviction(self) -> None:
        ex = BoundedScanExecutor(max_stored_results=2, max_active=4)

        ids = []
        for i in range(4):
            sid = ex.submit_scan(f"job_{i}", _noop_scan)
            ids.append(sid)

        for sid in ids:
            for _ in range(30):
                time.sleep(0.05)
                job = ex.get_job(sid)
                if job and job.is_terminal:
                    break

        # Trigger cleanup by submitting another job
        ex.submit_scan("trigger", _noop_scan)
        assert len(ex.list_jobs()) <= 4
        ex.shutdown(wait=False)


class TestRemovePath:
    def test_remove_file(self, tmp_path: Path) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"data")
        BoundedScanExecutor._remove_path(f)
        assert not f.exists()

    def test_remove_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "subdir"
        d.mkdir()
        (d / "f.bin").write_bytes(b"data")
        BoundedScanExecutor._remove_path(d)
        assert not d.exists()

    def test_remove_nonexistent_does_not_raise(self, tmp_path: Path) -> None:
        BoundedScanExecutor._remove_path(tmp_path / "ghost.bin")

    def test_remove_oserror_is_silenced(self, tmp_path: Path) -> None:
        with patch("pathlib.Path.is_file", side_effect=OSError("boom")):
            BoundedScanExecutor._remove_path(tmp_path / "x.bin")


# ---------------------------------------------------------------------------
# security.py — sanitize_relative_path
# ---------------------------------------------------------------------------

class TestSanitizeRelativePath:
    def test_simple_path(self) -> None:
        result = sanitize_relative_path("model/config.json")
        assert result == "model/config.json"

    def test_windows_backslash_converted(self) -> None:
        result = sanitize_relative_path("model\\config.json")
        assert "config.json" in result

    def test_drive_letter_stripped(self) -> None:
        result = sanitize_relative_path("C:/model/config.json")
        assert "C:" not in result
        assert "config.json" in result

    def test_path_traversal_raises(self) -> None:
        with pytest.raises(WebSecurityError) as exc_info:
            sanitize_relative_path("../../etc/passwd")
        assert exc_info.value.error_code == "PATH_TRAVERSAL_BLOCKED"

    def test_path_too_deep_raises(self) -> None:
        deep = "/".join(["a"] * 20)
        with pytest.raises(WebSecurityError) as exc_info:
            sanitize_relative_path(deep)
        assert exc_info.value.error_code == "PATH_TOO_DEEP"

    def test_empty_after_sanitization_gets_generated_name(self) -> None:
        result = sanitize_relative_path("...")
        assert result.endswith(".bin")
        assert "model_" in result

    def test_dot_segments_stripped(self) -> None:
        result = sanitize_relative_path("./model/./config.json")
        assert result.startswith("model")


# ---------------------------------------------------------------------------
# security.py — parse_multipart_upload edge cases
# ---------------------------------------------------------------------------

class TestParseMultipartUploadEdgeCases:
    def test_invalid_content_type_raises(self, tmp_path: Path) -> None:
        body = b"dummy"
        with pytest.raises(WebSecurityError) as exc_info:
            parse_multipart_upload(
                io.BytesIO(body),
                "application/json",
                len(body),
                tmp_path,
            )
        assert exc_info.value.error_code == "INVALID_CONTENT_TYPE"

    def test_empty_content_type_raises(self, tmp_path: Path) -> None:
        body = b"dummy"
        with pytest.raises(WebSecurityError):
            parse_multipart_upload(io.BytesIO(body), "", len(body), tmp_path)

    def test_stream_exceeds_budget_during_read(self, tmp_path: Path) -> None:
        # content_length fits, but actual stream read exceeds budget
        boundary = "----B"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="f.bin"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
            + "A" * 200
            + f"\r\n--{boundary}--\r\n"
        ).encode()

        # Pass a small content_length that doesn't exceed budget, but read more
        class TrickyStream(io.RawIOBase):
            def read(self, n: int = -1) -> bytes:
                return body

            def readinto(self, b: Any) -> int:
                data = body[: len(b)]
                b[: len(data)] = data
                return len(data)

        with pytest.raises(PayloadTooLargeError):
            parse_multipart_upload(
                io.BytesIO(body),
                f"multipart/form-data; boundary={boundary}",
                len(body),
                tmp_path,
                max_upload_bytes=100,  # body is much larger
            )

    def test_folder_upload_relative_path(self, tmp_path: Path) -> None:
        body, ctype = _make_boundary_payload("model/weights.bin", b"\x00" * 16)
        files, _ = parse_multipart_upload(
            io.BytesIO(body), ctype, len(body), tmp_path, max_upload_bytes=10 * 1024
        )
        assert len(files) == 1
        assert files[0].name == "weights.bin"

    def test_form_field_only(self, tmp_path: Path) -> None:
        boundary = "----B"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="mode"\r\n\r\n'
            "strict\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        files, fields = parse_multipart_upload(
            io.BytesIO(body),
            f"multipart/form-data; boundary={boundary}",
            len(body),
            tmp_path,
            max_upload_bytes=10 * 1024,
        )
        assert files == []
        assert fields.get("mode") == "strict"


# ---------------------------------------------------------------------------
# professional_report.py — ProfessionalJSONExporter.export()
# ---------------------------------------------------------------------------

class TestProfessionalJSONExporterExport:
    def _make_report(self) -> Any:
        from model_scanner.presentation.report_model import ReportBuilder
        return ReportBuilder.build(_make_scan_result())

    def test_export_to_file_path(self, tmp_path: Path) -> None:
        from model_scanner.presentation.professional_report import ProfessionalJSONExporter
        report = self._make_report()
        dest = tmp_path / "report.json"
        content = ProfessionalJSONExporter.export(report, dest)
        assert dest.exists()
        assert "scan_id" in content

    def test_export_to_text_stream(self) -> None:
        from model_scanner.presentation.professional_report import ProfessionalJSONExporter
        report = self._make_report()
        buf = io.StringIO()
        content = ProfessionalJSONExporter.export(report, buf)
        assert "scan_id" in content
        assert "scan_id" in buf.getvalue()

    def test_export_invalid_destination_raises(self) -> None:
        from model_scanner.presentation.professional_report import ProfessionalJSONExporter
        report = self._make_report()
        with pytest.raises(TypeError, match="Destination must be"):
            ProfessionalJSONExporter.export(report, 42)  # type: ignore[arg-type]

    def test_export_creates_parent_directory(self, tmp_path: Path) -> None:
        from model_scanner.presentation.professional_report import ProfessionalJSONExporter
        report = self._make_report()
        dest = tmp_path / "new_dir" / "report.json"
        ProfessionalJSONExporter.export(report, str(dest))
        assert dest.exists()


# ---------------------------------------------------------------------------
# report_model.py — private helper coverage
# ---------------------------------------------------------------------------

class TestReportModelHelpers:
    def test_build_clean_report(self) -> None:
        from model_scanner.presentation.report_model import ReportBuilder
        report = ReportBuilder.build(_make_scan_result())
        assert report.scan_id
        assert report.scan_outcome == "COMPLETE"

    def test_confidence_rationale_high(self) -> None:
        from model_scanner.presentation.report_model import _confidence_rationale
        finding = _make_finding(confidence=Confidence.HIGH)
        rationale = _confidence_rationale(finding)
        assert "Deterministic" in rationale

    def test_confidence_rationale_medium(self) -> None:
        from model_scanner.presentation.report_model import _confidence_rationale
        finding = _make_finding(confidence=Confidence.MEDIUM)
        rationale = _confidence_rationale(finding)
        assert "likely" in rationale

    def test_confidence_rationale_low(self) -> None:
        from model_scanner.presentation.report_model import _confidence_rationale
        finding = _make_finding(confidence=Confidence.LOW)
        rationale = _confidence_rationale(finding)
        assert "Heuristic" in rationale

    def test_catalog_cwe_ids_prefers_finding(self) -> None:
        from model_scanner.presentation.report_model import _catalog_cwe_ids
        result = _catalog_cwe_ids("P101", ("CWE-502",))
        assert "CWE-502" in result

    def test_catalog_cwe_ids_from_catalog_when_empty(self) -> None:
        from model_scanner.presentation.report_model import _catalog_cwe_ids
        result = _catalog_cwe_ids("P101", ())
        assert isinstance(result, tuple)

    def test_catalog_owasp_prefers_finding(self) -> None:
        from model_scanner.presentation.report_model import _catalog_owasp
        result = _catalog_owasp("P101", "LLM-Custom")
        assert result == "LLM-Custom"

    def test_catalog_owasp_fallback_unknown_rule(self) -> None:
        from model_scanner.presentation.report_model import _catalog_owasp
        result = _catalog_owasp("UNKNOWN_RULE_9999", None)
        assert result == "Not directly mapped"

    def test_catalog_owasp_unmapped_catalog_rule(self) -> None:
        # ST202 now has owasp_category assigned (D22 coverage fix)
        from model_scanner.presentation.report_model import _catalog_owasp
        result = _catalog_owasp("ST202", None)
        assert result == "ML06:2023 AI Supply Chain Attacks"

    def test_catalog_owasp_mapped_catalog_rule(self) -> None:
        # P101 maps to ML06:2023
        from model_scanner.presentation.report_model import _catalog_owasp
        result = _catalog_owasp("P101", None)
        assert result == "ML06:2023 AI Supply Chain Attacks"

    def test_catalog_mitre_prefers_finding(self) -> None:
        from model_scanner.presentation.report_model import _catalog_mitre
        result = _catalog_mitre("P101", "T1059")
        assert result == "T1059"

    def test_catalog_mitre_fallback_unknown_rule(self) -> None:
        from model_scanner.presentation.report_model import _catalog_mitre
        result = _catalog_mitre("UNKNOWN_RULE_9999", None)
        assert result == "Not mapped"

    def test_catalog_impact_unknown_rule(self) -> None:
        from model_scanner.presentation.report_model import _catalog_impact
        result = _catalog_impact("UNKNOWN_RULE_9999", "")
        assert "not available" in result.lower()

    def test_catalog_security_issue_unknown_rule(self) -> None:
        from model_scanner.presentation.report_model import _catalog_security_issue
        result = _catalog_security_issue("UNKNOWN_RULE_9999", "")
        assert "Not available" in result

    def test_catalog_impact_diverging_description(self) -> None:
        from model_scanner.presentation.report_model import _catalog_impact
        # Impact is now category-based rather than description-based; for P101
        # (code_execution) the result describes the consequence, not the observation.
        result = _catalog_impact("P101", "Custom description that differs.")
        assert isinstance(result, str)
        assert len(result) > 0
        # Category-based impact for code_execution mentions exploitation/execution
        assert any(
            kw in result.lower()
            for kw in ("exploitation", "execution", "code", "privilege")
        )

    def test_catalog_security_issue_with_known_rule(self) -> None:
        from model_scanner.presentation.report_model import _catalog_security_issue
        # P101 is a known rule; if description matches catalog, use catalog text
        result = _catalog_security_issue("P101", "")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Additional executor coverage — cancel a running scan (was_running=True)
# ---------------------------------------------------------------------------

class TestCancelRunningJob:
    def test_cancel_running_job_returns_was_running_true(self) -> None:
        ex = BoundedScanExecutor(max_active=1, max_queued=5)

        started = [False]

        def slow_scan() -> ScanResult:
            started[0] = True
            time.sleep(3.0)
            return _noop_scan()

        scan_id = ex.submit_scan("slow", slow_scan)

        # Wait until actually running
        for _ in range(40):
            time.sleep(0.05)
            job = ex.get_job(scan_id)
            if job and job.status == ScanJobStatus.SCANNING:
                break

        result = ex.cancel_scan(scan_id)
        assert result["was_running"] is True
        assert result["freed_slot"] is False
        ex.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Additional executor coverage — submit with cleanup_paths hitting remove
# ---------------------------------------------------------------------------

class TestSubmitWithCleanupPaths:
    def test_submit_with_cleanup_paths(self, tmp_path: Path) -> None:
        ex = BoundedScanExecutor()
        upload_dir = tmp_path / "upload"
        upload_dir.mkdir()
        (upload_dir / "file.bin").write_bytes(b"data")

        scan_id = ex.submit_scan(
            "target", _noop_scan, cleanup_paths=[upload_dir]
        )
        # Wait for completion
        for _ in range(40):
            time.sleep(0.05)
            job = ex.get_job(scan_id)
            if job and job.is_terminal:
                break
        # Cleanup should have been called — directory removed
        assert not upload_dir.exists()
        ex.shutdown(wait=False)


# ---------------------------------------------------------------------------
# pdf_report.py — is_available() when weasyprint is mocked present
# ---------------------------------------------------------------------------

class TestPDFReportRendererAvailability:
    def test_is_available_when_weasyprint_present(self) -> None:
        import sys
        from unittest.mock import MagicMock

        from model_scanner.presentation.pdf_report import PDFReportRenderer

        fake_wp = MagicMock()
        original = sys.modules.get("weasyprint")
        sys.modules["weasyprint"] = fake_wp
        try:
            assert PDFReportRenderer.is_available() is True
        finally:
            if original is None:
                sys.modules.pop("weasyprint", None)
            else:
                sys.modules["weasyprint"] = original

    def test_is_available_when_weasyprint_absent(self) -> None:
        import sys

        from model_scanner.presentation.pdf_report import PDFReportRenderer

        original = sys.modules.pop("weasyprint", None)
        try:
            assert PDFReportRenderer.is_available() is False
        finally:
            if original is not None:
                sys.modules["weasyprint"] = original


# ---------------------------------------------------------------------------
# Additional security.py — multipart parsing string payload paths (lines 235-242)
# ---------------------------------------------------------------------------

class TestParseMultipartStringPayload:
    def test_multipart_with_bytes_payload_saved(self, tmp_path: Path) -> None:
        body, ctype = _make_boundary_payload("weights.bin", b"\xde\xad\xbe\xef" * 4)
        files, _ = parse_multipart_upload(
            io.BytesIO(body), ctype, len(body), tmp_path, max_upload_bytes=1024 * 1024
        )
        assert len(files) == 1
        assert files[0].read_bytes() == b"\xde\xad\xbe\xef" * 4


# ---------------------------------------------------------------------------
# pdf_report.py — render() happy path (mocked weasyprint)
# ---------------------------------------------------------------------------

class TestPDFReportRendererRender:
    def test_render_calls_weasyprint(self, tmp_path: Path) -> None:
        import sys
        from unittest.mock import MagicMock

        from model_scanner.presentation.pdf_report import PDFReportRenderer
        from model_scanner.presentation.report_model import ReportBuilder

        report = ReportBuilder.build(_make_scan_result())

        fake_wp = MagicMock()
        fake_html_instance = MagicMock()
        fake_wp.HTML.return_value = fake_html_instance

        original = sys.modules.get("weasyprint")
        sys.modules["weasyprint"] = fake_wp
        try:
            dest = tmp_path / "out.pdf"
            PDFReportRenderer.render(report, dest)
            fake_wp.HTML.assert_called_once()
            fake_html_instance.write_pdf.assert_called_once()
        finally:
            if original is None:
                sys.modules.pop("weasyprint", None)
            else:
                sys.modules["weasyprint"] = original

    def test_render_raises_import_error_when_absent(self, tmp_path: Path) -> None:
        import sys

        from model_scanner.presentation.pdf_report import PDFReportRenderer
        from model_scanner.presentation.report_model import ReportBuilder

        report = ReportBuilder.build(_make_scan_result())

        original = sys.modules.pop("weasyprint", None)
        try:
            with pytest.raises(ImportError, match="weasyprint"):
                PDFReportRenderer.render(report, tmp_path / "out.pdf")
        finally:
            if original is not None:
                sys.modules["weasyprint"] = original


# ---------------------------------------------------------------------------
# executor.py — submit when pool is already shut down (RuntimeError path)
# ---------------------------------------------------------------------------

class TestSubmitAfterShutdown:
    def test_submit_after_shutdown_raises_runtime_error(self) -> None:
        ex = BoundedScanExecutor()
        ex.shutdown(wait=True)
        with pytest.raises(RuntimeError):
            ex.submit_scan("target", _noop_scan)


# ---------------------------------------------------------------------------
# executor.py — submit with cleanup_paths when queue is full (path removal)
# ---------------------------------------------------------------------------

class TestSubmitQueueFullCleansUpPaths:
    def test_cleanup_paths_removed_on_queue_full(self, tmp_path: Path) -> None:
        from model_scanner.presentation.web.executor import ScanQueueFullError

        ex = BoundedScanExecutor(max_active=1, max_queued=1)

        def slow() -> ScanResult:
            time.sleep(2.0)
            return _noop_scan()

        ex.submit_scan("slow", slow)  # active
        ex.submit_scan("queued", _noop_scan)  # queued (fills the 1-slot queue)
        time.sleep(0.05)

        upload_dir = tmp_path / "upload"
        upload_dir.mkdir()
        (upload_dir / "file.bin").write_bytes(b"data")

        with pytest.raises(ScanQueueFullError):
            ex.submit_scan("overflow", _noop_scan, cleanup_paths=[upload_dir])

        assert not upload_dir.exists()
        ex.shutdown(wait=False)
