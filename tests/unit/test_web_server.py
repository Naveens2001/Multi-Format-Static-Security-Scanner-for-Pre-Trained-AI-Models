"""
tests.unit.test_web_server
==========================

Unit tests for Phase 9 Web Server, REST API, Bounded Scan Executor, and Security Controls.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from model_scanner.core.engine import ScanOrchestrator
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


def _build_multipart_payload(
    filename: str,
    file_bytes: bytes,
    fields: dict[str, str] | None = None,
) -> tuple[bytes, str]:
    """Construct a standard multipart/form-data byte payload with boundary."""
    boundary = "----ModelScannerBoundary7MA4YWxkTrZu0gW"
    lines: list[bytes] = []

    if fields:
        for k, v in fields.items():
            lines.append(f"--{boundary}\r\n".encode("latin-1"))
            lines.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("latin-1"))
            lines.append(f"{v}\r\n".encode("latin-1"))

    lines.append(f"--{boundary}\r\n".encode("latin-1"))
    lines.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("latin-1")
    )
    lines.append(b"Content-Type: application/octet-stream\r\n\r\n")
    lines.append(file_bytes)
    lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode("latin-1"))

    body = b"".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


# ---------------------------------------------------------------------------
# Test 1: Static Asset Delivery
# ---------------------------------------------------------------------------


def test_static_asset_serving(running_server: tuple[WebServer, str]) -> None:
    """Verify GET / serves index.html and static assets return valid MIME types."""
    _, base_url = running_server

    # 1. Root index.html
    req = urllib.request.Request(f"{base_url}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert "text/html" in resp.headers.get("Content-Type", "")
        content = resp.read().decode("utf-8")
        assert "AI Model Security Scanner" in content

    # 2. CSS
    req_css = urllib.request.Request(f"{base_url}/static/css/styles.css")
    with urllib.request.urlopen(req_css) as resp:
        assert resp.status == 200
        assert "text/css" in resp.headers.get("Content-Type", "")

    # 3. JavaScript
    req_js = urllib.request.Request(f"{base_url}/static/js/app.js")
    with urllib.request.urlopen(req_js) as resp:
        assert resp.status == 200
        assert "javascript" in resp.headers.get("Content-Type", "")


# ---------------------------------------------------------------------------
# Test 2: Rule Catalog Endpoints
# ---------------------------------------------------------------------------


def test_rule_catalog_api_endpoints(running_server: tuple[WebServer, str]) -> None:
    """Verify /api/rules and /api/rules/{rule_id} return accurate metadata."""
    _, base_url = running_server

    # 1. List all rules (86 prior + MS101 = 87)
    req = urllib.request.Request(f"{base_url}/api/rules")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["total"] == 87
        assert len(data["rules"]) == 87

        # owasp_category and mitre_technique must be present in every list entry
        rules_by_id = {r["rule_id"]: r for r in data["rules"]}
        assert "owasp_category" in rules_by_id["P101"], "list endpoint missing owasp_category"
        assert "mitre_technique" in rules_by_id["P101"], "list endpoint missing mitre_technique"

        # ML06-mapped rule
        assert rules_by_id["P101"]["owasp_category"] == "ML06:2023 AI Supply Chain Attacks"
        # ST202 now has OWASP mapping (D22 coverage fix)
        assert rules_by_id["ST202"]["owasp_category"] == "ML06:2023 AI Supply Chain Attacks"

    # 2. Specific rule lookup
    req_p101 = urllib.request.Request(f"{base_url}/api/rules/P101")
    with urllib.request.urlopen(req_p101) as resp:
        assert resp.status == 200
        rule_data = json.loads(resp.read().decode("utf-8"))
        assert rule_data["rule_id"] == "P101"
        assert rule_data["name"] == "Dangerous Global Import"
        assert rule_data["cve_ids"] == []  # no verified CVE assigned to P101
        assert rule_data["owasp_category"] == "ML06:2023 AI Supply Chain Attacks"

    # 3. Non-existent rule returns 404
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/api/rules/INVALID999")
    assert exc_info.value.code == 404


# ---------------------------------------------------------------------------
# Test 3: Local Upload Scan Flow (Clean SafeTensors)
# ---------------------------------------------------------------------------


def test_local_upload_clean_safetensors(running_server: tuple[WebServer, str]) -> None:
    """Verify uploading a clean SafeTensors model processes to COMPLETED with score 0."""
    _, base_url = running_server

    # Minimal valid SafeTensors file
    header_dict = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    header_json = json.dumps(header_dict).encode("utf-8")
    header_len = len(header_json)
    safetensors_bytes = header_len.to_bytes(8, "little") + header_json + b"\x00\x00\x80\x3f"

    body, ctype = _build_multipart_payload("clean_model.safetensors", safetensors_bytes)
    req = urllib.request.Request(
        f"{base_url}/api/scans/upload",
        data=body,
        headers={"Content-Type": ctype},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 202
        res_data = json.loads(resp.read().decode("utf-8"))
        scan_id = res_data["scan_id"]
        assert res_data["status"] == "QUEUED"

    # Poll for completion
    completed_job: dict[str, Any] | None = None
    for _ in range(30):
        time.sleep(0.1)
        with urllib.request.urlopen(f"{base_url}/api/scans/{scan_id}") as poll_resp:
            job_data = json.loads(poll_resp.read().decode("utf-8"))
            if job_data["status"] == "COMPLETED":
                completed_job = job_data
                break

    assert completed_job is not None, "Scan did not reach COMPLETED state in time"
    result = completed_job["result"]
    assert result["summary"]["max_risk_score"] == 0
    assert result["summary"]["overall_outcome"] == "COMPLETE"
    assert result["artifacts"][0]["detected_format"] == "SAFETENSORS"


# ---------------------------------------------------------------------------
# Test 4: Local Upload Scan Flow (Malicious Pickle RCE)
# ---------------------------------------------------------------------------


def test_local_upload_malicious_pickle(running_server: tuple[WebServer, str]) -> None:
    """Verify uploading a malicious Pickle RCE model flags CRITICAL risk and P101 finding."""
    _, base_url = running_server

    # Pickle RCE payload: os.system('id')
    pickle_bytes = b"\x80\x03cposix\nsystem\n(S'id'\ntR."
    body, ctype = _build_multipart_payload("malicious_weights.pkl", pickle_bytes)

    req = urllib.request.Request(
        f"{base_url}/api/scans/upload",
        data=body,
        headers={"Content-Type": ctype},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 202
        scan_id = json.loads(resp.read().decode("utf-8"))["scan_id"]

    # Poll for completion
    completed_job: dict[str, Any] | None = None
    for _ in range(30):
        time.sleep(0.1)
        with urllib.request.urlopen(f"{base_url}/api/scans/{scan_id}") as poll_resp:
            job_data = json.loads(poll_resp.read().decode("utf-8"))
            if job_data["status"] == "COMPLETED":
                completed_job = job_data
                break

    assert completed_job is not None
    result = completed_job["result"]
    assert result["summary"]["max_risk_score"] >= 90
    assert result["summary"]["overall_outcome"] == "WARNING"

    findings = result["artifacts"][0]["findings"]
    rule_ids = [f["rule_id"] for f in findings]
    assert "P101" in rule_ids


# ---------------------------------------------------------------------------
# Test 5: Report Download Endpoint
# ---------------------------------------------------------------------------


def test_download_json_report(running_server: tuple[WebServer, str]) -> None:
    """Verify GET /api/scans/{scan_id}/report returns canonical JSON download."""
    _, base_url = running_server

    pickle_bytes = b"\x80\x04\x95\x05\x00\x00\x00\x00\x00\x00\x00K*."
    body, ctype = _build_multipart_payload("scalar.pkl", pickle_bytes)

    req = urllib.request.Request(
        f"{base_url}/api/scans/upload",
        data=body,
        headers={"Content-Type": ctype},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        scan_id = json.loads(resp.read().decode("utf-8"))["scan_id"]

    # Wait for completion
    for _ in range(30):
        time.sleep(0.1)
        with urllib.request.urlopen(f"{base_url}/api/scans/{scan_id}") as poll_resp:
            if json.loads(poll_resp.read().decode("utf-8"))["status"] == "COMPLETED":
                break

    report_req = urllib.request.Request(f"{base_url}/api/scans/{scan_id}/report")
    with urllib.request.urlopen(report_req) as rep_resp:
        assert rep_resp.status == 200
        assert "application/json" in rep_resp.headers.get("Content-Type", "")
        assert f'filename="scan-report-{scan_id}.json"' in rep_resp.headers.get(
            "Content-Disposition", ""
        )
        report_data = json.loads(rep_resp.read().decode("utf-8"))
        assert "scan_id" in report_data
        assert "summary" in report_data


# ---------------------------------------------------------------------------
# Test 6: Hugging Face Endpoint Validation and Flow
# ---------------------------------------------------------------------------


def test_hf_url_scan_endpoint_success(running_server: tuple[WebServer, str]) -> None:
    """Verify /api/scans/hf acquires and completes scan with mocked HuggingFaceAcquisition."""
    import hashlib
    import io
    from unittest.mock import patch

    from model_scanner.core.models import Artifact, InputSourceType

    _, base_url = running_server

    # Synthetic clean SafeTensors artifact in memory
    header_dict = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    header_json = json.dumps(header_dict).encode("utf-8")
    header_len = len(header_json)
    safetensors_bytes = header_len.to_bytes(8, "little") + header_json + b"\x00\x00\x80\x3f"

    mock_artifact = Artifact(
        artifact_id="test-hf-art-1234",
        source_path="https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors",
        input_source=InputSourceType.PUBLIC_URL,
        size_bytes=len(safetensors_bytes),
        sha256=hashlib.sha256(safetensors_bytes).hexdigest(),
        stream=io.BytesIO(safetensors_bytes),
    )

    with patch(
        "model_scanner.acquisition.huggingface.HuggingFaceAcquisition.acquire",
        return_value=[mock_artifact],
    ):
        payload = json.dumps(
            {"url": "https://huggingface.co/bert-base-uncased"}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/scans/hf",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 202
            data = json.loads(resp.read().decode("utf-8"))
            scan_id = data["scan_id"]

        # Poll for completion
        completed_job: dict[str, Any] | None = None
        for _ in range(30):
            time.sleep(0.1)
            with urllib.request.urlopen(f"{base_url}/api/scans/{scan_id}") as poll_resp:
                job_data = json.loads(poll_resp.read().decode("utf-8"))
                if job_data["status"] == "COMPLETED":
                    completed_job = job_data
                    break

        assert completed_job is not None, "HF scan did not reach COMPLETED state"
        result = completed_job["result"]
        assert result["summary"]["max_risk_score"] == 0
        assert result["summary"]["overall_outcome"] == "COMPLETE"
        assert len(result["artifacts"]) == 1
        assert result["artifacts"][0]["detected_format"] == "SAFETENSORS"


def test_hf_url_scan_acquisition_error(running_server: tuple[WebServer, str]) -> None:
    """Verify acquisition failure transitions job to ERROR without crashing server."""
    from unittest.mock import patch

    from model_scanner.core.exceptions import AcquisitionError

    _, base_url = running_server

    with patch(
        "model_scanner.acquisition.huggingface.HuggingFaceAcquisition.acquire",
        side_effect=AcquisitionError("Connection timed out querying HF Hub API"),
    ):
        payload = json.dumps(
            {"url": "https://huggingface.co/org/failing-model"}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/scans/hf",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 202
            scan_id = json.loads(resp.read().decode("utf-8"))["scan_id"]

        # Poll for error state
        error_job: dict[str, Any] | None = None
        for _ in range(30):
            time.sleep(0.1)
            with urllib.request.urlopen(f"{base_url}/api/scans/{scan_id}") as poll_resp:
                job_data = json.loads(poll_resp.read().decode("utf-8"))
                if job_data["status"] == "ERROR":
                    error_job = job_data
                    break

        assert error_job is not None, "HF scan did not reach ERROR state"
        assert "Connection timed out" in str(error_job["error_message"])


def test_hf_url_scan_invalid_provider(running_server: tuple[WebServer, str]) -> None:
    """Verify non-HuggingFace URLs are rejected with HTTP 400."""
    _, base_url = running_server

    bad_payload = json.dumps(
        {"url": "https://malicious-site.example.com/model.bin"}
    ).encode("utf-8")
    bad_req = urllib.request.Request(
        f"{base_url}/api/scans/hf",
        data=bad_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(bad_req)
    assert exc_info.value.code == 400


# ---------------------------------------------------------------------------
# Test 7: Bounded Queue Rejection (HTTP 429)
# ---------------------------------------------------------------------------


def test_bounded_scan_executor_queue_limit() -> None:
    """Verify BoundedScanExecutor rejects submissions exceeding max_queued with 429."""
    executor = BoundedScanExecutor(max_active=1, max_queued=2)

    def slow_task() -> Any:
        time.sleep(0.5)
        return ScanOrchestrator()

    # Job 1: Active
    s1 = executor.submit_scan("job1", slow_task)
    # Job 2: Queued (1/2)
    s2 = executor.submit_scan("job2", slow_task)
    # Job 3: Queued (2/2)
    s3 = executor.submit_scan("job3", slow_task)

    assert s1 and s2 and s3

    # Job 4: Must raise ScanQueueFullError (capacity exceeded)
    with pytest.raises(ScanQueueFullError):
        executor.submit_scan("job4", slow_task)

    executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Test 8: Path Sanitization and Traversal Guards
# ---------------------------------------------------------------------------


def test_filename_sanitization() -> None:
    """Verify path traversal sequences and special characters are stripped."""
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\Windows\\System32\\model.pkl") == "model.pkl"
    assert sanitize_filename("my model (1).safetensors") == "my_model__1_.safetensors"
    assert sanitize_filename("..") != ".."
    assert sanitize_filename("") != ""


# ---------------------------------------------------------------------------
# Test 9: Result Store TTL Eviction
# ---------------------------------------------------------------------------


def test_job_ttl_eviction() -> None:
    """Verify jobs older than TTL are automatically evicted."""
    executor = BoundedScanExecutor(ttl_seconds=0.1)
    scan_id = executor.submit_scan("test_ttl", lambda: pytest.fail("unused"))  # type: ignore[arg-type]

    job = executor.get_job(scan_id)
    assert job is not None
    job.status = ScanJobStatus.COMPLETED
    job.created_at = time.time() - 1.0  # 1 second ago (> 0.1s TTL)

    # Trigger maintenance cleanup on next submit
    executor.submit_scan("dummy", lambda: pytest.fail("unused"))  # type: ignore[arg-type]
    assert executor.get_job(scan_id) is None
    executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Test 10: Direct Multipart Parser and Isolated Directory Helper
# ---------------------------------------------------------------------------


def test_isolated_upload_directory_lifecycle(tmp_path: Path) -> None:
    """Verify isolated upload directory creation and path isolation."""
    upload_dir = create_isolated_upload_dir(base_dir=tmp_path)
    assert upload_dir.exists()
    assert upload_dir.is_dir()
    assert str(tmp_path) in str(upload_dir)


def test_parse_multipart_direct(tmp_path: Path) -> None:
    """Verify direct parsing of multipart upload payload."""
    import io

    file_content = b"header_data_bytes_test"
    body, ctype = _build_multipart_payload(
        "test_weights.bin",
        file_content,
        fields={"strict_mode": "true"},
    )

    stream = io.BytesIO(body)
    files, fields = parse_multipart_upload(
        stream,
        ctype,
        len(body),
        tmp_path,
        max_upload_bytes=1024 * 1024,
    )

    assert len(files) == 1
    assert files[0].name == "test_weights.bin"
    assert files[0].read_bytes() == file_content
    assert fields.get("strict_mode") == "true"


def test_parse_multipart_payload_too_large(tmp_path: Path) -> None:
    """Verify parse_multipart_upload raises PayloadTooLargeError when exceeding budget."""
    import io

    body, ctype = _build_multipart_payload("large.bin", b"A" * 100)
    stream = io.BytesIO(body)

    with pytest.raises(PayloadTooLargeError):
        parse_multipart_upload(
            stream,
            ctype,
            len(body),
            tmp_path,
            max_upload_bytes=50,  # 50 bytes ceiling (< 100 bytes)
        )


def test_local_upload_legacy_pytorch_is_analyzed(
    running_server: tuple[WebServer, str],
) -> None:
    """Verify uploading a legacy non-ZIP PyTorch .bin is analyzed, not skipped."""
    _, base_url = running_server

    # Synthetic legacy PyTorch non-ZIP structure (pickle header + raw floats)
    legacy_bytes = (
        b"\x80\x02}q\x00(X\x06\x00\x00\x00weightq\x01u."
        b"\x00\x00\x80?\x00\x00\x00@\xcd\xcc\x8c?RAW_TENSOR_DATA"
    )
    body, ctype = _build_multipart_payload("pytorch_model.bin", legacy_bytes)

    req = urllib.request.Request(
        f"{base_url}/api/scans/upload",
        data=body,
        headers={"Content-Type": ctype},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 202
        scan_id = json.loads(resp.read().decode("utf-8"))["scan_id"]

    completed_job: dict[str, Any] | None = None
    for _ in range(30):
        time.sleep(0.1)
        with urllib.request.urlopen(f"{base_url}/api/scans/{scan_id}") as poll_resp:
            job_data = json.loads(poll_resp.read().decode("utf-8"))
            if job_data["status"] == "COMPLETED":
                completed_job = job_data
                break

    assert completed_job is not None
    result = completed_job["result"]
    artifact_res = result["artifacts"][0]

    assert artifact_res["scan_outcome"] == "COMPLETE"
    assert artifact_res["detected_format"] == "LEGACY_PYTORCH"
    assert artifact_res["risk_score"]["final_score"] == 0
    assert artifact_res["risk_score"]["risk_band"] == "CLEAN"
    assert len(artifact_res["findings"]) == 0

    # No UNSUPPORTED_FORMAT warning: the artifact was actually inspected.
    warning_codes = [w.get("warning_code", "") for w in artifact_res.get("warnings", [])]
    assert "UNSUPPORTED_FORMAT" not in warning_codes

    # And no false-positive structural anomalies from the raw tensor payload.
    assert artifact_res["errors"] == []


# ---------------------------------------------------------------------------
# Test 11: GET /api/scans — capacity snapshot with job list
# ---------------------------------------------------------------------------


def test_get_api_scans_capacity_snapshot(running_server: tuple[WebServer, str]) -> None:
    """GET /api/scans returns capacity info and a list of recent jobs."""
    _, base_url = running_server

    req = urllib.request.Request(f"{base_url}/api/scans")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "waiting" in data
        assert "running" in data
        assert "jobs" in data
        assert isinstance(data["jobs"], list)


# ---------------------------------------------------------------------------
# Test 12: DELETE /api/scans/{id} — cancel a queued scan
# ---------------------------------------------------------------------------


def test_delete_cancel_queued_scan(running_server: tuple[WebServer, str]) -> None:
    """DELETE /api/scans/{id} cancels a queued scan and returns 200."""
    _, base_url = running_server

    # Upload a scan first so we have a real scan_id
    pickle_bytes = b"\x80\x03cposix\nsystem\n(S'id'\ntR."
    body, ctype = _build_multipart_payload("test.pkl", pickle_bytes)
    req = urllib.request.Request(
        f"{base_url}/api/scans/upload",
        data=body,
        headers={"Content-Type": ctype},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        scan_id = json.loads(resp.read().decode("utf-8"))["scan_id"]

    # The scan may complete before we cancel — either outcome is acceptable
    del_req = urllib.request.Request(
        f"{base_url}/api/scans/{scan_id}",
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(del_req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "scan_id" in data
    except urllib.error.HTTPError as e:
        # 409 CONFLICT = scan already terminal (completed before cancel reached it)
        assert e.code == 409


def test_delete_cancel_nonexistent_scan(running_server: tuple[WebServer, str]) -> None:
    """DELETE /api/scans/{id} for an unknown ID returns 404."""
    _, base_url = running_server

    del_req = urllib.request.Request(
        f"{base_url}/api/scans/00000000-0000-0000-0000-000000000000",
        method="DELETE",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(del_req)
    assert exc_info.value.code == 404


# ---------------------------------------------------------------------------
# Test 13: GET /api/scans/{id}/report error paths
# ---------------------------------------------------------------------------


def test_get_report_unknown_scan_id(running_server: tuple[WebServer, str]) -> None:
    """GET /api/scans/{id}/report returns 404 for unknown scan IDs."""
    _, base_url = running_server

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/api/scans/deadbeef-0000-0000-0000-000000000000/report")
    assert exc_info.value.code == 404


def test_get_html_report_unknown_scan_id(running_server: tuple[WebServer, str]) -> None:
    """GET /api/scans/{id}/report/html returns 404 for unknown scan IDs."""
    _, base_url = running_server

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(
            f"{base_url}/api/scans/deadbeef-0000-0000-0000-000000000000/report/html"
        )
    assert exc_info.value.code == 404


def test_get_html_report_success(running_server: tuple[WebServer, str]) -> None:
    """GET /api/scans/{id}/report/html returns HTML for a completed scan."""
    _, base_url = running_server

    # Upload and wait for completion
    pickle_bytes = b"\x80\x04\x95\x05\x00\x00\x00\x00\x00\x00\x00K*."
    body, ctype = _build_multipart_payload("scalar.pkl", pickle_bytes)
    req = urllib.request.Request(
        f"{base_url}/api/scans/upload",
        data=body,
        headers={"Content-Type": ctype},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        scan_id = json.loads(resp.read().decode("utf-8"))["scan_id"]

    for _ in range(40):
        time.sleep(0.1)
        with urllib.request.urlopen(f"{base_url}/api/scans/{scan_id}") as poll_resp:
            if json.loads(poll_resp.read().decode("utf-8"))["status"] == "COMPLETED":
                break

    with urllib.request.urlopen(f"{base_url}/api/scans/{scan_id}/report/html") as resp:
        assert resp.status == 200
        assert "text/html" in resp.headers.get("Content-Type", "")
        content = resp.read().decode("utf-8")
        assert "<html" in content.lower()


# ---------------------------------------------------------------------------
# Test 14: 404 for unknown GET routes
# ---------------------------------------------------------------------------


def test_get_unknown_route_returns_404(running_server: tuple[WebServer, str]) -> None:
    """Unknown routes return 404 with a JSON body."""
    _, base_url = running_server

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/nonexistent/path")
    assert exc_info.value.code == 404
