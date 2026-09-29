import json
import time
import urllib.request
from http import HTTPStatus
from unittest.mock import patch

import pytest

from model_scanner.core.engine import ScanOrchestrator
from model_scanner.core.models import InputSourceType, ScanOutcome, ScanRequest
from model_scanner.presentation.web.server import WebServer


def test_zero_artifact_engine_outcome():
    """Verify that a scan producing 0 artifacts is marked as UNSUPPORTED, not COMPLETE."""
    orchestrator = ScanOrchestrator()
    request = ScanRequest(
        target="https://huggingface.co/MustEr/catch_me_if_you_can",
        input_source_type=InputSourceType.PUBLIC_URL
    )

    with patch(
        "model_scanner.acquisition.huggingface.HuggingFaceAcquisition.acquire",
        return_value=[]
    ):
        result = orchestrator.scan(request)

    assert result.summary.overall_outcome == ScanOutcome.UNSUPPORTED
    assert result.summary.total_artifacts == 0
    assert result.summary.unsupported_artifacts == 0
    assert result.summary.exit_code == 0
    assert len(result.artifacts) == 0


@pytest.fixture(scope="module")
def web_server():
    server = WebServer(host="127.0.0.1", port=0)
    server.start(background=True)
    time.sleep(0.5)
    yield server
    server.stop()


def test_zero_artifact_web_api(web_server):
    """Verify that the Web API propagates the UNSUPPORTED outcome for zero artifacts."""
    host, port = web_server.server_address
    url = f"http://{host}:{port}/api/scans/hf"

    data = json.dumps({"url": "https://huggingface.co/MustEr/catch_me_if_you_can"}).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    with patch(
        "model_scanner.acquisition.huggingface.HuggingFaceAcquisition.acquire",
        return_value=[]
    ):
        with urllib.request.urlopen(req) as response:
            assert response.status == HTTPStatus.ACCEPTED
            scan_data = json.loads(response.read().decode())
            scan_id = scan_data["scan_id"]

        time.sleep(0.5)

        status_url = f"http://{host}:{port}/api/scans/{scan_id}"
        with urllib.request.urlopen(status_url) as status_resp:
            assert status_resp.status == HTTPStatus.OK
            result_data = json.loads(status_resp.read().decode())

    assert result_data["status"] == "COMPLETED"
    summary = result_data["result"]["summary"]
    assert summary["overall_outcome"] == "UNSUPPORTED"
    assert summary["total_artifacts"] == 0
    assert summary["unsupported_artifacts"] == 0
