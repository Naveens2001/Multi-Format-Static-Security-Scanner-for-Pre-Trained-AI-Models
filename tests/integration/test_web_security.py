import time
import urllib.error
import urllib.request
from http import HTTPStatus

import pytest

from model_scanner.presentation.web.server import WebServer


@pytest.fixture(scope="module")
def web_server():
    server = WebServer(host="127.0.0.1", port=0)
    server.start(background=True)
    time.sleep(0.5) # Wait for server to start
    yield server
    server.stop()

def test_path_traversal_blocked(web_server):
    host, port = web_server.server_address
    url = f"http://{host}:{port}/%2fetc/passwd"
    try:
        urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        assert e.code == HTTPStatus.NOT_FOUND
    else:
        pytest.fail("Path traversal exploit succeeded; expected 404 Not Found")
