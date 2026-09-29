"""
tests.unit.test_acquisition
===========================

Unit tests for Phase 3 input acquisition subsystem:
  - Streaming SHA-256 fingerprinting (utils/hashing.py)
  - Local file acquisition (acquisition/local.py)
  - Local directory recursive acquisition (acquisition/local.py)
  - Public URL acquisition with SSRF guards and SafeRedirectHandler (acquisition/url.py)
  - Hugging Face acquisition adapter (acquisition/huggingface.py)
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model_scanner.acquisition.base import InputAcquisitionProvider
from model_scanner.acquisition.huggingface import (
    HuggingFaceAcquisition,
    _extract_repo_id,
    _filter_model_candidates,
    _is_hf_direct_file,
    _is_hf_repo,
    _is_suspicious_extensionless,
    is_hf_url,
)
from model_scanner.acquisition.local import (
    LocalDirectoryAcquisition,
    LocalFileAcquisition,
)
from model_scanner.acquisition.url import (
    PublicUrlAcquisition,
    SafeRedirectHandler,
    _is_private_address,
    _validate_url,
)
from model_scanner.core.exceptions import (
    AcquisitionError,
    InputError,
    ResourceLimitError,
)
from model_scanner.core.models import (
    Artifact,
    InputSourceType,
    ScanConfig,
    ScanRequest,
)
from model_scanner.utils.hashing import (
    sha256_incremental,
    sha256_of_bytes,
    sha256_of_file,
    sha256_of_stream,
)

# ---------------------------------------------------------------------------
# 1. SHA-256 Fingerprinting Tests
# ---------------------------------------------------------------------------


class TestHashingUtils:
    """Test SHA-256 fingerprinting functions."""

    def test_sha256_of_bytes(self) -> None:
        data = b"hello world security scanner"
        expected = hashlib.sha256(data).hexdigest()
        digest = sha256_of_bytes(data)
        assert digest == expected
        assert len(digest) == 64
        assert digest.islower()

    def test_sha256_of_file(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.bin"
        data = b"A" * 100_000
        test_file.write_bytes(data)

        expected = hashlib.sha256(data).hexdigest()
        digest = sha256_of_file(test_file)
        assert digest == expected
        assert len(digest) == 64

    def test_sha256_of_stream(self) -> None:
        data = b"model stream data 12345"
        stream = io.BytesIO(data)
        expected = hashlib.sha256(data).hexdigest()
        digest = sha256_of_stream(stream)
        assert digest == expected

    def test_sha256_incremental(self) -> None:
        h = sha256_incremental()
        chunk1 = b"part 1: "
        chunk2 = b"part 2."
        h.update(chunk1)
        h.update(chunk2)
        expected = hashlib.sha256(chunk1 + chunk2).hexdigest()
        assert h.hexdigest() == expected


# ---------------------------------------------------------------------------
# 2. Local File Acquisition Tests
# ---------------------------------------------------------------------------


class TestLocalFileAcquisition:
    """Test LocalFileAcquisition provider."""

    def test_valid_file_acquisition(self, tmp_path: Path) -> None:
        test_file = tmp_path / "model.safetensors"
        content = b"fake safetensors content"
        test_file.write_bytes(content)

        req = ScanRequest(
            target=str(test_file),
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        provider = LocalFileAcquisition()
        artifacts = provider.acquire(req)

        assert len(artifacts) == 1
        art = artifacts[0]
        assert isinstance(art, Artifact)
        assert art.input_source == InputSourceType.LOCAL_FILE
        assert art.size_bytes == len(content)
        assert art.sha256 == hashlib.sha256(content).hexdigest()
        assert art.local_path == test_file.resolve()

    def test_missing_file_raises_input_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.pt"
        req = ScanRequest(
            target=str(missing),
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        provider = LocalFileAcquisition()
        with pytest.raises(InputError) as exc_info:
            provider.acquire(req)
        assert exc_info.value.error_code == "FILE_NOT_FOUND"

    def test_directory_target_raises_input_error(self, tmp_path: Path) -> None:
        req = ScanRequest(
            target=str(tmp_path),
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        provider = LocalFileAcquisition()
        with pytest.raises(InputError) as exc_info:
            provider.acquire(req)
        assert exc_info.value.error_code == "NOT_A_FILE"

    def test_oversized_file_raises_resource_limit(self, tmp_path: Path) -> None:
        test_file = tmp_path / "large.bin"
        test_file.write_bytes(b"X" * 1024)

        config = ScanConfig(max_download_file_bytes=512)
        req = ScanRequest(
            target=str(test_file),
            input_source_type=InputSourceType.LOCAL_FILE,
            config=config,
        )
        provider = LocalFileAcquisition()
        with pytest.raises(ResourceLimitError) as exc_info:
            provider.acquire(req)
        assert exc_info.value.limit_name == "max_download_file_bytes"
        assert exc_info.value.actual_value == 1024

    def test_non_str_path_target_raises_input_error(self) -> None:
        req = ScanRequest(
            target=12345,  # type: ignore[arg-type]  # not str or Path
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        provider = LocalFileAcquisition()
        with pytest.raises(InputError):
            provider.acquire(req)


# ---------------------------------------------------------------------------
# 3. Local Directory Acquisition Tests
# ---------------------------------------------------------------------------


class TestLocalDirectoryAcquisition:
    """Test LocalDirectoryAcquisition provider."""

    def test_recursive_discovery(self, tmp_path: Path) -> None:
        (tmp_path / "root_model.pt").write_bytes(b"root pt")
        sub1 = tmp_path / "subdir1"
        sub1.mkdir()
        (sub1 / "model.safetensors").write_bytes(b"sub safetensors")
        (sub1 / "notes.txt").write_bytes(b"text file - should be ignored")

        sub2 = tmp_path / "subdir2" / "nested"
        sub2.mkdir(parents=True)
        (sub2 / "weights.gguf").write_bytes(b"nested gguf")

        provider = LocalDirectoryAcquisition()
        req = ScanRequest(
            target=str(tmp_path),
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        artifacts = provider.acquire(req)

        assert len(artifacts) == 3
        discovered_names = [Path(a.source_path).name for a in artifacts]
        assert "root_model.pt" in discovered_names
        assert "model.safetensors" in discovered_names
        assert "weights.gguf" in discovered_names
        assert "notes.txt" not in discovered_names

    def test_hidden_directories_skipped(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "hidden.pkl").write_bytes(b"git pkl")

        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "env.bin").write_bytes(b"venv bin")

        normal_dir = tmp_path / "models"
        normal_dir.mkdir()
        (normal_dir / "valid.pkl").write_bytes(b"valid pkl")

        provider = LocalDirectoryAcquisition()
        req = ScanRequest(
            target=str(tmp_path),
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        artifacts = provider.acquire(req)

        assert len(artifacts) == 1
        assert Path(artifacts[0].source_path).name == "valid.pkl"

    def test_include_exclude_filters(self, tmp_path: Path) -> None:
        (tmp_path / "model_v1.pt").write_bytes(b"pt1")
        (tmp_path / "model_v2.pt").write_bytes(b"pt2")
        (tmp_path / "test_model.pt").write_bytes(b"pt_test")

        config = ScanConfig(
            include_pattern=r"model_v\d",
            exclude_pattern=r"v2",
        )
        provider = LocalDirectoryAcquisition()
        req = ScanRequest(
            target=str(tmp_path),
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
            config=config,
        )
        artifacts = provider.acquire(req)

        assert len(artifacts) == 1
        assert Path(artifacts[0].source_path).name == "model_v1.pt"

    def test_deterministic_ordering(self, tmp_path: Path) -> None:
        (tmp_path / "c.pt").write_bytes(b"c")
        (tmp_path / "a.pt").write_bytes(b"a")
        (tmp_path / "b.pt").write_bytes(b"b")

        provider = LocalDirectoryAcquisition()
        req = ScanRequest(
            target=str(tmp_path),
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        artifacts = provider.acquire(req)
        names = [Path(a.source_path).name for a in artifacts]
        assert names == ["a.pt", "b.pt", "c.pt"]

    def test_nonexistent_directory_raises_input_error(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "does_not_exist"
        provider = LocalDirectoryAcquisition()
        req = ScanRequest(
            target=str(missing_dir),
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        with pytest.raises(InputError) as exc_info:
            provider.acquire(req)
        assert exc_info.value.error_code == "DIR_NOT_FOUND"


    def test_non_str_path_target_raises_input_error(self) -> None:
        req = ScanRequest(
            target=12345,  # type: ignore[arg-type]
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        provider = LocalDirectoryAcquisition()
        with pytest.raises(InputError):
            provider.acquire(req)

    def test_file_as_directory_target_raises_input_error(self, tmp_path: Path) -> None:
        file_path = tmp_path / "model.safetensors"
        file_path.write_bytes(b"FAKE")
        req = ScanRequest(
            target=str(file_path),
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        provider = LocalDirectoryAcquisition()
        with pytest.raises(InputError) as exc_info:
            provider.acquire(req)
        assert exc_info.value.error_code == "NOT_A_DIRECTORY"

    def test_oversized_file_in_dir_raises_resource_limit(self, tmp_path: Path) -> None:
        large = tmp_path / "model.safetensors"
        large.write_bytes(b"X" * 200)
        config = ScanConfig(max_download_file_bytes=100)
        req = ScanRequest(
            target=str(tmp_path),
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
            config=config,
        )
        provider = LocalDirectoryAcquisition()
        with pytest.raises(ResourceLimitError):
            provider.acquire(req)


# ---------------------------------------------------------------------------
# 4. Public URL Acquisition & Redirect Security Tests
# ---------------------------------------------------------------------------


class TestPublicUrlAcquisition:
    """Test PublicUrlAcquisition with SSRF, redirect interception, and size bounds."""

    def test_disallowed_schemes_rejected(self) -> None:
        disallowed = [
            "file:///etc/passwd",
            "ftp://ftp.example.com/model.pt",
            "gopher://gopher.example.com",
            "data:text/plain;base64,SGVsbG8=",
        ]
        for url in disallowed:
            with pytest.raises(InputError) as exc_info:
                _validate_url(url)
            assert exc_info.value.error_code == "INVALID_URL_SCHEME"

    def test_ssrf_private_ip_blocked(self) -> None:
        private_hosts = [
            "127.0.0.1",
            "10.0.0.5",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",  # AWS metadata service
            "localhost",
        ]
        for host in private_hosts:
            assert _is_private_address(host) is True

    def test_ssrf_validation_blocks_private_url(self) -> None:
        with pytest.raises(AcquisitionError) as exc_info:
            _validate_url("http://127.0.0.1:8080/model.bin")
        assert exc_info.value.error_code == "SSRF_BLOCKED"

    def test_safe_redirect_handler_intercepts_private_redirect(self) -> None:
        """SafeRedirectHandler must block redirect to private IP BEFORE connecting."""
        handler = SafeRedirectHandler()
        req = urllib.request.Request("http://example.com/start")

        private_destinations = [
            "http://127.0.0.1/admin",
            "http://10.0.0.1/secret",
            "http://192.168.1.1/router",
            "http://localhost:8000/internal",
            "http://169.254.169.254/latest/meta-data/",
        ]

        for target in private_destinations:
            with pytest.raises(AcquisitionError) as exc_info:
                handler.redirect_request(
                    req=req,
                    fp=None,
                    code=302,
                    msg="Found",
                    headers={},
                    newurl=target,
                )
            assert exc_info.value.error_code == "SSRF_BLOCKED"

    def test_safe_redirect_handler_intercepts_disallowed_scheme(self) -> None:
        """SafeRedirectHandler must block redirect to file:// or ftp:// schemes."""
        handler = SafeRedirectHandler()
        req = urllib.request.Request("http://example.com/start")

        for bad_url in ["file:///etc/passwd", "ftp://example.com/pub/model.pt"]:
            with pytest.raises(InputError) as exc_info:
                handler.redirect_request(
                    req=req,
                    fp=None,
                    code=302,
                    msg="Found",
                    headers={},
                    newurl=bad_url,
                )
            assert exc_info.value.error_code == "INVALID_URL_SCHEME"

    def test_safe_redirect_handler_max_redirects(self) -> None:
        handler = SafeRedirectHandler(max_redirects=2)
        req = urllib.request.Request("http://example.com/0")

        with patch("model_scanner.acquisition.url.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 0))]
            # Hop 1
            handler.redirect_request(req, None, 302, "Found", {}, "http://example.com/1")
            # Hop 2
            handler.redirect_request(req, None, 302, "Found", {}, "http://example.com/2")
            # Hop 3 -> limit exceeded
            with pytest.raises(AcquisitionError) as exc_info:
                handler.redirect_request(req, None, 302, "Found", {}, "http://example.com/3")
            assert exc_info.value.error_code == "TOO_MANY_REDIRECTS"

    @patch("model_scanner.acquisition.url.socket.getaddrinfo")
    @patch.object(urllib.request.OpenerDirector, "open")
    def test_successful_mocked_download_with_public_redirect(
        self, mock_open: MagicMock, mock_getaddrinfo: MagicMock
    ) -> None:
        # Mock public IP resolution
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 0))
        ]

        fake_bytes = b"fake remote model content after redirect"
        mock_resp = MagicMock()
        mock_resp.url = "https://cdn.example.com/model.safetensors"
        mock_resp.headers = {"Content-Length": str(len(fake_bytes))}
        mock_resp.read.side_effect = [fake_bytes, b""]
        mock_resp.__enter__.return_value = mock_resp

        mock_open.return_value = mock_resp

        provider = PublicUrlAcquisition()
        req = ScanRequest(
            target="http://models.example.com/redirect-to-cdn",
            input_source_type=InputSourceType.PUBLIC_URL,
        )
        artifacts = provider.acquire(req)

        assert len(artifacts) == 1
        art = artifacts[0]
        assert art.input_source == InputSourceType.PUBLIC_URL
        assert art.size_bytes == len(fake_bytes)
        assert art.sha256 == hashlib.sha256(fake_bytes).hexdigest()

        if art.local_path:
            art.local_path.unlink(missing_ok=True)

    @patch("model_scanner.acquisition.url.socket.getaddrinfo")
    @patch.object(urllib.request.OpenerDirector, "open")
    def test_content_length_limit_exceeded(
        self, mock_open: MagicMock, mock_getaddrinfo: MagicMock
    ) -> None:
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 0))
        ]

        mock_resp = MagicMock()
        mock_resp.url = "https://models.example.com/huge.safetensors"
        mock_resp.headers = {"Content-Length": "2000000000"}  # 2 GB
        mock_resp.__enter__.return_value = mock_resp
        mock_open.return_value = mock_resp

        config = ScanConfig(max_download_file_bytes=1000)
        provider = PublicUrlAcquisition()
        req = ScanRequest(
            target="https://models.example.com/huge.safetensors",
            input_source_type=InputSourceType.PUBLIC_URL,
            config=config,
        )
        with pytest.raises(ResourceLimitError) as exc_info:
            provider.acquire(req)
        assert exc_info.value.limit_name == "max_download_file_bytes"


# ---------------------------------------------------------------------------
# 5. Hugging Face Acquisition Tests
# ---------------------------------------------------------------------------


class TestHuggingFaceAcquisition:
    """Test Hugging Face acquisition adapter."""

    def test_url_classification(self) -> None:
        direct = "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin"
        repo = "https://huggingface.co/bert-base-uncased"
        dataset = "https://huggingface.co/datasets/scikit-learn/Fish"
        dataset_repo = "https://huggingface.co/datasets/scikit-learn"
        dataset_direct = "https://huggingface.co/datasets/scikit-learn/Fish/resolve/main/data.safetensors"

        assert _is_hf_direct_file(direct) is True
        assert _is_hf_repo(direct) is False

        assert _is_hf_direct_file(repo) is False
        assert _is_hf_repo(repo) is True

        assert _is_hf_direct_file(dataset_direct) is True

        from model_scanner.acquisition.huggingface import _is_hf_dataset
        assert _is_hf_dataset(dataset) is True
        assert _is_hf_repo(dataset) is False

        assert _is_hf_dataset(dataset_repo) is True
        assert _is_hf_repo(dataset_repo) is False

        assert is_hf_url(direct) is True
        assert is_hf_url(repo) is True
        assert is_hf_url(dataset) is True
        assert is_hf_url(dataset_repo) is True
        assert is_hf_url(dataset_direct) is True
        assert is_hf_url("https://example.com/model.bin") is False

    def test_extract_repo_id(self) -> None:
        url1 = "https://huggingface.co/openai/whisper-tiny"
        url2 = "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors"
        assert _extract_repo_id(url1) == "openai/whisper-tiny"
        assert _extract_repo_id(url2) == "openai/whisper-tiny"

    def test_filter_model_candidates_prefers_safetensors(self) -> None:
        """When SafeTensors is present alongside .bin weights, SafeTensors is preferred."""
        siblings = [
            {"rfilename": "model.safetensors", "size": 1000},
            {"rfilename": "pytorch_model.bin", "size": 2000},
            {"rfilename": "model.py", "size": 500},            # forbidden script
            {"rfilename": "setup.py", "size": 300},            # forbidden script
            {"rfilename": "README.md", "size": 100},           # non-model
            {"rfilename": ".gitattributes", "size": 50},       # non-model
            {"rfilename": "weights.gguf", "size": 5000},
        ]
        candidates = _filter_model_candidates(siblings)
        candidate_names = [c["rfilename"] for c in candidates]

        assert "model.safetensors" in candidate_names
        assert "weights.gguf" in candidate_names
        assert "pytorch_model.bin" not in candidate_names  # Filtered in favor of SafeTensors
        assert "model.py" not in candidate_names
        assert "setup.py" not in candidate_names
        assert "README.md" not in candidate_names

    def test_filter_model_candidates_bin_only_fallback(self) -> None:
        """When only .bin weights are available, they are preserved for scanning."""
        siblings = [
            {"rfilename": "pytorch_model.bin", "size": 2000},
            {"rfilename": "README.md", "size": 100},
        ]
        candidates = _filter_model_candidates(siblings)
        candidate_names = [c["rfilename"] for c in candidates]

        assert "pytorch_model.bin" in candidate_names
        assert "README.md" not in candidate_names

    def test_filter_model_candidates_onnx_extension_acquired(self) -> None:
        """Regression: .onnx files must not be filtered out by _filter_model_candidates.

        .onnx was absent from _HF_MODEL_EXTENSIONS, causing ONNX-only repos to
        return empty candidate lists and produce UNSUPPORTED instead of routing
        to ONNXAnalyzer.
        """
        siblings = [
            {"rfilename": "model.onnx", "size": 65536},
            {"rfilename": "config.json", "size": 200},
            {"rfilename": "README.md", "size": 300},
        ]
        candidates = _filter_model_candidates(siblings)
        candidate_names = [c["rfilename"] for c in candidates]

        assert "model.onnx" in candidate_names
        assert "config.json" not in candidate_names
        assert "README.md" not in candidate_names

    def test_filter_model_candidates_keras_h5_extensions_acquired(self) -> None:
        """Regression: .h5, .hdf5, and .keras files must not be filtered out.

        These extensions were absent from _HF_MODEL_EXTENSIONS, causing repos
        that contain only Keras weights to return empty candidate lists and
        produce an UNSUPPORTED result instead of routing to KerasAnalyzer.
        """
        siblings = [
            {"rfilename": "sample_victim_data.h5", "size": 4096},
            {"rfilename": "model.hdf5", "size": 8192},
            {"rfilename": "model.keras", "size": 2048},
            {"rfilename": "README.md", "size": 100},
            {"rfilename": "config.json", "size": 200},
        ]
        candidates = _filter_model_candidates(siblings)
        candidate_names = [c["rfilename"] for c in candidates]

        assert "sample_victim_data.h5" in candidate_names
        assert "model.hdf5" in candidate_names
        assert "model.keras" in candidate_names
        assert "README.md" not in candidate_names
        assert "config.json" not in candidate_names

    @patch("model_scanner.acquisition.huggingface._fetch_repo_file_list")
    @patch.object(PublicUrlAcquisition, "acquire")
    def test_hf_repo_acquisition_flow(
        self, mock_acquire: MagicMock, mock_fetch: MagicMock, tmp_path: Path
    ) -> None:
        mock_fetch.return_value = [
            {"rfilename": "model.safetensors", "size": 100},
            {"rfilename": "README.md", "size": 50},
        ]

        dummy_art = Artifact(
            artifact_id="art-1",
            source_path="https://huggingface.co/org/model/resolve/main/model.safetensors",
            input_source=InputSourceType.PUBLIC_URL,
            size_bytes=100,
            sha256="a" * 64,
            local_path=tmp_path / "dummy.safetensors",
        )
        mock_acquire.return_value = [dummy_art]

        provider = HuggingFaceAcquisition()
        req = ScanRequest(
            target="https://huggingface.co/org/model",
            input_source_type=InputSourceType.PUBLIC_URL,
        )
        artifacts = provider.acquire(req)

        assert len(artifacts) == 1
        assert artifacts[0].size_bytes == 100

    @patch("model_scanner.acquisition.huggingface._fetch_repo_file_list")
    @patch.object(PublicUrlAcquisition, "acquire")
    def test_hf_dataset_url_returns_empty_list(
        self, mock_acquire: MagicMock, mock_fetch: MagicMock
    ) -> None:
        provider = HuggingFaceAcquisition()

        # Test C: dataset repo with 3 segments
        req1 = ScanRequest(
            target="https://huggingface.co/datasets/scikit-learn/Fish",
            input_source_type=InputSourceType.PUBLIC_URL,
        )
        artifacts1 = provider.acquire(req1)
        assert artifacts1 == []
        mock_fetch.assert_not_called()
        mock_acquire.assert_not_called()

        # Test D: dataset repo with 2 segments
        req2 = ScanRequest(
            target="https://huggingface.co/datasets/scikit-learn",
            input_source_type=InputSourceType.PUBLIC_URL,
        )
        artifacts2 = provider.acquire(req2)
        assert artifacts2 == []
        mock_fetch.assert_not_called()
        mock_acquire.assert_not_called()

        # Test E: dataset direct file URL
        req3 = ScanRequest(
            target="https://huggingface.co/datasets/scikit-learn/Fish/resolve/main/data.safetensors",
            input_source_type=InputSourceType.PUBLIC_URL,
        )
        provider.acquire(req3)
        # It should hit the direct file acquisition, which delegates to PublicUrlAcquisition.acquire
        mock_acquire.assert_called_once()
        mock_fetch.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Safety & Architecture Guarantees
# ---------------------------------------------------------------------------


class TestAcquisitionSafetyGuarantees:
    """Verify zero forbidden dynamic execution in acquisition subsystem."""

    def test_acquisition_provider_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            InputAcquisitionProvider()  # type: ignore[abstract]

    def test_no_torch_or_safetensors_imported(self) -> None:
        forbidden = ["torch", "safetensors", "transformers", "huggingface_hub"]
        for mod in forbidden:
            assert mod not in sys.modules, f"Forbidden module {mod} was imported!"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows symlink creation requires elevated privileges not available in CI",
)
class TestSymlinkEscapeGuard:
    """A4: LocalDirectoryAcquisition must block out-of-root symlinks silently."""

    def test_symlink_to_parent_directory_is_blocked(self, tmp_path: Path) -> None:
        """A symlink pointing outside the scan root must be skipped, not followed."""
        scan_root = tmp_path / "scan"
        scan_root.mkdir()
        outside = tmp_path / "outside.pkl"
        outside.write_bytes(b"\x80\x02}q\x00.")

        link = scan_root / "escape.pkl"
        link.symlink_to(outside)

        acq = LocalDirectoryAcquisition()
        request = ScanRequest(
            scan_id="test",
            target=scan_root,
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        artifacts = acq.acquire(request)
        assert len(artifacts) == 0, "Out-of-root symlink must not produce an artifact"
        assert len(acq.skipped_symlinks) == 1
        assert acq.skipped_symlinks[0].warning_code == "SYMLINK_ESCAPE_BLOCKED"

    def test_symlink_to_absolute_system_path_is_blocked(self, tmp_path: Path) -> None:
        """A symlink to an absolute path outside the scan root must be skipped."""
        scan_root = tmp_path / "scan"
        scan_root.mkdir()

        link = scan_root / "passwd.pkl"
        # /etc/passwd may not exist on all platforms but symlink creation is still valid
        link.symlink_to(Path("/etc/passwd"))

        acq = LocalDirectoryAcquisition()
        request = ScanRequest(
            scan_id="test",
            target=scan_root,
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        artifacts = acq.acquire(request)
        assert len(artifacts) == 0
        assert any(
            w.warning_code in ("SYMLINK_ESCAPE_BLOCKED", "SYMLINK_LOOP_SKIPPED")
            for w in acq.skipped_symlinks
        )

    def test_circular_symlink_pair_is_skipped(self, tmp_path: Path) -> None:
        """A->B->A circular symlink pair must not cause infinite recursion."""
        scan_root = tmp_path / "scan"
        scan_root.mkdir()

        # Create A -> B -> A cycle — both have .pkl extension so they'd be candidates
        link_a = scan_root / "a.pkl"
        link_b = scan_root / "b.pkl"
        link_a.symlink_to(link_b)
        link_b.symlink_to(link_a)

        acq = LocalDirectoryAcquisition()
        request = ScanRequest(
            scan_id="test",
            target=scan_root,
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        # Must not raise and must produce no artifacts (both links are unresolvable loops)
        artifacts = acq.acquire(request)
        assert len(artifacts) == 0
        assert len(acq.skipped_symlinks) >= 1

    def test_benign_files_still_acquired_alongside_blocked_symlink(
        self, tmp_path: Path
    ) -> None:
        """Legitimate files in the same directory are acquired even when a symlink is blocked."""
        scan_root = tmp_path / "scan"
        scan_root.mkdir()

        good = scan_root / "good.pkl"
        good.write_bytes(b"\x80\x02}q\x00.")

        outside = tmp_path / "bad.pkl"
        outside.write_bytes(b"\x80\x02}q\x00.")
        link = scan_root / "evil.pkl"
        link.symlink_to(outside)

        acq = LocalDirectoryAcquisition()
        request = ScanRequest(
            scan_id="test",
            target=scan_root,
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        artifacts = acq.acquire(request)
        assert len(artifacts) == 1
        assert artifacts[0].source_path == str(good.resolve())
        assert len(acq.skipped_symlinks) == 1


# ---------------------------------------------------------------------------
# Regression: TF SavedModel directory discovery (saved_model.pb by filename)
# ---------------------------------------------------------------------------


class TestSavedModelDirectoryDiscovery:
    """Regression tests for saved_model.pb filename-based directory discovery.

    saved_model.pb must be discovered during LOCAL_DIRECTORY scans even though
    .pb is not in _MODEL_EXTENSIONS.  FormatDispatcher already detects the
    format by filename; this class verifies the acquisition path feeds it.

    Requirements:
    a. A directory containing saved_model.pb has that file discovered.
    b. The discovered file is detected as TF_SAVEDMODEL by the engine.
    c. An arbitrary .pb file NOT named saved_model.pb is NOT discovered.
    d. Existing directory acquisition behaviour for other extensions is unchanged.
    """

    def test_saved_model_pb_is_discovered_in_directory(self, tmp_path: Path) -> None:
        """saved_model.pb inside a directory must be returned as a candidate."""
        (tmp_path / "saved_model.pb").write_bytes(b"\x0a\x04test")

        provider = LocalDirectoryAcquisition()
        req = ScanRequest(
            target=str(tmp_path),
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        artifacts = provider.acquire(req)

        names = [Path(a.source_path).name for a in artifacts]
        assert "saved_model.pb" in names, (
            "saved_model.pb must be discovered during directory scan"
        )

    def test_saved_model_pb_in_subdirectory_is_discovered(self, tmp_path: Path) -> None:
        """saved_model.pb nested inside a sub-directory must also be found."""
        subdir = tmp_path / "1"
        subdir.mkdir()
        (subdir / "saved_model.pb").write_bytes(b"\x0a\x04test")

        provider = LocalDirectoryAcquisition()
        req = ScanRequest(
            target=str(tmp_path),
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        artifacts = provider.acquire(req)

        names = [Path(a.source_path).name for a in artifacts]
        assert "saved_model.pb" in names

    def test_arbitrary_pb_file_is_not_discovered(self, tmp_path: Path) -> None:
        """A .pb file NOT named saved_model.pb must NOT be discovered.

        Adding .pb to _MODEL_EXTENSIONS would acquire every protobuf file, which
        is wrong.  Only the exact filename saved_model.pb must be admitted.
        """
        (tmp_path / "other_proto.pb").write_bytes(b"\x0a\x04test")
        (tmp_path / "model_config.pb").write_bytes(b"\x0a\x04test")

        provider = LocalDirectoryAcquisition()
        req = ScanRequest(
            target=str(tmp_path),
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        artifacts = provider.acquire(req)

        names = [Path(a.source_path).name for a in artifacts]
        assert "other_proto.pb" not in names, "Arbitrary .pb must not be discovered"
        assert "model_config.pb" not in names, "Arbitrary .pb must not be discovered"

    def test_existing_extension_behavior_unchanged(self, tmp_path: Path) -> None:
        """Ensure standard model extensions are still discovered alongside saved_model.pb."""
        (tmp_path / "model.safetensors").write_bytes(b"sfdata")
        (tmp_path / "model.gguf").write_bytes(b"GGUFdata")
        (tmp_path / "saved_model.pb").write_bytes(b"\x0a\x04test")
        (tmp_path / "readme.txt").write_bytes(b"not a model")

        provider = LocalDirectoryAcquisition()
        req = ScanRequest(
            target=str(tmp_path),
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        artifacts = provider.acquire(req)

        names = {Path(a.source_path).name for a in artifacts}
        assert "model.safetensors" in names
        assert "model.gguf" in names
        assert "saved_model.pb" in names
        assert "readme.txt" not in names

    def test_saved_model_routed_to_tf_savedmodel_format(self, tmp_path: Path) -> None:
        """Full pipeline: saved_model.pb in a directory → TF_SAVEDMODEL format detected."""
        from model_scanner.core.engine import ScanOrchestrator
        from model_scanner.core.models import ModelFormat, ScanOutcome

        (tmp_path / "saved_model.pb").write_bytes(b"\x0a\x04test")

        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        orc = ScanOrchestrator()
        result = orc.scan(request)

        tf_artifacts = [
            a for a in result.artifacts
            if a.detected_format == ModelFormat.TF_SAVEDMODEL
        ]
        assert len(tf_artifacts) == 1, (
            f"Expected 1 TF_SAVEDMODEL artifact, got {len(tf_artifacts)}: "
            f"{[(a.name, a.detected_format.value) for a in result.artifacts]}"
        )
        tf = tf_artifacts[0]
        assert tf.name == "saved_model.pb"
        assert tf.scan_outcome in (ScanOutcome.COMPLETE, ScanOutcome.WARNING), (
            f"Expected COMPLETE or WARNING outcome, got {tf.scan_outcome.value}"
        )

    def test_saved_model_analyzer_is_tensorflow_analyzer(self, tmp_path: Path) -> None:
        """Verify get_analyzer(TF_SAVEDMODEL) returns a TensorFlowAnalyzer instance."""
        from model_scanner.analyzers.tensorflow.analyzer import TensorFlowAnalyzer
        from model_scanner.core.engine import get_analyzer
        from model_scanner.core.models import ModelFormat

        analyzer = get_analyzer(ModelFormat.TF_SAVEDMODEL)
        assert isinstance(analyzer, TensorFlowAnalyzer), (
            f"Expected TensorFlowAnalyzer, got {type(analyzer).__name__}"
        )


# ---------------------------------------------------------------------------
# 7. TC1: Extensionless / hidden Pickle candidate acquisition
# ---------------------------------------------------------------------------


class TestSuspiciousExtensionlessAcquisition:
    """Hidden/extensionless files are peeked and only fetched if Pickle magic.

    A raw Pickle payload hidden in an extensionless file (e.g. '.metadata') is
    excluded by extension-based filtering; the secondary pass recovers it.
    """

    def test_is_suspicious_helper_classification(self) -> None:
        assert _is_suspicious_extensionless(".metadata") is True
        assert _is_suspicious_extensionless("checkpoint") is True
        assert _is_suspicious_extensionless("model") is True
        # Non-model files must be excluded.
        assert _is_suspicious_extensionless("README") is False
        assert _is_suspicious_extensionless("config.json") is False
        assert _is_suspicious_extensionless("vocab.json") is False
        assert _is_suspicious_extensionless(".gitattributes") is False
        assert _is_suspicious_extensionless("model.py") is False
        assert _is_suspicious_extensionless("model.safetensors") is False

    @patch("model_scanner.acquisition.huggingface._peek_is_pickle", return_value=True)
    @patch("model_scanner.acquisition.huggingface._fetch_repo_file_list")
    @patch.object(PublicUrlAcquisition, "acquire")
    def test_extensionless_pickle_candidate_included(
        self, mock_acquire: MagicMock, mock_fetch: MagicMock,
        mock_peek: MagicMock, tmp_path: Path,
    ) -> None:
        mock_fetch.return_value = [
            {"rfilename": ".metadata", "size": 100},
            {"rfilename": "README.md", "size": 50},
        ]
        dummy = Artifact(
            artifact_id="art-meta",
            source_path="https://huggingface.co/org/model/resolve/main/.metadata",
            input_source=InputSourceType.PUBLIC_URL,
            size_bytes=100,
            sha256="a" * 64,
            local_path=tmp_path / "dummy.metadata",
        )
        mock_acquire.return_value = [dummy]

        provider = HuggingFaceAcquisition()
        req = ScanRequest(
            target="https://huggingface.co/org/model",
            input_source_type=InputSourceType.PUBLIC_URL,
        )
        artifacts = provider.acquire(req)

        assert len(artifacts) == 1
        mock_peek.assert_called_once()  # peeked before full download
        mock_acquire.assert_called_once()

    @patch("model_scanner.acquisition.huggingface._peek_is_pickle", return_value=False)
    @patch("model_scanner.acquisition.huggingface._fetch_repo_file_list")
    @patch.object(PublicUrlAcquisition, "acquire")
    def test_extensionless_harmless_excluded(
        self, mock_acquire: MagicMock, mock_fetch: MagicMock,
        mock_peek: MagicMock,
    ) -> None:
        # A 'config' file whose peeked bytes are NOT Pickle magic must be dropped.
        mock_fetch.return_value = [
            {"rfilename": "config", "size": 100},
        ]
        provider = HuggingFaceAcquisition()
        req = ScanRequest(
            target="https://huggingface.co/org/model",
            input_source_type=InputSourceType.PUBLIC_URL,
        )
        artifacts = provider.acquire(req)

        assert artifacts == []
        mock_peek.assert_called_once()
        mock_acquire.assert_not_called()  # never fully downloaded

    @patch("model_scanner.acquisition.huggingface._peek_is_pickle", return_value=True)
    @patch("model_scanner.acquisition.huggingface._fetch_repo_file_list")
    @patch.object(PublicUrlAcquisition, "acquire")
    def test_readme_not_included_as_hidden_candidate(
        self, mock_acquire: MagicMock, mock_fetch: MagicMock,
        mock_peek: MagicMock,
    ) -> None:
        mock_fetch.return_value = [
            {"rfilename": "README", "size": 100},
            {"rfilename": "LICENSE", "size": 100},
        ]
        provider = HuggingFaceAcquisition()
        req = ScanRequest(
            target="https://huggingface.co/org/model",
            input_source_type=InputSourceType.PUBLIC_URL,
        )
        artifacts = provider.acquire(req)

        assert artifacts == []
        mock_peek.assert_not_called()  # README/LICENSE never even peeked
        mock_acquire.assert_not_called()

    @patch("model_scanner.acquisition.huggingface._peek_is_pickle", return_value=True)
    @patch("model_scanner.acquisition.huggingface._fetch_repo_file_list")
    @patch.object(PublicUrlAcquisition, "acquire")
    def test_oversized_extensionless_excluded(
        self, mock_acquire: MagicMock, mock_fetch: MagicMock,
        mock_peek: MagicMock,
    ) -> None:
        config = ScanConfig(max_download_file_bytes=1000)
        mock_fetch.return_value = [
            {"rfilename": ".metadata", "size": 5000},  # exceeds per-file limit
        ]
        provider = HuggingFaceAcquisition()
        req = ScanRequest(
            target="https://huggingface.co/org/model",
            input_source_type=InputSourceType.PUBLIC_URL,
            config=config,
        )
        artifacts = provider.acquire(req)

        assert artifacts == []
        mock_peek.assert_not_called()  # size-gated before any peek
        mock_acquire.assert_not_called()

    # --- TC1-specific regression tests ---

    def test_is_suspicious_edge_cases(self) -> None:
        """Edge cases for the extensionless classifier."""
        # TC1 exact filename
        assert _is_suspicious_extensionless(".metadata") is True
        # Nested path: only the basename is checked for extension; should include.
        assert _is_suspicious_extensionless("subdir/checkpoint") is True
        # .git paths are always excluded regardless of basename.
        assert _is_suspicious_extensionless(".git/config") is False
        # .github also starts with ".git" → excluded.
        assert _is_suspicious_extensionless(".github/workflows/evil") is False
        # Hidden file with a known non-model second extension → excluded.
        assert _is_suspicious_extensionless(".hidden.json") is False
        # Hidden file with an UNKNOWN second extension (.bak) → excluded because
        # suffix is non-empty and not a known model format.  This is intentionally
        # conservative: the secondary pass targets completely extension-free names.
        assert _is_suspicious_extensionless(".metadata.bak") is False
        # Completely bare name with no leading dot and no extension → included.
        assert _is_suspicious_extensionless("data") is True

    @patch("model_scanner.acquisition.huggingface._peek_is_pickle", return_value=True)
    @patch("model_scanner.acquisition.huggingface._fetch_repo_file_list")
    @patch.object(PublicUrlAcquisition, "acquire")
    def test_tc1_metadata_acquisition_chain(
        self, mock_acquire: MagicMock, mock_fetch: MagicMock,
        mock_peek: MagicMock, tmp_path: Path,
    ) -> None:
        """TC1 regression: the exact `.metadata` filename triggers secondary-pass
        acquisition when Pickle magic bytes are present, and primary-pass
        model files in the same repo are acquired normally alongside it."""
        mock_fetch.return_value = [
            {"rfilename": "model.safetensors", "size": 1024},
            {"rfilename": ".metadata", "size": 234},
        ]
        safetensors_artifact = Artifact(
            artifact_id="art-st",
            source_path="https://huggingface.co/org/model/resolve/main/model.safetensors",
            input_source=InputSourceType.PUBLIC_URL,
            size_bytes=1024,
            sha256="b" * 64,
            local_path=tmp_path / "model.safetensors",
        )
        metadata_artifact = Artifact(
            artifact_id="art-meta",
            source_path="https://huggingface.co/org/model/resolve/main/.metadata",
            input_source=InputSourceType.PUBLIC_URL,
            size_bytes=234,
            sha256="4de9561840e43aa6d1e25c01c9ad78290caf36227d58e0a1aa76f624420c3334",
            local_path=tmp_path / ".metadata",
        )
        # Primary pass returns the safetensors; secondary pass returns the hidden file.
        mock_acquire.side_effect = [[safetensors_artifact], [metadata_artifact]]

        provider = HuggingFaceAcquisition()
        req = ScanRequest(
            target="https://huggingface.co/org/model",
            input_source_type=InputSourceType.PUBLIC_URL,
        )
        artifacts = provider.acquire(req)

        assert len(artifacts) == 2
        source_origins = {a.source_path for a in artifacts}
        assert any(".metadata" in s for s in source_origins)
        assert any(".safetensors" in s for s in source_origins)
        mock_peek.assert_called_once()  # peeked exactly once for .metadata
        assert mock_acquire.call_count == 2  # primary + secondary download

    @patch("model_scanner.acquisition.huggingface._peek_is_pickle", return_value=False)
    @patch("model_scanner.acquisition.huggingface._fetch_repo_file_list")
    @patch.object(PublicUrlAcquisition, "acquire")
    def test_non_pickle_binary_extensionless_not_downloaded(
        self, mock_acquire: MagicMock, mock_fetch: MagicMock,
        mock_peek: MagicMock,
    ) -> None:
        """Test 8: random binary with no Pickle magic bytes is NOT downloaded.

        The peek step acts as a gate: only files with binary Pickle magic
        (0x80 opcode or legacy MARK opcode) proceed to full acquisition.
        Arbitrary binary files (ELF, PE, ZIP, etc.) are skipped.
        """
        mock_fetch.return_value = [{"rfilename": "model", "size": 200}]
        provider = HuggingFaceAcquisition()
        req = ScanRequest(
            target="https://huggingface.co/org/model",
            input_source_type=InputSourceType.PUBLIC_URL,
        )
        artifacts = provider.acquire(req)

        assert artifacts == []
        mock_peek.assert_called_once()
        mock_acquire.assert_not_called()

    @patch("model_scanner.acquisition.huggingface._peek_is_pickle", return_value=True)
    @patch("model_scanner.acquisition.huggingface._fetch_repo_file_list")
    @patch.object(PublicUrlAcquisition, "acquire")
    def test_strict_mode_extensionless_still_acquired(
        self, mock_acquire: MagicMock, mock_fetch: MagicMock,
        mock_peek: MagicMock, tmp_path: Path,
    ) -> None:
        """Test 9: strict_mode=True does not suppress the extensionless secondary pass.

        The secondary-pass logic gates on magic bytes and file size, not on
        strict_mode. Strict mode controls scan thoroughness, not acquisition
        safety gates.
        """
        config = ScanConfig(strict_mode=True)
        mock_fetch.return_value = [{"rfilename": ".metadata", "size": 100}]
        dummy = Artifact(
            artifact_id="art-strict",
            source_path="https://huggingface.co/org/model/resolve/main/.metadata",
            input_source=InputSourceType.PUBLIC_URL,
            size_bytes=100,
            sha256="c" * 64,
            local_path=tmp_path / ".metadata",
        )
        mock_acquire.return_value = [dummy]

        provider = HuggingFaceAcquisition()
        req = ScanRequest(
            target="https://huggingface.co/org/model",
            input_source_type=InputSourceType.PUBLIC_URL,
            config=config,
        )
        artifacts = provider.acquire(req)

        assert len(artifacts) == 1
        mock_peek.assert_called_once()
        mock_acquire.assert_called_once()


# ---------------------------------------------------------------------------
# 8. TC4: saved_model.pb candidate acquisition
# ---------------------------------------------------------------------------


class TestSavedModelPbCandidate:
    """.pb handling: saved_model.pb is a candidate; arbitrary .pb is not."""

    def test_saved_model_pb_included(self) -> None:
        siblings = [
            {"rfilename": "saved_model.pb", "size": 4096},
            {"rfilename": "keras_metadata.pb", "size": 2048},
            {"rfilename": "variables/variables.index", "size": 128},
        ]
        candidates = _filter_model_candidates(siblings)
        names = [c["rfilename"] for c in candidates]
        assert "saved_model.pb" in names
        assert "keras_metadata.pb" not in names  # arbitrary .pb excluded

    def test_saved_model_pb_kept_with_safetensors_present(self) -> None:
        siblings = [
            {"rfilename": "model.safetensors", "size": 1000},
            {"rfilename": "saved_model.pb", "size": 4096},
        ]
        candidates = _filter_model_candidates(siblings)
        names = [c["rfilename"] for c in candidates]
        assert "model.safetensors" in names
        assert "saved_model.pb" in names  # kept despite SafeTensors preference
