"""
model_scanner.acquisition.url
==============================

Controlled public HTTP/HTTPS URL acquisition with SSRF protections,
size quota enforcement, streaming SHA-256 fingerprinting, and timeout
ceiling.

SECURITY GUARANTEE
------------------
This module NEVER deserializes, executes, or loads model weights.
All downloaded bytes are written to a temporary in-memory buffer.
No downloaded Python scripts, setup.py, or dynamic modules are imported
or executed.

Design Reference
----------------
Section 11 of docs/implementation_design.md
"""

from __future__ import annotations

import ipaddress
import re
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Sequence
from http.client import HTTPMessage
from pathlib import Path
from typing import IO

from model_scanner.acquisition.base import InputAcquisitionProvider
from model_scanner.core.exceptions import AcquisitionError, InputError, ResourceLimitError
from model_scanner.core.models import Artifact, InputSourceType, ScanConfig, ScanRequest
from model_scanner.utils.hashing import sha256_incremental

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALLOWED_SCHEMES: frozenset[str] = frozenset(["http", "https"])

# RFC-1918 private address ranges + loopback + link-local + reserved
_PRIVATE_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),        # loopback
    ipaddress.ip_network("169.254.0.0/16"),      # link-local
    ipaddress.ip_network("0.0.0.0/8"),           # this network
    ipaddress.ip_network("100.64.0.0/10"),       # shared address space
    ipaddress.ip_network("192.0.0.0/24"),        # IETF protocol
    ipaddress.ip_network("192.88.99.0/24"),      # 6to4 relay
    ipaddress.ip_network("198.18.0.0/15"),       # benchmarking
    ipaddress.ip_network("198.51.100.0/24"),     # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),      # TEST-NET-3
    ipaddress.ip_network("240.0.0.0/4"),         # reserved
    ipaddress.ip_network("255.255.255.255/32"),  # broadcast
    # IPv6 private/loopback
    ipaddress.ip_network("::1/128"),             # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),            # unique local
    ipaddress.ip_network("fe80::/10"),           # link-local
    ipaddress.ip_network("::/128"),              # unspecified
]

_DOWNLOAD_CHUNK_SIZE: int = 65_536   # 64 KB per Section 11.2
_MAX_REDIRECT_HOPS: int = 5          # max redirect chain length


def _is_private_address(hostname: str) -> bool:
    """Return True if hostname resolves to a private / loopback IP address.

    Performs a DNS resolution and checks all returned addresses against the
    SSRF-guard blocklist of RFC-1918, loopback, and reserved ranges.

    Parameters
    ----------
    hostname:
        The hostname (or literal IP) to check.

    Returns
    -------
    bool
        True if the address is private/reserved (SSRF risk).
    """
    try:
        # getaddrinfo returns (family, type, proto, canonname, sockaddr)
        results = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        # Resolution failure — conservatively treat as unsafe
        return True

    for _family, _type, _proto, _canonname, sockaddr in results:
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        for network in _PRIVATE_NETWORKS:
            if addr in network:
                return True
    return False


def _validate_url(url: str) -> urllib.parse.ParseResult:
    """Validate that URL uses http:// or https:// and passes SSRF checks.

    Parameters
    ----------
    url:
        The URL to validate.

    Returns
    -------
    urllib.parse.ParseResult
        The parsed URL components.

    Raises
    ------
    InputError
        If the URL scheme is not http or https.
    AcquisitionError
        If the resolved hostname falls within a private/reserved IP range
        (SSRF protection).
    """
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise InputError(
            f"URL scheme {parsed.scheme!r} is not allowed. "
            f"Only http:// and https:// are accepted.",
            error_code="INVALID_URL_SCHEME",
            details=url,
        )

    if not parsed.netloc:
        raise InputError(
            "URL has no hostname / netloc component.",
            error_code="INVALID_URL",
            details=url,
        )

    hostname = parsed.hostname or ""
    if not hostname:
        raise InputError(
            "URL hostname is empty.",
            error_code="INVALID_URL",
            details=url,
        )

    if _is_private_address(hostname):
        raise AcquisitionError(
            f"URL hostname {hostname!r} resolves to a private or reserved "
            f"IP address. SSRF protection triggered.",
            error_code="SSRF_BLOCKED",
            details=url,
        )

    return parsed


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Custom redirect handler that validates every redirect target BEFORE connecting.

    Enforces:
    - Scheme validation (http/https only)
    - Hostname validation
    - SSRF resolution against RFC-1918, loopback, link-local, reserved ranges
    - Maximum redirect hop limit (default 5)

    Raises
    ------
    AcquisitionError
        If redirect count exceeds max limit or destination resolves to a private IP.
    InputError
        If redirect scheme is disallowed.
    """

    def __init__(self, max_redirects: int = _MAX_REDIRECT_HOPS) -> None:
        super().__init__()
        self.max_redirects = max_redirects
        self.redirect_count = 0

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        self.redirect_count += 1
        if self.redirect_count > self.max_redirects:
            raise AcquisitionError(
                f"Too many redirects (exceeded limit of {self.max_redirects})",
                error_code="TOO_MANY_REDIRECTS",
                details=newurl,
            )

        # Resolve relative redirect URLs against the source request URL
        resolved_url = urllib.parse.urljoin(req.full_url, newurl)

        # STRICT SSRF AND SCHEME VALIDATION BEFORE FOLLOWING THE REDIRECT
        _validate_url(resolved_url)

        # Build and return the new validated Request
        return super().redirect_request(req, fp, code, msg, headers, resolved_url)


class PublicUrlAcquisition(InputAcquisitionProvider):
    """Acquires a model artifact from a public HTTP/HTTPS URL.

    Security enforcements:
    - Rejects non-HTTP/HTTPS URL schemes.
    - SSRF: blocks URLs resolving to RFC-1918/loopback/reserved IPs.
    - Pre-flight redirect interception (SafeRedirectHandler) to validate redirect
      destinations BEFORE connecting to them.
    - Enforces per-file download size quota via ``Content-Length`` pre-check
      and streaming byte counter.
    - Enforces download timeout (``config.download_timeout_sec``).
    - Downloads bytes in 64 KB chunks into a temporary file.
    - Computes SHA-256 during streaming without accumulating the full file
      in memory before hashing.
    - Never imports, executes, or deserializes downloaded content.

    Design Reference: Section 11.2

    Raises
    ------
    InputError
        If the URL scheme is disallowed.
    AcquisitionError
        If the request fails (network error, SSRF block, timeout, redirect
        to a private address).
    ResourceLimitError
        If ``Content-Length`` or streaming byte count exceeds limit.
    """

    def acquire(self, request: ScanRequest) -> Sequence[Artifact]:
        """Download and acquire a single URL artifact.

        Parameters
        ----------
        request:
            Must have ``input_source_type == PUBLIC_URL`` and ``target``
            as a valid http/https URL string.

        Returns
        -------
        Sequence[Artifact]
            A single-element list containing the acquired Artifact.
        """
        url = request.target
        if not isinstance(url, str):
            raise InputError(
                "PublicUrlAcquisition requires a str URL target.",
                details=repr(url),
            )

        config = request.config or ScanConfig()

        # Step 1: URL scheme and SSRF validation on initial URL
        _validate_url(url)

        # Steps 2-6: HEAD pre-flight, bounded streaming download, SHA-256
        artifact = self._download(url, config)
        return [artifact]

    def _download(self, url: str, config: ScanConfig) -> Artifact:
        """Execute the bounded streaming download and return an Artifact.

        Parameters
        ----------
        url:
            Validated http/https URL.
        config:
            Active ScanConfig carrying quota and timeout parameters.

        Returns
        -------
        Artifact
            Acquired and SHA-256-fingerprinted artifact backed by a temp file.
        """
        timeout = config.download_timeout_sec
        size_limit = config.max_download_file_bytes

        # Create openers equipped with SafeRedirectHandler to intercept redirects BEFORE following
        head_handler = SafeRedirectHandler(max_redirects=_MAX_REDIRECT_HOPS)
        head_opener = urllib.request.build_opener(head_handler)

        # Step 2: HEAD pre-flight to check Content-Length
        head_req = urllib.request.Request(url, method="HEAD")  # noqa: S310
        try:
            with head_opener.open(head_req, timeout=timeout) as head_resp:
                content_length_str = head_resp.headers.get("Content-Length", "")
                if content_length_str:
                    try:
                        content_length = int(content_length_str)
                        if content_length > size_limit:
                            raise ResourceLimitError(
                                f"Content-Length {content_length} exceeds maximum "
                                f"file size limit {size_limit}",
                                error_code="FILE_TOO_LARGE",
                                details=url,
                                limit_name="max_download_file_bytes",
                                limit_value=size_limit,
                                actual_value=content_length,
                            )
                    except ValueError:
                        pass  # Unparseable Content-Length — proceed with streaming guard
        except (ResourceLimitError, InputError, AcquisitionError):
            raise
        except (urllib.error.URLError, OSError) as exc:
            # HEAD failures are non-fatal — some servers don't support HEAD.
            # Proceed with GET; the streaming byte guard will enforce limits.
            _ = exc

        # Step 3-5: Streaming GET download with byte counter, SafeRedirectHandler, and SHA-256
        get_handler = SafeRedirectHandler(max_redirects=_MAX_REDIRECT_HOPS)
        get_opener = urllib.request.build_opener(get_handler)
        get_req = urllib.request.Request(url)  # noqa: S310
        try:
            with get_opener.open(get_req, timeout=timeout) as response:


                # Preserve original filename extension if valid to maintain format hints
                parsed_path = urllib.parse.urlparse(url).path
                raw_filename = Path(urllib.parse.unquote(parsed_path)).name
                raw_ext = (
                    Path(raw_filename).suffix.lower()
                    if "." in raw_filename
                    else ""
                )
                valid_exts = {
                    ".safetensors",
                    ".pt",
                    ".pth",
                    ".bin",
                    ".ckpt",
                    ".gguf",
                    ".ggml",
                    ".pkl",
                    ".pickle",
                    ".joblib",
                    ".dill",
                }
                safe_suffix = raw_ext if raw_ext in valid_exts else ".tmp_artifact"
                safe_stem = Path(raw_filename).stem if raw_filename else "artifact"
                clean_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", safe_stem)[:24]
                h = sha256_incremental()
                total_bytes = 0

                # Write to a named temporary file to avoid full memory accumulation
                with tempfile.NamedTemporaryFile(
                    prefix=f"dl_{clean_prefix}_",
                    suffix=safe_suffix,
                    delete=False,
                ) as tmp_fh:
                    tmp_path = Path(tmp_fh.name)
                    while True:
                        chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > size_limit:
                            tmp_path.unlink(missing_ok=True)
                            raise ResourceLimitError(
                                f"Download exceeded streaming size limit of {size_limit} bytes",
                                error_code="DOWNLOAD_TOO_LARGE",
                                details=url,
                                limit_name="max_download_file_bytes",
                                limit_value=size_limit,
                                actual_value=total_bytes,
                            )
                        h.update(chunk)
                        tmp_fh.write(chunk)

        except (ResourceLimitError, InputError, AcquisitionError):
            raise
        except urllib.error.URLError as exc:
            raise AcquisitionError(
                f"Network error downloading {url}: {exc.reason}",
                error_code="NETWORK_ERROR",
                details=str(exc),
            ) from exc
        except TimeoutError as exc:
            raise AcquisitionError(
                f"Download timed out after {timeout}s: {url}",
                error_code="DOWNLOAD_TIMEOUT",
                details=str(exc),
            ) from exc
        except OSError as exc:
            raise AcquisitionError(
                f"I/O error during download of {url}: {exc}",
                error_code="IO_ERROR",
                details=str(exc),
            ) from exc

        sha256 = h.hexdigest()

        return Artifact(
            artifact_id=str(uuid.uuid4()),
            source_path=url,
            input_source=InputSourceType.PUBLIC_URL,
            size_bytes=total_bytes,
            sha256=sha256,
            local_path=tmp_path,
        )


__all__ = [
    "PublicUrlAcquisition",
    "SafeRedirectHandler",
]
