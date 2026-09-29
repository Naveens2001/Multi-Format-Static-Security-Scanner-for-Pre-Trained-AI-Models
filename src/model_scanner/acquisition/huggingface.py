"""
model_scanner.acquisition.huggingface
======================================

Hugging Face repository and direct model file acquisition adapter.

This module is NOT a separate scanning engine. It is purely an acquisition
adapter that classifies a Hugging Face URL, enumerates candidate model
files from the public HF Model Hub API if needed, and delegates actual
downloading to PublicUrlAcquisition.

Once artifacts are acquired they enter the same format detection and
analysis pipeline as any other artifact.

SECURITY GUARANTEE
------------------
- Does NOT use the huggingface_hub Python SDK.
- Uses standard-library urllib only.
- Never imports, executes, or deserializes downloaded content.
- Repository metadata (JSON API responses) is treated as untrusted input.
- .py files, setup.py, scripts, and config code are NEVER downloaded.

Design Reference
----------------
Section 12 of docs/implementation_design.md
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence

from model_scanner.acquisition.base import InputAcquisitionProvider
from model_scanner.acquisition.url import PublicUrlAcquisition, SafeRedirectHandler, _validate_url
from model_scanner.core.exceptions import AcquisitionError, InputError, ResourceLimitError
from model_scanner.core.models import Artifact, InputSourceType, ScanConfig, ScanRequest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HF Model Hub API base URL for repository metadata
_HF_API_BASE: str = "https://huggingface.co/api/models"

# Supported candidate model file extensions to download from HF repos
_HF_MODEL_EXTENSIONS: frozenset[str] = frozenset(
    [".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".gguf", ".pkl", ".pickle",
     ".h5", ".hdf5", ".keras", ".onnx", ".pb", ".pbtxt"]
)

# Bare names of the TensorFlow SavedModel graph file. Only these exact .pb/.pbtxt
# filenames are treated as model candidates — arbitrary .pb protobufs are not.
# saved_model.pbtxt is the text-proto variant of the same TF SavedModel graph.
# (Finding 2: previously only .pb was matched; .pbtxt entries were silently dropped)
_HF_SAVEDMODEL_PB_NAMES: frozenset[str] = frozenset(["saved_model.pb", "saved_model.pbtxt"])

# Bare filenames (no extension) that are known non-model repository files and
# must be excluded from the suspicious extensionless/hidden-file scan.
_HF_NON_MODEL_BARE_NAMES: frozenset[str] = frozenset([
    "README", "LICENSE", "NOTICE", "CHANGELOG", "AUTHORS",
    "CONTRIBUTING", "Makefile", "Dockerfile",
])

# Extensions that identify ordinary non-model config/doc/asset files. A hidden
# or extensionless file bearing one of these is never a suspicious pickle candidate.
_HF_NON_MODEL_EXTENSIONS: frozenset[str] = frozenset([
    ".json", ".txt", ".md", ".rst", ".yaml", ".yml", ".toml",
    ".cfg", ".ini", ".csv", ".tsv", ".xml", ".html", ".pdf",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
    ".gitattributes", ".gitignore", ".gitmodules",
])

# Extensions that must NEVER be downloaded (executable/script files)
_HF_FORBIDDEN_EXTENSIONS: frozenset[str] = frozenset(
    [".py", ".sh", ".bash", ".bat", ".cmd", ".ps1", ".js", ".rb", ".pl"]
)

# Regex patterns to identify the two HF URL categories
# Pattern 1: Direct resolve file URL
#   https://huggingface.co/{org}/{model}/resolve/{ref}/{path} or https://huggingface.co/{model}/resolve/{ref}/{path}
_HF_DIRECT_FILE_RE: re.Pattern[str] = re.compile(
    r"^https://huggingface\.co/(?:models/)?([a-zA-Z0-9_\-\.]+(?:/[a-zA-Z0-9_\-\.]+){0,2})/resolve/[^/]+/.+",
    re.IGNORECASE,
)

# Pattern 2: Repository root URL
#   https://huggingface.co/{org}/{model} or https://huggingface.co/{model}
_HF_REPO_RE: re.Pattern[str] = re.compile(
    r"^https://huggingface\.co/(?:models/)?([a-zA-Z0-9_\-\.]+(?:/[a-zA-Z0-9_\-\.]+)?)/?$",
    re.IGNORECASE,
)

# Pattern 3: Dataset URL
#   https://huggingface.co/datasets/{org}/{dataset}
_HF_DATASET_RE: re.Pattern[str] = re.compile(
    r"^https://huggingface\.co/datasets(?:/.*)?$",
    re.IGNORECASE,
)

# Maximum number of model files to download from a single repository
_MAX_REPO_FILES: int = 200

_DOWNLOAD_TIMEOUT: int = 60  # fallback if config not available


def _is_hf_direct_file(url: str) -> bool:
    """Return True if the URL is a direct HF model file download link."""
    return bool(_HF_DIRECT_FILE_RE.match(url))


def _is_hf_dataset(url: str) -> bool:
    """Return True if the URL is a HF dataset URL."""
    return bool(_HF_DATASET_RE.match(url))


def _is_hf_repo(url: str) -> bool:
    """Return True if the URL is a HF repository root URL."""
    # Dataset URLs natively match _HF_REPO_RE because datasets/org looks like org/model
    if _is_hf_dataset(url):
        return False
    return bool(_HF_REPO_RE.match(url))


def _extract_repo_id(url: str) -> str:
    """Extract the {org}/{model} repository ID from a HF URL.

    Parameters
    ----------
    url:
        A HF URL (either direct file or repo root).

    Returns
    -------
    str
        Repository ID in the form ``"org/model"``.
    """
    m = _HF_DIRECT_FILE_RE.match(url) or _HF_REPO_RE.match(url)
    if m:
        return m.group(1)
    raise InputError(
        f"Cannot extract repository ID from URL: {url!r}",
        error_code="INVALID_HF_URL",
        details=url,
    )


def _fetch_repo_file_list(repo_id: str, timeout: int) -> list[dict[str, object]]:
    """Query the HF Model Hub API for the repository file tree.

    Returns the ``siblings`` list from the API JSON response, which contains
    ``{"rfilename": str, "size": int | None, ...}`` entries.

    Parameters
    ----------
    repo_id:
        Repository identifier, e.g. ``"facebook/opt-125m"``.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    list[dict[str, object]]
        Raw file-metadata entries from the API.

    Raises
    ------
    AcquisitionError
        If the API request fails or the response is malformed.
    """
    api_url = f"{_HF_API_BASE}/{repo_id}"
    _validate_url(api_url)
    req = urllib.request.Request(  # noqa: S310
        api_url,
        headers={"Accept": "application/json"},
    )
    handler = SafeRedirectHandler()
    opener = urllib.request.build_opener(handler)
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read(4 * 1024 * 1024)  # cap API response at 4 MB
    except (ResourceLimitError, InputError, AcquisitionError):
        raise
    except urllib.error.HTTPError as exc:
        raise AcquisitionError(
            f"HF API returned HTTP {exc.code} for {api_url}",
            error_code="HF_API_ERROR",
            details=str(exc),
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise AcquisitionError(
            f"Failed to contact HF API at {api_url}: {exc}",
            error_code="HF_API_NETWORK_ERROR",
            details=str(exc),
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AcquisitionError(
            f"HF API response is not valid JSON for {api_url}",
            error_code="HF_API_INVALID_JSON",
            details=str(exc),
        ) from exc

    if not isinstance(data, dict):
        raise AcquisitionError(
            "HF API response is not a JSON object.",
            error_code="HF_API_UNEXPECTED_FORMAT",
            details=repr(type(data)),
        )

    siblings_obj = data.get("siblings", [])
    if not isinstance(siblings_obj, list):
        raise AcquisitionError(
            "HF API response missing 'siblings' list.",
            error_code="HF_API_MISSING_SIBLINGS",
            details=repr(type(siblings_obj)),
        )

    return [item for item in siblings_obj if isinstance(item, dict)]


def _filter_model_candidates(
    siblings: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Filter the repository file list to only include candidate model files.

    Excludes:
    - Files with forbidden extensions (.py, .sh, etc.)
    - Files that are not in the set of supported model extensions
    - Non-model files (.gitattributes, .md, .txt, LICENSE, etc.)

    Parameters
    ----------
    siblings:
        Raw file-metadata entries from the HF API.

    Returns
    -------
    list[dict[str, object]]
        Filtered entries for candidate model files only.
    """
    candidates = []
    for entry in siblings:
        rfilename = str(entry.get("rfilename", ""))
        if not rfilename:
            continue

        base_name = rfilename.rsplit("/", 1)[-1]

        # Lower-cased suffix for matching
        suffix = "." + rfilename.rsplit(".", 1)[-1].lower() if "." in rfilename else ""

        # Absolute block: never download executable/script files
        if suffix in _HF_FORBIDDEN_EXTENSIONS:
            continue

        # .pb/.pbtxt: only the SavedModel graph file itself is a candidate.
        # Arbitrary protobufs (config protos, etc.) are not model weights.
        if suffix in (".pb", ".pbtxt"):
            if base_name.lower() not in _HF_SAVEDMODEL_PB_NAMES:
                continue
            candidates.append(entry)
            continue

        # Only download supported model format extensions
        if suffix not in _HF_MODEL_EXTENSIONS:
            continue

        candidates.append(entry)

    # Prefer SafeTensors when repository contains both SafeTensors and PyTorch/Pickle weights.
    # saved_model.pb is a TF graph (not an alternative weight format), so it is kept
    # even when SafeTensors are present.
    has_safetensors = any(
        str(entry.get("rfilename", "")).lower().endswith(".safetensors")
        for entry in candidates
    )
    if has_safetensors:
        candidates = [
            entry
            for entry in candidates
            if (
                str(entry.get("rfilename", "")).lower().endswith(
                    (".safetensors", ".gguf", ".ggml")
                )
                or str(entry.get("rfilename", "")).rsplit("/", 1)[-1].lower()
                in _HF_SAVEDMODEL_PB_NAMES
            )
        ]

    return candidates


def _is_suspicious_extensionless(rfilename: str) -> bool:
    """Return True if a repo filename is a suspicious hidden/extensionless candidate.

    Targets files that carry no recognised extension yet could hold a raw Pickle
    payload (e.g. ``.metadata``, ``checkpoint``, ``model``). These are NOT
    downloaded blindly — the caller peeks the first bytes and only fetches the
    full file if binary Pickle magic is present.

    Excludes:
      - anything under a ``.git`` path,
      - known non-model bare names (README, LICENSE, ...),
      - files bearing a recognised non-model extension (.json/.txt/.md/...),
      - files bearing a forbidden or known model extension (handled elsewhere).
    """
    if not rfilename:
        return False

    base_name = rfilename.rsplit("/", 1)[-1]
    if not base_name:
        return False

    # Skip all git metadata files entirely.
    if base_name.startswith(".git") or "/.git" in rfilename or rfilename.startswith(".git"):
        return False

    lower = base_name.lower()

    # Determine extension, treating a leading-dot hidden file with a single dot
    # (e.g. ".metadata") as having NO recognised extension.
    if base_name.startswith("."):
        # ".metadata" -> stem "metadata", no real extension.
        # ".foo.json" -> real extension ".json".
        remainder = base_name[1:]
        suffix = "." + remainder.rsplit(".", 1)[-1].lower() if "." in remainder else ""
    else:
        suffix = "." + base_name.rsplit(".", 1)[-1].lower() if "." in base_name else ""

    # Reject known non-model / forbidden / known-model extensions.
    if suffix and (
        suffix in _HF_NON_MODEL_EXTENSIONS
        or suffix in _HF_FORBIDDEN_EXTENSIONS
        or suffix in _HF_MODEL_EXTENSIONS
    ):
        return False

    # Exclude known non-model bare names (case-insensitive).
    if base_name in _HF_NON_MODEL_BARE_NAMES:
        return False
    if lower.rstrip(".") in {n.lower() for n in _HF_NON_MODEL_BARE_NAMES}:
        return False

    # Qualifying candidates carry no recognised extension:
    #   1. hidden file with no recognised second extension (".metadata"), or
    #   2. bare name with no extension at all ("metadata", "checkpoint", "model").
    return suffix == ""


def _peek_is_pickle(file_url: str, timeout: int, file_quota: int) -> bool:
    """Fetch only the first 16 bytes of a URL and test for binary Pickle magic.

    Uses a bounded HTTP Range request (stdlib urllib). Falls back to a small
    capped read (<= 64 KB) if the CDN ignores the Range header. NEVER downloads
    the whole file here and NEVER deserializes any content.
    """
    from model_scanner.detection.magic import is_pickle_magic

    _validate_url(file_url)
    # Cap the fallback read so a server ignoring Range cannot stream a huge file.
    fallback_cap = min(64 * 1024, max(16, file_quota))
    req = urllib.request.Request(  # noqa: S310
        file_url,
        headers={"Range": "bytes=0-15", "Accept": "*/*"},
    )
    handler = SafeRedirectHandler()
    opener = urllib.request.build_opener(handler)
    try:
        with opener.open(req, timeout=timeout) as resp:
            head = resp.read(fallback_cap)
    except (ResourceLimitError, InputError, AcquisitionError):
        raise
    except (urllib.error.URLError, OSError):
        return False

    return is_pickle_magic(head[:16])


class HuggingFaceAcquisition(InputAcquisitionProvider):
    """Hugging Face acquisition adapter.

    Supports two URL categories (Section 12.2):

    1. **Direct file URL** (``/resolve/`` segment present):
       Delegates immediately to ``PublicUrlAcquisition``.

    2. **Repository root URL** (no ``/resolve/`` segment):
       - Queries HF Model Hub API to list repository files.
       - Filters to candidate model format extensions.
       - Enforces cumulative repository download quota.
       - Enforces per-file size limit.
       - Downloads each candidate via ``PublicUrlAcquisition``.

    The acquired Artifacts are identical in structure to locally-acquired
    artifacts and pass through the same format detection pipeline.

    Raises
    ------
    InputError
        If the URL is not a recognised HF URL.
    AcquisitionError
        If the HF API is unreachable or returns an error.
    ResourceLimitError
        If cumulative download quota or individual file quota is exceeded.
    """

    def __init__(self) -> None:
        self._url_acquirer = PublicUrlAcquisition()

    def acquire(self, request: ScanRequest) -> Sequence[Artifact]:
        """Acquire model artifacts from a Hugging Face URL.

        Parameters
        ----------
        request:
            Must have ``input_source_type == PUBLIC_URL`` and ``target``
            a Hugging Face URL string.

        Returns
        -------
        Sequence[Artifact]
            Ordered sequence of acquired Artifacts (one per candidate file).
        """
        url = request.target
        if not isinstance(url, str):
            raise InputError(
                "HuggingFaceAcquisition requires a str URL target.",
                details=repr(url),
            )

        config = request.config or ScanConfig()

        if _is_hf_direct_file(url):
            return self._acquire_direct_file(url, request)
        elif _is_hf_dataset(url):
            # Dataset repositories are explicitly not supported.
            # Returning an empty list triggers the correct UNSUPPORTED
            # outcome in the ScanOrchestrator without calling the model API.
            return []
        elif _is_hf_repo(url):
            return self._acquire_repository(url, config)
        else:
            raise InputError(
                f"URL does not match a recognised Hugging Face pattern: {url!r}",
                error_code="INVALID_HF_URL",
                details=url,
            )

    def _acquire_direct_file(
        self, url: str, original_request: ScanRequest
    ) -> Sequence[Artifact]:
        """Delegate a direct HF file URL to PublicUrlAcquisition."""
        return self._url_acquirer.acquire(original_request)

    def _acquire_repository(
        self, repo_url: str, config: ScanConfig
    ) -> Sequence[Artifact]:
        """Enumerate and download all candidate model files from an HF repo.

        Parameters
        ----------
        repo_url:
            Repository root URL, e.g.
            ``https://huggingface.co/facebook/opt-125m``.
        config:
            Active ScanConfig with quota parameters.

        Returns
        -------
        Sequence[Artifact]
            Acquired Artifacts in file-listing order.
        """
        repo_id = _extract_repo_id(repo_url)
        timeout = config.download_timeout_sec
        repo_quota = config.max_download_repo_bytes
        file_quota = config.max_download_file_bytes

        # Fetch and filter file list (treat as untrusted input)
        siblings = _fetch_repo_file_list(repo_id, timeout)
        candidates = _filter_model_candidates(siblings)

        if len(candidates) > _MAX_REPO_FILES:
            raise AcquisitionError(
                f"Repository {repo_id!r} has {len(candidates)} candidate model "
                f"files — exceeds maximum of {_MAX_REPO_FILES}.",
                error_code="HF_TOO_MANY_FILES",
                details=repo_url,
            )

        # Build the secondary (extensionless/hidden) candidate list, then cap it so
        # that total files inspected (primary + secondary) never exceeds _MAX_REPO_FILES.
        # Sort by ascending reported size (smallest first) before capping so that:
        #   (a) the selection is deterministic, and
        #   (b) we prefer smaller candidates, which are cheaper to peek/scan.
        # A malicious repo can supply >200 extensionless files; without this cap,
        # an unbounded number of HTTP Range requests would be issued. (Finding 1)
        secondary_slots = max(0, _MAX_REPO_FILES - len(candidates))
        raw_secondary = [
            entry
            for entry in siblings
            if _is_suspicious_extensionless(str(entry.get("rfilename", "")))
        ]
        # Sort by size ascending; entries with no size info sort last (treat as large)
        raw_secondary.sort(
            key=lambda e: (e.get("size") is None, e.get("size") or 0)
        )
        secondary = raw_secondary[:secondary_slots]

        if not candidates and not secondary:
            return []

        artifacts: list[Artifact] = []
        cumulative_bytes = 0

        for entry in candidates:
            rfilename = str(entry.get("rfilename", ""))
            # Construct the download URL for this file
            file_url = f"https://huggingface.co/{repo_id}/resolve/main/{rfilename}"

            # Pre-check individual file size from API metadata if available
            api_size = entry.get("size")
            if isinstance(api_size, int) and api_size > 0:
                if api_size > file_quota:
                    raise ResourceLimitError(
                        f"File {rfilename!r} in {repo_id!r} exceeds per-file "
                        f"size limit ({api_size} > {file_quota})",
                        error_code="FILE_TOO_LARGE",
                        details=file_url,
                        limit_name="max_download_file_bytes",
                        limit_value=file_quota,
                        actual_value=api_size,
                    )
                # Check cumulative quota
                if cumulative_bytes + api_size > repo_quota:
                    raise ResourceLimitError(
                        f"Cumulative repository download would exceed repo quota "
                        f"of {repo_quota} bytes",
                        error_code="REPO_QUOTA_EXCEEDED",
                        details=repo_url,
                        limit_name="max_download_repo_bytes",
                        limit_value=repo_quota,
                        actual_value=cumulative_bytes + api_size,
                    )

            # Validate the file URL before downloading
            _validate_url(file_url)

            sub_request = ScanRequest(
                target=file_url,
                input_source_type=InputSourceType.PUBLIC_URL,
                config=config,
            )
            file_artifacts = self._url_acquirer.acquire(sub_request)

            for art in file_artifacts:
                cumulative_bytes += art.size_bytes
                if cumulative_bytes > repo_quota:
                    raise ResourceLimitError(
                        f"Cumulative repository download exceeded repo quota "
                        f"of {repo_quota} bytes",
                        error_code="REPO_QUOTA_EXCEEDED",
                        details=repo_url,
                        limit_name="max_download_repo_bytes",
                        limit_value=repo_quota,
                        actual_value=cumulative_bytes,
                    )
                artifacts.append(art)

        # Secondary pass: suspicious hidden/extensionless files. These are NOT
        # downloaded blindly. We peek the first 16 bytes; only files whose bytes
        # match binary Pickle magic AND which pass the per-file size limit are
        # fetched in full and routed to the pickle analyzer.
        for entry in secondary:
            rfilename = str(entry.get("rfilename", ""))
            if not rfilename:
                continue
            file_url = f"https://huggingface.co/{repo_id}/resolve/main/{rfilename}"

            # Size gate: skip (do not raise) oversized secondary candidates.
            api_size = entry.get("size")
            if isinstance(api_size, int) and api_size > 0 and api_size > file_quota:
                continue

            # Peek magic bytes only; discard immediately if not Pickle.
            try:
                if not _peek_is_pickle(file_url, timeout, file_quota):
                    continue
            except (ResourceLimitError, InputError, AcquisitionError):
                continue

            _validate_url(file_url)
            sub_request = ScanRequest(
                target=file_url,
                input_source_type=InputSourceType.PUBLIC_URL,
                config=config,
            )
            try:
                file_artifacts = self._url_acquirer.acquire(sub_request)
            except (ResourceLimitError, InputError, AcquisitionError):
                continue

            for art in file_artifacts:
                cumulative_bytes += art.size_bytes
                if cumulative_bytes > repo_quota:
                    raise ResourceLimitError(
                        f"Cumulative repository download exceeded repo quota "
                        f"of {repo_quota} bytes",
                        error_code="REPO_QUOTA_EXCEEDED",
                        details=repo_url,
                        limit_name="max_download_repo_bytes",
                        limit_value=repo_quota,
                        actual_value=cumulative_bytes,
                    )
                artifacts.append(art)

        return artifacts


__all__ = [
    "HuggingFaceAcquisition",
    "is_hf_url",
]


def is_hf_url(url: str) -> bool:
    """Return True if the URL matches a known Hugging Face URL pattern.

    Parameters
    ----------
    url:
        URL string to test.

    Returns
    -------
    bool
        True if this looks like a HF direct file, dataset, or repository URL.
    """
    return _is_hf_direct_file(url) or _is_hf_dataset(url) or _is_hf_repo(url)
