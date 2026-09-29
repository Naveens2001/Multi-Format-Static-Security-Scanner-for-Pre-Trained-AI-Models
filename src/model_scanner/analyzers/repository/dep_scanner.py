"""
model_scanner.analyzers.repository.dep_scanner
===============================================

Offline dependency vulnerability scanning (B3).

Discovery
---------
Parses (stdlib only, all bounded):
  - ``requirements.txt`` (including ``==``, ``>=``, ``~=``, extras, comments,
    ``-r`` includes — bounded include depth).
  - ``pyproject.toml`` via stdlib ``tomllib`` (Python >=3.11).
  - ``environment.yml`` via the same minimal bounded key/value reader as A3
    (NOT a YAML library).
  - ``transformers_version`` / ``torch_dtype`` pins from ``config.json``.

Matching
--------
Vendors a **pinned, dated snapshot** of the PyPA Advisory Database (OSV schema)
under ``data/advisories/``.  The snapshot date is declared in
``data/advisories/snapshot_meta.json`` (optional; if absent, dep scanning is
reported as not-performed).

If the snapshot is absent, the scanner reports dependency scanning as
**not performed** — never as clean.

Rules
-----
  SC201 — Unpinned dependency (``>=`` with no upper bound, or ``*``).
  SC202 — Dependency pinned to a version with a known advisory.
  SC203 — Direct URL / VCS dependency (``git+https://``, direct wheel URL).
  SC204 — Typosquatting heuristic (edit distance 1 from top PyPI packages).

References
----------
OSV schema: https://ossf.github.io/osv-schema/
pip-audit / osv-scanner are design references for advisory representation.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_REQUIREMENTS_BYTES: int = 256 * 1024   # 256 KB
_MAX_PYPROJECT_BYTES: int = 512 * 1024      # 512 KB
_MAX_ENVYML_BYTES: int = 256 * 1024         # 256 KB
_MAX_INCLUDE_DEPTH: int = 3                 # -r include recursion limit
_MAX_ADVISORY_FILES: int = 10_000           # advisory file scan limit


@dataclass(frozen=True)
class ParsedDep:
    """A single parsed dependency.

    Attributes
    ----------
    name:
        Package name (lowercased, normalized).
    raw_spec:
        The raw version specifier string (e.g. ``==1.2.3``, ``>=1.0``, ``*``).
    is_pinned:
        True if the spec pins to exactly one version (``==``).
    is_vcs_or_url:
        True if the spec is a VCS (``git+``) or direct URL dependency.
    source_file:
        File where the dependency was found.
    line_number:
        Line number (1-based) in source_file.
    """

    name: str
    raw_spec: str
    is_pinned: bool
    is_vcs_or_url: bool
    source_file: str
    line_number: int


@dataclass(frozen=True)
class DepFinding:
    """A dependency-related supply chain finding (B3).

    Attributes
    ----------
    rule_id:
        SC201 / SC202 / SC203 / SC204.
    description:
        Human-readable description.
    dep_name:
        The package name.
    dep_spec:
        The raw version specifier.
    source_file:
        File containing the dependency declaration.
    line_number:
        Line number (1-based).
    advisory_id:
        For SC202: the OSV advisory ID that matched.
    snapshot_date:
        For SC202: the date of the advisory snapshot used.
    """

    rule_id: str
    description: str
    dep_name: str
    dep_spec: str
    source_file: str
    line_number: int
    advisory_id: str = ""
    snapshot_date: str = "unknown"


# ---------------------------------------------------------------------------
# PEP 440 version comparison (stdlib only)
# ---------------------------------------------------------------------------


def _normalize_version(v: str) -> tuple[int, ...]:
    """Parse a PEP 440 version string into a comparable tuple.

    Only handles the most common case: ``N.N.N[.postN][aN|bN|rcN]``.
    Ignores local version labels.
    """
    v = v.split("+")[0].split("!", 1)[-1]  # strip epoch and local
    # Replace pre-release / post-release markers
    v = re.sub(r"(a|b|rc|alpha|beta)(\d*)", r".dev.\2", v, flags=re.I)
    v = re.sub(r"\.post(\d+)", r".\1", v)
    parts: list[int] = []
    for segment in v.split("."):
        if segment.isdigit():
            parts.append(int(segment))
        else:
            # Try to extract leading digits
            m = re.match(r"(\d+)", segment)
            parts.append(int(m.group(1)) if m else 0)
    return tuple(parts) if parts else (0,)


def _version_satisfies_range(
    version: str,
    introduced: str,
    fixed: str | None,
) -> bool:
    """Return True if ``version`` falls in [introduced, fixed) (OSV ECOSYSTEM range).

    Parameters
    ----------
    version:
        The installed/pinned version string.
    introduced:
        The ``introduced`` event version (OSV).  ``"0"`` means all versions.
    fixed:
        The ``fixed`` event version (OSV).  ``None`` means no upper bound.
    """
    try:
        ver = _normalize_version(version)
        intro = _normalize_version(introduced) if introduced and introduced != "0" else (0,)
        if ver < intro:
            return False
        if fixed is not None:
            fix = _normalize_version(fixed)
            if ver >= fix:
                return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Requirements.txt parser
# ---------------------------------------------------------------------------

# Matches: pkg_name[extras]==version, >=version, ~=version, !=version, etc.
# Also matches git+https:// and direct URLs.
_REQ_LINE_RE: re.Pattern[str] = re.compile(
    r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)"  # package name
    r"(\[.*?\])?"                                       # optional extras
    r"\s*(.*?)$",                                       # version spec (greedy)
    re.ASCII,
)
_VCS_URL_RE: re.Pattern[str] = re.compile(r"^(git\+|hg\+|svn\+|bzr\+|https?://.*\.whl)")
_EXACT_PIN_RE: re.Pattern[str] = re.compile(r"^==\s*[\d.]")
_LOOSE_SPEC_RE: re.Pattern[str] = re.compile(
    r"^(>=|~=|<=|!=|>|<)\s*[\d.].*$|^\*$"
)
_INCLUDE_RE: re.Pattern[str] = re.compile(r"^-r\s+(.+)$")


def _normalize_name(name: str) -> str:
    """PEP 503 package name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_requirements_file(
    path: Path,
    source_label: str,
    *,
    depth: int = 0,
) -> list[ParsedDep]:
    """Parse a requirements.txt file into ``ParsedDep`` objects."""
    if depth > _MAX_INCLUDE_DEPTH:
        return []

    try:
        raw = path.read_bytes()[:_MAX_REQUIREMENTS_BYTES]
        lines = raw.decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []

    deps: list[ParsedDep] = []

    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Strip inline comments
        line = line.split("#", 1)[0].strip()

        # Handle -r include
        include_m = _INCLUDE_RE.match(line)
        if include_m:
            included = path.parent / include_m.group(1).strip()
            deps.extend(
                _parse_requirements_file(
                    included,
                    source_label=str(included.name),
                    depth=depth + 1,
                )
            )
            continue

        # VCS / URL deps
        if _VCS_URL_RE.match(line):
            # Extract package name from egg fragment if present
            egg_m = re.search(r"#egg=([A-Za-z0-9._-]+)", line)
            pkg_name = _normalize_name(egg_m.group(1)) if egg_m else "<unknown>"
            deps.append(
                ParsedDep(
                    name=pkg_name,
                    raw_spec=line[:200],
                    is_pinned=False,
                    is_vcs_or_url=True,
                    source_file=source_label,
                    line_number=lineno,
                )
            )
            continue

        # Regular dep
        m = _REQ_LINE_RE.match(line)
        if not m:
            continue

        name = _normalize_name(m.group(1))
        spec = m.group(4).strip()

        is_pinned = bool(_EXACT_PIN_RE.match(spec))
        is_vcs = False

        deps.append(
            ParsedDep(
                name=name,
                raw_spec=spec or "*",
                is_pinned=is_pinned,
                is_vcs_or_url=is_vcs,
                source_file=source_label,
                line_number=lineno,
            )
        )

    return deps


def _parse_pyproject_toml(path: Path) -> list[ParsedDep]:
    """Parse ``pyproject.toml`` for dependencies (stdlib tomllib)."""
    try:
        raw = path.read_bytes()[:_MAX_PYPROJECT_BYTES]
        data = tomllib.loads(raw.decode("utf-8", errors="replace"))
    except (OSError, tomllib.TOMLDecodeError):
        return []

    deps: list[ParsedDep] = []

    # [project.dependencies]
    project_deps = (
        data.get("project", {}).get("dependencies", [])
        or data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    )
    source_label = "pyproject.toml"

    if isinstance(project_deps, list):
        for i, item in enumerate(project_deps, start=1):
            if not isinstance(item, str):
                continue
            # Parse PEP 508 dep string
            m = _REQ_LINE_RE.match(item.strip())
            if not m:
                continue
            name = _normalize_name(m.group(1))
            spec = m.group(4).strip()
            is_pinned = bool(_EXACT_PIN_RE.match(spec))
            deps.append(
                ParsedDep(
                    name=name,
                    raw_spec=spec or "*",
                    is_pinned=is_pinned,
                    is_vcs_or_url=_VCS_URL_RE.match(spec) is not None,
                    source_file=source_label,
                    line_number=i,
                )
            )
    elif isinstance(project_deps, dict):
        # Poetry-style dict: name -> version_spec
        for i, (pkg, ver) in enumerate(project_deps.items(), start=1):
            if pkg == "python":
                continue
            name = _normalize_name(pkg)
            spec = ver if isinstance(ver, str) else "*"
            is_pinned = bool(_EXACT_PIN_RE.match(spec))
            deps.append(
                ParsedDep(
                    name=name,
                    raw_spec=spec,
                    is_pinned=is_pinned,
                    is_vcs_or_url=False,
                    source_file=source_label,
                    line_number=i,
                )
            )

    return deps


def _parse_environment_yml_deps(path: Path) -> list[ParsedDep]:
    """Extract dependencies from a conda ``environment.yml`` using the minimal
    bounded key/value reader (NOT a YAML library)."""
    try:
        raw = path.read_bytes()[:_MAX_ENVYML_BYTES]
        lines = raw.decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []

    deps: list[ParsedDep] = []
    in_deps = False
    source_label = "environment.yml"

    for lineno, raw_line in enumerate(lines, start=1):
        stripped = raw_line.rstrip()

        # Detect the 'dependencies:' section
        if re.match(r"^dependencies\s*:", stripped):
            in_deps = True
            continue

        # Exit deps section on a new top-level key
        if in_deps and stripped and not stripped.startswith(" ") and not stripped.startswith("-"):
            in_deps = False
            continue

        if not in_deps:
            continue

        # Parse list items: "  - package>=version" or "  - package==version"
        m = re.match(r"^\s+-\s+([A-Za-z0-9][A-Za-z0-9._-]*)(.*)$", stripped)
        if not m:
            continue

        name = _normalize_name(m.group(1))
        rest = m.group(2).strip()

        # pip: section (nested) — just skip marker lines
        if name.lower() == "pip":
            continue

        # Simple version spec extraction
        spec_m = re.match(r"^(==|>=|~=|!=|>|<|<=)(.+)$", rest)
        if spec_m:
            spec = spec_m.group(1) + spec_m.group(2).strip()
        elif rest:
            spec = rest
        else:
            spec = "*"

        is_pinned = bool(_EXACT_PIN_RE.match(spec))
        deps.append(
            ParsedDep(
                name=name,
                raw_spec=spec,
                is_pinned=is_pinned,
                is_vcs_or_url=False,
                source_file=source_label,
                line_number=lineno,
            )
        )

    return deps


# ---------------------------------------------------------------------------
# Advisory database loading
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_ADVISORIES_DIR = _DATA_DIR / "advisories"
_SNAPSHOT_META_FILE = _ADVISORIES_DIR / "snapshot_meta.json"


def _load_snapshot_date() -> str | None:
    """Return the advisory snapshot date string, or None if absent."""
    if not _SNAPSHOT_META_FILE.is_file():
        return None
    try:
        meta = json.loads(_SNAPSHOT_META_FILE.read_bytes())
        return meta.get("snapshot_date") if isinstance(meta, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _load_advisories() -> list[dict[str, Any]]:
    """Load all OSV advisory JSON files from the advisories directory.

    Returns an empty list if the directory is absent or empty.
    """
    if not _ADVISORIES_DIR.is_dir():
        return []

    advisories: list[dict[str, Any]] = []
    count = 0

    for json_file in _ADVISORIES_DIR.rglob("*.json"):
        if json_file.name == "snapshot_meta.json":
            continue
        if count >= _MAX_ADVISORY_FILES:
            break
        try:
            data = json.loads(json_file.read_bytes())
            if isinstance(data, dict):
                advisories.append(data)
        except (OSError, json.JSONDecodeError):
            pass
        count += 1

    return advisories


def _extract_affected_versions(
    advisory: dict[str, Any],
    package_name: str,
) -> list[tuple[str, str | None]]:
    """Extract (introduced, fixed) version pairs for a package from an OSV advisory.

    Returns a list of (introduced, fixed) tuples.  ``fixed`` is None when
    there is no upper bound.
    """
    ranges: list[tuple[str, str | None]] = []

    for affected in advisory.get("affected", []):
        pkg = affected.get("package", {})
        if not isinstance(pkg, dict):
            continue
        # Match package name (case-insensitive, PEP 503 normalized)
        affected_name = _normalize_name(pkg.get("name", ""))
        if affected_name != _normalize_name(package_name):
            continue

        for rng in affected.get("ranges", []):
            if rng.get("type") not in ("ECOSYSTEM", "SEMVER"):
                continue
            events = rng.get("events", [])
            introduced: str | None = None
            fixed: str | None = None
            for event in events:
                if "introduced" in event:
                    introduced = str(event["introduced"])
                elif "fixed" in event:
                    fixed = str(event["fixed"])
                    if introduced is not None:
                        ranges.append((introduced, fixed))
                        introduced = None
                        fixed = None
            if introduced is not None:
                ranges.append((introduced, None))  # no upper bound

    return ranges


def _extract_pinned_version(spec: str) -> str | None:
    """Extract the exact version from a ``==X.Y.Z`` specifier."""
    m = re.match(r"^==\s*([\d.]+(?:[a-zA-Z0-9.]+)?)", spec)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Typosquat detection
# ---------------------------------------------------------------------------

_TOP_PACKAGES: list[str] = []


def _load_top_packages() -> list[str]:
    """Load the top PyPI package name list (built-in data file)."""
    global _TOP_PACKAGES
    if _TOP_PACKAGES:
        return _TOP_PACKAGES
    try:
        data_file = _DATA_DIR / "top_pypi_packages.json"
        _TOP_PACKAGES = json.loads(data_file.read_bytes())
    except (OSError, json.JSONDecodeError):
        _TOP_PACKAGES = []
    return _TOP_PACKAGES


def _edit_distance_1(a: str, b: str) -> bool:
    """Return True if strings ``a`` and ``b`` have edit distance exactly 1."""
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return False

    if len(a) == len(b):
        # Substitution
        diffs = sum(c1 != c2 for c1, c2 in zip(a, b, strict=False))
        return diffs == 1

    # Insertion or deletion
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    i = j = diffs = 0
    while i < len(longer) and j < len(shorter):
        if longer[i] != shorter[j]:
            diffs += 1
            if diffs > 1:
                return False
            i += 1
        else:
            i += 1
            j += 1
    return True


def _check_typosquat(name: str) -> str | None:
    """Return the popular package being typosquatted, or None."""
    norm = _normalize_name(name)
    top_norm = [_normalize_name(p) for p in _load_top_packages()]
    for popular in top_norm:
        if norm != popular and _edit_distance_1(norm, popular):
            return popular
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_dependencies(
    repo_root: Path,
    config_json: dict[str, Any] | None = None,
) -> tuple[list[DepFinding], bool]:
    """Scan repository dependency declarations for supply chain risks (B3).

    Parameters
    ----------
    repo_root:
        Root directory of the model repository.
    config_json:
        Parsed ``config.json`` dict (for extracting version pins).

    Returns
    -------
    tuple[list[DepFinding], bool]
        ``(findings, scanning_performed)``.  If ``scanning_performed`` is
        False, the advisory snapshot was absent — no clean/dirty claim is made.
    """
    snapshot_date = _load_snapshot_date()
    advisories = _load_advisories()
    scanning_performed = snapshot_date is not None

    # Collect all dependency declarations
    all_deps: list[ParsedDep] = []

    req_file = repo_root / "requirements.txt"
    if req_file.is_file():
        all_deps.extend(_parse_requirements_file(req_file, "requirements.txt"))

    pyproject = repo_root / "pyproject.toml"
    if pyproject.is_file():
        all_deps.extend(_parse_pyproject_toml(pyproject))

    env_yml = repo_root / "environment.yml"
    if env_yml.is_file():
        all_deps.extend(_parse_environment_yml_deps(env_yml))

    # Pin version from config.json if present
    if config_json:
        for key in ("transformers_version", "torch_version"):
            ver = config_json.get(key)
            if isinstance(ver, str) and ver:
                pkg_name = "transformers" if "transformers" in key else "torch"
                all_deps.append(
                    ParsedDep(
                        name=pkg_name,
                        raw_spec=f"=={ver}",
                        is_pinned=True,
                        is_vcs_or_url=False,
                        source_file="config.json",
                        line_number=0,
                    )
                )

    findings: list[DepFinding] = []

    for dep in all_deps:
        # SC203: VCS or direct URL dependency
        if dep.is_vcs_or_url:
            findings.append(
                DepFinding(
                    rule_id="SC203",
                    description=(
                        f"Direct VCS/URL dependency {dep.name!r}: {dep.raw_spec!r}. "
                        f"Cannot audit; unpinned and bypasses registry checks."
                    ),
                    dep_name=dep.name,
                    dep_spec=dep.raw_spec,
                    source_file=dep.source_file,
                    line_number=dep.line_number,
                )
            )
            continue

        # SC201: Unpinned dependency
        if not dep.is_pinned and _LOOSE_SPEC_RE.match(dep.raw_spec):
            findings.append(
                DepFinding(
                    rule_id="SC201",
                    description=(
                        f"Unpinned dependency {dep.name!r} ({dep.raw_spec!r}) allows "
                        f"future updates to silently introduce vulnerable versions."
                    ),
                    dep_name=dep.name,
                    dep_spec=dep.raw_spec,
                    source_file=dep.source_file,
                    line_number=dep.line_number,
                )
            )

        # SC202: Advisory match (only for pinned versions)
        if dep.is_pinned and scanning_performed:
            pinned_ver = _extract_pinned_version(dep.raw_spec)
            if pinned_ver:
                for advisory in advisories:
                    adv_id = advisory.get("id", "")
                    ranges = _extract_affected_versions(advisory, dep.name)
                    for intro, fix in ranges:
                        if _version_satisfies_range(pinned_ver, intro, fix):
                            findings.append(
                                DepFinding(
                                    rule_id="SC202",
                                    description=(
                                        f"Dependency {dep.name}=={pinned_ver} is affected "
                                        f"by advisory {adv_id}."
                                    ),
                                    dep_name=dep.name,
                                    dep_spec=dep.raw_spec,
                                    source_file=dep.source_file,
                                    line_number=dep.line_number,
                                    advisory_id=adv_id,
                                    snapshot_date=snapshot_date or "unknown",
                                )
                            )
                            break  # one finding per advisory

        # SC204: Typosquat heuristic
        popular = _check_typosquat(dep.name)
        if popular:
            findings.append(
                DepFinding(
                    rule_id="SC204",
                    description=(
                        f"Package name {dep.name!r} is edit-distance-1 from {popular!r}. "
                        f"Possible typosquatting."
                    ),
                    dep_name=dep.name,
                    dep_spec=dep.raw_spec,
                    source_file=dep.source_file,
                    line_number=dep.line_number,
                )
            )

    return findings, scanning_performed


__all__ = [
    "DepFinding",
    "ParsedDep",
    "scan_dependencies",
]
