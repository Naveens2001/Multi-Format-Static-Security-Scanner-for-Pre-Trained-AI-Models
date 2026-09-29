"""
model_scanner.analyzers.repository.analyzer
============================================

Supply chain analysis lane orchestrator (B4).

``RepositoryAnalyzer`` operates on a **directory root** rather than a single
``Artifact``.  It coordinates:

  A3  — Repository metadata reading (metadata_reader.py)
  B1  — Config RCE vector detection (config_scanner.py) + AST scan
  A2  — Shard integrity (shard_checker.py)
  B2  — Provenance indicators (provenance.py)
  B3  — Dependency scanning (dep_scanner.py)

The results are returned as a list of ``Finding`` objects that use the
``REPOSITORY`` model format and the ``SC1xx`` / ``SC2xx`` / ``G00x`` rule
namespaces.  Findings are passed through the standard normalizer before
being surfaced in ``ArtifactScanResult``.

SECURITY GUARANTEE
------------------
- No dynamic execution, no network calls at scan time.
- All reads are bounded.  Stdlib only.
"""

from __future__ import annotations

import hashlib
import io
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from model_scanner.core.models import (
    Artifact,
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    InputSourceType,
    ModelFormat,
    ScanConfig,
    Severity,
)

# ---------------------------------------------------------------------------
# Severity and rule metadata tables (inline — not fetched from catalog)
# These MUST stay in sync with rules/catalog.py.
# ---------------------------------------------------------------------------

_RULE_SEVERITY: dict[str, Severity] = {
    "SC101": Severity.HIGH,
    "SC102": Severity.HIGH,
    "SC103": Severity.MEDIUM,
    "SC104": Severity.CRITICAL,
    "SC105": Severity.MEDIUM,
    "SC106": Severity.INFO,
    "SC107": Severity.HIGH,
    "SC108": Severity.MEDIUM,
    "SC109": Severity.MEDIUM,
    "SC201": Severity.LOW,
    "SC202": Severity.HIGH,
    "SC203": Severity.MEDIUM,
    "SC204": Severity.HIGH,
    "G005": Severity.HIGH,
    "G006": Severity.MEDIUM,
    "G007": Severity.HIGH,
    "G008": Severity.MEDIUM,
}

_RULE_NAMES: dict[str, str] = {
    "SC101": "Remote Code Execution via auto_map / Custom Class Reference",
    "SC102": "trust_remote_code Enabled",
    "SC103": "Bundled Executable Python Code",
    "SC104": "Dangerous System Call in Bundled Code",
    "SC105": "Network Access in Bundled Code",
    "SC106": "Unsigned Artifact — No Signature File Found",
    "SC107": "Hash Self-Inconsistency (Possible Tampering)",
    "SC108": "LFS Pointer Stub — Content Not Fetched",
    "SC109": "Config Declared Format Disagrees with Detected Format",
    "SC201": "Unpinned Dependency",
    "SC202": "Dependency with Known Advisory",
    "SC203": "Direct VCS/URL Dependency",
    "SC204": "Possible Typosquatting (Edit Distance 1)",
    "G005": "Shard Declared in Index but Missing on Disk",
    "G006": "Undeclared Shard File Present",
    "G007": "Shard Size Disagrees with Index",
    "G008": "Shard Count Mismatch",
}

_RULE_REMEDIATIONS: dict[str, str] = {
    "SC101": (
        "Inspect the referenced Python module before loading the model. "
        "Avoid loading models with custom code unless you trust the source."
    ),
    "SC102": (
        "Do not set trust_remote_code=True without auditing all bundled Python files. "
        "Prefer models without custom code requirements."
    ),
    "SC103": (
        "Audit all bundled Python files before loading the model. "
        "Run a static AST scan and code review on modeling_*.py and similar files."
    ),
    "SC104": (
        "Do not load this model without auditing the bundled Python code. "
        "The dangerous call may execute arbitrary system commands."
    ),
    "SC105": (
        "Audit the bundled Python code for unintended network access. "
        "Network calls in model-loading code may exfiltrate data or download payloads."
    ),
    "SC106": (
        "Consider verifying artifact provenance via a signed release or Sigstore. "
        "An unsigned artifact does not guarantee authenticity."
    ),
    "SC107": (
        "Do not use this artifact — the SHA-256 hash does not match the index declaration. "
        "Re-download from a trusted source and verify again."
    ),
    "SC108": (
        "Run `git lfs pull` to fetch actual weight content before scanning or deploying. "
        "An LFS stub contains only metadata, not real model weights."
    ),
    "SC109": (
        "Verify that the model format matches the declared library. "
        "Inconsistency may indicate a mislabelled or repackaged artifact."
    ),
    "SC201": (
        "Pin all dependencies to exact versions (==) in deployment artifacts. "
        "Unpinned dependencies allow silent vulnerability introduction."
    ),
    "SC202": (
        "Upgrade the affected dependency to a version that does not have a known advisory. "
        "Check the advisory for affected version ranges and patches."
    ),
    "SC203": (
        "Replace direct VCS/URL dependencies with pinned PyPI versions. "
        "Direct VCS deps bypass the PyPI registry security model."
    ),
    "SC204": (
        "Verify the package name is correct and not a typosquatting package. "
        "Compare carefully against the intended popular package name."
    ),
    "G005": (
        "Re-download all declared shards from the original source. "
        "A missing shard indicates an incomplete or tampered download."
    ),
    "G006": (
        "Verify whether the undeclared shard file is legitimate. "
        "Undeclared shards may indicate unauthorized additions."
    ),
    "G007": (
        "Re-download the affected shard from the original source. "
        "A size mismatch may indicate truncation or tampering."
    ),
    "G008": (
        "Verify the shard count against the original repository. "
        "A count mismatch may indicate missing or extra shards."
    ),
}

_RULE_CATEGORIES: dict[str, str] = {
    "SC101": "supply_chain",
    "SC102": "supply_chain",
    "SC103": "supply_chain",
    "SC104": "code_execution",
    "SC105": "supply_chain",
    "SC106": "provenance",
    "SC107": "provenance",
    "SC108": "provenance",
    "SC109": "provenance",
    "SC201": "supply_chain",
    "SC202": "supply_chain",
    "SC203": "supply_chain",
    "SC204": "supply_chain",
    "G005": "structural_anomaly",
    "G006": "structural_anomaly",
    "G007": "structural_anomaly",
    "G008": "structural_anomaly",
}

_RULE_CONFIDENCE: dict[str, Confidence] = {
    "SC101": Confidence.HIGH,
    "SC102": Confidence.HIGH,
    "SC103": Confidence.HIGH,
    "SC104": Confidence.HIGH,
    "SC105": Confidence.MEDIUM,
    "SC106": Confidence.LOW,
    "SC107": Confidence.HIGH,
    "SC108": Confidence.HIGH,
    "SC109": Confidence.MEDIUM,
    "SC201": Confidence.HIGH,
    "SC202": Confidence.HIGH,
    "SC203": Confidence.HIGH,
    "SC204": Confidence.LOW,
    "G005": Confidence.HIGH,
    "G006": Confidence.MEDIUM,
    "G007": Confidence.HIGH,
    "G008": Confidence.MEDIUM,
}


def _repo_sha256(repo_root: Path) -> str:
    """Compute a stable pseudo-SHA-256 for a repository root path."""
    return hashlib.sha256(str(repo_root.resolve()).encode("utf-8")).hexdigest()


def _make_finding(
    rule_id: str,
    description: str,
    repo_sha: str,
    index: int,
    matched_value: str = "",
    context: str = "",
) -> Finding:
    """Build a repository-level Finding."""
    return Finding(
        finding_id=f"FINDING-{rule_id}-{repo_sha[:8]}-{index:03d}",
        rule_id=rule_id,
        rule_name=_RULE_NAMES.get(rule_id, rule_id),
        severity=_RULE_SEVERITY.get(rule_id, Severity.INFO),
        confidence=_RULE_CONFIDENCE.get(rule_id, Confidence.LOW),
        category=_RULE_CATEGORIES.get(rule_id, "supply_chain"),
        format=ModelFormat.REPOSITORY,
        artifact_sha256=repo_sha,
        evidence=Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=None,
            byte_length=None,
            archive_member=None,
            matched_value=matched_value[:512] if matched_value else "",
            context_snippet=context[:512] if context else None,
        ),
        description=description[:1000],
        remediation=_RULE_REMEDIATIONS.get(rule_id, ""),
    )


def make_repository_artifact(repo_root: Path) -> Artifact:
    """Create a synthetic Artifact representing a model repository directory."""
    repo_sha = _repo_sha256(repo_root)
    return Artifact(
        artifact_id=str(uuid.uuid4()),
        source_path=str(repo_root),
        input_source=InputSourceType.LOCAL_DIRECTORY,
        size_bytes=0,
        sha256=repo_sha,
        stream=io.BytesIO(b""),
    )


class RepositoryAnalyzer:
    """Supply chain analysis lane orchestrator (B4).

    Runs all repository-level checks and returns ``Finding`` objects.
    Does NOT implement ``BaseFormatAnalyzer`` because it operates on a
    directory, not a single ``Artifact``.

    Usage::

        analyzer = RepositoryAnalyzer()
        findings = analyzer.analyze_directory(repo_root, weight_files, config)
    """

    def analyze_directory(
        self,
        repo_root: Path,
        weight_files: list[Path],
        config: ScanConfig | None = None,
    ) -> list[Finding]:
        """Run all supply chain checks for a model repository.

        Parameters
        ----------
        repo_root:
            Root directory of the model repository.
        weight_files:
            Model weight files discovered by the acquisition layer.
        config:
            Optional scan configuration.

        Returns
        -------
        list[Finding]
            All supply chain findings.  Caller normalizes via FindingCorrelator.
        """
        repo_sha = _repo_sha256(repo_root)
        findings: list[Finding] = []
        counter = 0

        def _add(
            rule_id: str,
            desc: str,
            matched: str = "",
            ctx: str = "",
            severity: Severity | None = None,
            remediation: str | None = None,
            confidence: Confidence | None = None,
        ) -> None:
            nonlocal counter
            counter += 1
            f = _make_finding(rule_id, desc, repo_sha, counter, matched, ctx)
            if severity is not None:
                f = replace(f, severity=severity)
            if remediation is not None:
                f = replace(f, remediation=remediation)
            if confidence is not None:
                f = replace(f, confidence=confidence)
            findings.append(f)

        # --- A3: Read repository metadata ---
        from model_scanner.analyzers.repository.metadata_reader import read_repo_metadata
        meta = read_repo_metadata(repo_root)

        # --- B1: Config RCE vectors ---
        from model_scanner.analyzers.repository.config_scanner import (
            find_bundled_code_files,
            scan_repo_configs,
        )
        config_findings = scan_repo_configs(repo_root)
        for cf in config_findings:
            _add(
                cf.rule_id,
                cf.description,
                matched=cf.matched_value,
                ctx=f"{cf.config_file} field={cf.field_path}",
            )

        # --- B1: Bundled Python code presence (SC103) ---
        bundled_py = find_bundled_code_files(repo_root)
        for bp in bundled_py:
            try:
                rel = str(bp.relative_to(repo_root))
            except ValueError:
                rel = str(bp)
            _add(
                "SC103",
                f"Bundled Python executable code found: {rel!r}. "
                f"This file may be executed when the model is loaded.",
                matched=rel,
            )

        # --- B1: AST scan of bundled Python files ---
        from model_scanner.analyzers.repository.py_ast_scanner import scan_python_files_in_repo
        ast_results = scan_python_files_in_repo(repo_root)
        for py_path, ast_findings in ast_results.items():
            try:
                rel = str(py_path.relative_to(repo_root))
            except ValueError:
                rel = str(py_path)
            for af in ast_findings:
                _add(
                    af.rule_id,
                    f"{rel}:{af.line_number}: {af.description}",
                    matched=af.matched_text,
                    ctx=f"line {af.line_number}, col {af.col_offset}",
                )

        # --- A2: Shard integrity ---
        from model_scanner.analyzers.repository.shard_checker import check_shard_integrity
        shard_findings = check_shard_integrity(repo_root, weight_files)
        for sf in shard_findings:
            _add(sf.rule_id, sf.description, matched=sf.shard_filename, ctx=sf.detail)

        # --- B2: Provenance indicators ---
        from model_scanner.analyzers.repository.provenance import (
            check_config_format_consistency,
            check_index_hash_consistency,
            check_lfs_pointer_stubs,
            check_signature_artifacts,
        )

        detected_formats: set[str] = set()
        # We don't have format info here, but can infer from extensions
        for wf in weight_files:
            ext = wf.suffix.lower()
            if ext in (".safetensors",):
                detected_formats.add("SAFETENSORS")
            elif ext in (".pt", ".pth", ".bin", ".ckpt"):
                detected_formats.add("PYTORCH_ZIP")
            elif ext in (".gguf",):
                detected_formats.add("GGUF")

        for pf in check_signature_artifacts(repo_root, weight_files):
            _add(pf.rule_id, pf.description, matched=pf.artifact_rel_path, ctx=pf.detail)

        for pf in check_index_hash_consistency(repo_root, weight_files):
            _add(pf.rule_id, pf.description, matched=pf.artifact_rel_path, ctx=pf.detail)

        for pf in check_lfs_pointer_stubs(repo_root, weight_files):
            _add(pf.rule_id, pf.description, matched=pf.artifact_rel_path, ctx=pf.detail)

        for pf in check_config_format_consistency(
            meta.model_type, meta.library_name, detected_formats
        ):
            _add(pf.rule_id, pf.description, matched=pf.artifact_rel_path, ctx=pf.detail)

        # --- B3: Dependency scanning ---
        from model_scanner.analyzers.repository.dep_scanner import scan_dependencies
        # Load config.json for version pin extraction
        config_json: dict[str, Any] | None = None
        config_path = repo_root / "config.json"
        if config_path.is_file():
            try:
                import json
                raw = config_path.read_bytes()[:65536]
                parsed = json.loads(raw.decode("utf-8", errors="replace"))
                config_json = parsed if isinstance(parsed, dict) else None
            except Exception:  # noqa: S110
                pass

        dep_findings, dep_performed = scan_dependencies(repo_root, config_json)
        # SC201/SC203/SC204 checks don't need the advisory snapshot — report always.
        # SC202 advisory matches only exist when the snapshot is present.
        for df in dep_findings:
            _add(
                df.rule_id,
                df.description,
                matched=f"{df.dep_name}{df.dep_spec}",
                ctx=f"{df.source_file}:{df.line_number}",
            )
        if not dep_performed:
            # Fail-closed: advisory snapshot absent — report SC202 not performed.
            # Coverage gap, not a vulnerability in the scanned repository. Emitting
            # this at SC202's HIGH severity fired on 100% of folder scans, including
            # clean baselines, which floored every repository at LOW_RISK and gave
            # the finding no discriminative value. G002 sets the precedent: report
            # a missing capability at INFO and say so plainly.
            _add(
                "SC202",
                "Dependency vulnerability scanning was NOT performed: no advisory "
                "database snapshot is installed. This is a scanner coverage gap, not "
                "a finding about this repository — its dependencies are unassessed, "
                "not proven safe. Install a snapshot under data/advisories/.",
                matched="snapshot_meta.json missing",
                ctx="See data/advisories/README.md for instructions.",
                severity=Severity.INFO,
                # Catalog remediation ("upgrade the affected dependency") assumes
                # an advisory matched. Nothing matched here because nothing was
                # checked, so the action is to install the snapshot and re-scan.
                remediation=(
                    "Install an advisory snapshot under data/advisories/ and re-run "
                    "the scan so dependencies are actually assessed."
                ),
                # LOW, not HIGH: this reports the absence of a capability, and
                # says nothing about the repository with any confidence.
                confidence=Confidence.LOW,
            )

        return findings


__all__ = [
    "RepositoryAnalyzer",
    "make_repository_artifact",
]
