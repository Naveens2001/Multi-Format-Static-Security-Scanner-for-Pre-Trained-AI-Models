"""
model_scanner.core.correlation
==============================

Evaluates findings across all artifacts in a scan to detect multi-signal
attack chains (C-series rules).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from model_scanner.core.models import (
    ArtifactScanResult,
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    ModelFormat,
)
from model_scanner.rules.catalog import RuleCatalog


class CrossArtifactCorrelator:
    """Evaluates findings across all artifacts to detect composite threats."""

    @classmethod
    def correlate_scan(
        cls, artifact_results: Sequence[ArtifactScanResult]
    ) -> tuple[Finding, ...]:
        """Evaluate all findings across all artifacts and return new correlated findings.

        Parameters
        ----------
        artifact_results:
            The complete list of ArtifactScanResult objects from the current scan.

        Returns
        -------
        tuple[Finding, ...]
            A tuple of newly generated correlated (C-series) findings.
        """
        findings_by_rule: dict[str, list[Finding]] = {}
        for result in artifact_results:
            for finding in result.findings:
                findings_by_rule.setdefault(finding.rule_id, []).append(finding)

        new_findings: list[Finding] = []

        def make_correlated_finding(
            rule_id: str, sources: tuple[Finding, ...]
        ) -> Finding:
            rule = RuleCatalog.get_rule(rule_id)

            source_ids = sorted(f.finding_id for f in sources)
            hash_input = f"{rule_id}:{','.join(source_ids)}".encode()
            finding_hash = hashlib.sha256(hash_input).hexdigest()[:12]

            context_snippets = [
                f"Source: {f.rule_id} ({f.finding_id})" for f in sources
            ]

            fired_ids = ", ".join(sorted(f.rule_id for f in sources))
            evidence = Evidence(
                coordinate_space=CoordinateSpace.GLOBAL,
                byte_offset=None,   # synthetic finding — no real byte location
                byte_length=None,
                archive_member=None,
                # matched_value encodes the source rule IDs, not a literal match
                matched_value=",".join(f.rule_id for f in sources),
                context_snippet="; ".join(context_snippets),
            )

            description = (
                f"Correlated multi-indicator risk: {fired_ids} co-present on the same artifact."
            )

            return Finding(
                finding_id=f"FINDING-{rule_id}-{finding_hash}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                category=rule.category,
                confidence=Confidence.HIGH,
                description=description,
                remediation=rule.remediation,
                format=sources[0].format if sources else ModelFormat.REPOSITORY,
                artifact_sha256=sources[0].artifact_sha256 if sources else "",
                evidence=evidence,
                cve_ids=rule.cve_ids,
                cwe_ids=rule.cwe_ids,
                owasp_category=rule.owasp_category,
                mitre_technique=rule.mitre_technique,
                correlated_finding_ids=tuple(source_ids),
            )

        # C001: Active Supply Chain Execution Intent — SC102 AND (SC101 OR SC103)
        sc102_list = findings_by_rule.get("SC102", [])
        sc101_list = findings_by_rule.get("SC101", [])
        sc103_list = findings_by_rule.get("SC103", [])

        if sc102_list and (sc101_list or sc103_list):
            for sc102 in sc102_list:
                for active_intent in sc101_list + sc103_list:
                    new_findings.append(
                        make_correlated_finding("C001", (sc102, active_intent))
                    )

        # C002: Networked Executable Payload — (P101 OR SC104) AND SC105
        p101_list = findings_by_rule.get("P101", [])
        sc104_list = findings_by_rule.get("SC104", [])
        sc105_list = findings_by_rule.get("SC105", [])

        if sc105_list and (p101_list or sc104_list):
            for sc105 in sc105_list:
                for exec_payload in p101_list + sc104_list:
                    new_findings.append(
                        make_correlated_finding("C002", (sc105, exec_payload))
                    )

        # C003: Obfuscated Payload Staging — P210 AND (P103 OR P105)
        p210_list = findings_by_rule.get("P210", [])
        p103_list = findings_by_rule.get("P103", [])
        p105_list = findings_by_rule.get("P105", [])

        if p210_list and (p103_list or p105_list):
            for p210 in p210_list:
                for dynamic_payload in p103_list + p105_list:
                    new_findings.append(
                        make_correlated_finding("C003", (p210, dynamic_payload))
                    )

        # C004: Imported AND Invoked Dangerous Callable
        #
        # C001-C003 above are all gated on SC-series rules, which only the
        # repository analyzer emits. That made every correlation unreachable for
        # a single uploaded file - a .bin containing a reverse shell reported
        # "no correlated findings". C004 correlates within one artifact on the
        # signal that actually matters in Pickle: the same callable being both
        # imported (GLOBAL/STACK_GLOBAL) and called (REDUCE).
        import_rules = ("P101", "P102")
        invoke_rules = ("P104", "P105")

        for result in artifact_results:
            imports: dict[tuple[str, str | None], Finding] = {}
            for f in result.findings:
                if f.rule_id in import_rules and f.evidence is not None:
                    key = (
                        str(f.evidence.matched_value or ""),
                        f.evidence.archive_member,
                    )
                    if key[0] and key not in imports:
                        imports[key] = f

            if not imports:
                continue

            for f in result.findings:
                if f.rule_id not in invoke_rules or f.evidence is None:
                    continue
                key = (
                    str(f.evidence.matched_value or ""),
                    f.evidence.archive_member,
                )
                imported = imports.get(key)
                if imported is not None:
                    new_findings.append(
                        make_correlated_finding("C004", (imported, f))
                    )

        # Deduplicate correlated findings (overlapping pairs may create identical IDs)
        unique_findings = {f.finding_id: f for f in new_findings}
        return tuple(unique_findings.values())

    @staticmethod
    def backlink_sources(
        artifact_results: Sequence[ArtifactScanResult],
        correlated: Sequence[Finding],
    ) -> dict[str, tuple[str, ...]]:
        """Map each source finding_id to the correlated findings that cite it.

        The correlated finding records its sources, but without the reverse link
        a source finding renders as "Correlated Finding IDs: None" even when it
        is half of a detected chain. Callers apply this to the source findings.

        Parameters
        ----------
        artifact_results:
            Artifacts whose findings may be cited as correlation sources.
        correlated:
            The correlated (C-series) findings produced by ``correlate_scan``.

        Returns
        -------
        dict[str, tuple[str, ...]]
            finding_id -> tuple of correlated finding_ids citing it.
        """
        known: set[str] = {
            f.finding_id for r in artifact_results for f in r.findings
        }
        backlinks: dict[str, list[str]] = {}
        for cf in correlated:
            for source_id in cf.correlated_finding_ids:
                if source_id in known:
                    backlinks.setdefault(source_id, []).append(cf.finding_id)
        return {k: tuple(sorted(set(v))) for k, v in backlinks.items()}


__all__ = ["CrossArtifactCorrelator"]
