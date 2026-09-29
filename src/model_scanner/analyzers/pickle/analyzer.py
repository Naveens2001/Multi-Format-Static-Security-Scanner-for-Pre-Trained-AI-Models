"""
model_scanner.analyzers.pickle.analyzer
=======================================

Static Pickle security analyzer executing opcode disassembly, call-chain resolution,
structural anomaly detection, and Finding generation.

SECURITY GUARANTEE
------------------
Zero dynamic execution, zero deserialization, zero class instantiation.
Uses standard-library static analysis only.

Design Reference
----------------
Section 14 of docs/implementation_design.md
"""

from __future__ import annotations

from collections.abc import Sequence

from model_scanner.analyzers.base import BaseFormatAnalyzer
from model_scanner.analyzers.pickle.catalog import (
    GlobalCategory,
    classify_global,
    is_dangerous_state_key,
    is_plain_state_key,
)
from model_scanner.analyzers.pickle.disassembler import PickleDisassembler
from model_scanner.analyzers.pickle.structural import StructuralAnomalyEngine
from model_scanner.core.models import (
    Artifact,
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    ModelFormat,
    ScanConfig,
    Severity,
)

# Rule metadata definitions
_RULE_METADATA: dict[str, dict[str, str]] = {
    "P101": {
        "name": "Dangerous Global Import",
        "category": "code_execution",
        "description": (
            "The Pickle stream imports a known dangerous module or callable capable "
            "of arbitrary code execution."
        ),
        "remediation": (
            "Do not deserialize this model. Re-serialize model weights using SafeTensors."
        ),
    },
    "P102": {
        "name": "Suspicious Global Import",
        "category": "code_execution",
        "description": (
            "The Pickle stream imports a dynamic loader, reflection, or native FFI binding symbol."
        ),
        "remediation": (
            "Audit referenced global symbols and verify if necessary for model inference."
        ),
    },
    "P103": {
        "name": "STACK_GLOBAL Dynamic Import",
        "category": "code_execution",
        "description": (
            "The Pickle stream uses STACK_GLOBAL opcode to import symbols constructed dynamically."
        ),
        "remediation": "Audit stack-constructed symbols and convert model weights to SafeTensors.",
    },
    "P104": {
        "name": "Dangerous REDUCE Invocation",
        "category": "code_execution",
        "description": (
            "A REDUCE operation targets a callable capable of executing system commands or code."
        ),
        "remediation": (
            "Reject this model immediately. It contains an arbitrary code execution payload."
        ),
    },
    "P105": {
        "name": "Suspicious REDUCE Invocation",
        "category": "code_execution",
        "description": "A REDUCE operation targets an unverified or dynamic callable.",
        "remediation": (
            "Inspect the callable target and replace with standard state dictionary weights."
        ),
    },
    "P106": {
        "name": "Suspicious BUILD State Mutation",
        "category": "code_execution",
        "description": "A BUILD operation mutates object state with suspicious parameters.",
        "remediation": (
            "Verify object state dictionary and ensure only tensor weights are reconstructed."
        ),
    },
    "P107": {
        "name": "Suspicious NEWOBJ Instantiation",
        "category": "code_execution",
        "description": "A NEWOBJ/NEWOBJ_EX opcode instantiates an unverified class.",
        "remediation": "Ensure only trusted tensor container classes are instantiated.",
    },
    "P201": {
        "name": "Malformed or Truncated Opcode Stream",
        "category": "structural_anomaly",
        "description": (
            "The Pickle opcode stream is corrupted, truncated, or contains malformed arguments."
        ),
        "remediation": "Inspect file integrity or re-download the model file.",
    },
    "P202": {
        "name": "Duplicate or Misplaced PROTO Header",
        "category": "structural_anomaly",
        "description": (
            "A PROTO opcode appears more than once or at an unexpected position in the stream."
        ),
        "remediation": "Clean up corrupted stream headers using standard serialization tools.",
    },
    "P203": {
        "name": "Unsupported Protocol Version",
        "category": "structural_anomaly",
        "description": (
            "The stream declares an invalid or unsupported Pickle protocol version (> 5)."
        ),
        "remediation": "Re-serialize the model using standard Pickle protocol 2, 4, or 5.",
    },
    "P204": {
        "name": "FRAME Boundary Desynchronization",
        "category": "structural_anomaly",
        "description": (
            "An opcode crosses beyond declared FRAME boundary in Protocol 4/5 framing."
        ),
        "remediation": "Re-serialize with standard pickle tools to fix framing desynchronization.",
    },
    "P205": {
        "name": "Invalid Memo Slot Reference",
        "category": "structural_anomaly",
        "description": "A GET or BINGET opcode references an unassigned memo slot index.",
        "remediation": "Verify that the model serialization was not tampered with or truncated.",
    },
    "P206": {
        "name": "Concatenated Pickle Streams",
        "category": "structural_anomaly",
        "description": (
            "Multiple valid Pickle streams delimited by STOP opcodes are concatenated in one file."
        ),
        "remediation": "Ensure each model file contains exactly one discrete serialized payload.",
    },
    "P207": {
        "name": "Trailing Unparsed Bytes",
        "category": "structural_anomaly",
        "description": "Unparsed trailing data detected after the final STOP opcode in the stream.",
        "remediation": "Strip unparsed payload trailing bytes to prevent payload smuggling.",
    },
    "P208": {
        "name": "Excessive Opcode or Memo Resource Usage",
        "category": "structural_anomaly",
        "description": (
            "Pickle stream exceeded maximum opcode count or memo entry ceiling (DoS guard)."
        ),
        "remediation": "Check model size or increase scanner resource ceilings if legitimate.",
    },
}


class PickleAnalyzer(BaseFormatAnalyzer):
    """Static analyzer for standalone and embedded Pickle byte streams."""

    def __init__(self) -> None:
        self._disassembler = PickleDisassembler()
        self._structural_engine = StructuralAnomalyEngine()

    def analyze(
        self, artifact: Artifact, config: ScanConfig | None = None
    ) -> Sequence[Finding]:
        """Analyze a standalone Pickle Artifact.

        Parameters
        ----------
        artifact:
            The acquired model artifact.
        config:
            Optional ScanConfig.

        Returns
        -------
        Sequence[Finding]
            Generated security findings.
        """
        data = self._read_artifact_bytes(artifact)
        detected = getattr(artifact, "detected_format", None)
        is_legacy = detected == ModelFormat.LEGACY_PYTORCH
        return self.analyze_bytes(
            data=data,
            artifact_sha256=artifact.sha256,
            coordinate_space=CoordinateSpace.ARTIFACT,
            archive_member=None,
            model_format=(
                ModelFormat.LEGACY_PYTORCH if is_legacy else ModelFormat.RAW_PICKLE
            ),
            config=config,
            legacy_pytorch=is_legacy,
        )

    def analyze_bytes(
        self,
        data: bytes,
        artifact_sha256: str,
        coordinate_space: CoordinateSpace = CoordinateSpace.ARTIFACT,
        archive_member: str | None = None,
        model_format: ModelFormat = ModelFormat.RAW_PICKLE,
        config: ScanConfig | None = None,
        legacy_pytorch: bool = False,
    ) -> Sequence[Finding]:
        """Perform static analysis on an in-memory Pickle byte buffer.

        Parameters
        ----------
        data:
            Raw Pickle bytes.
        artifact_sha256:
            64-hex SHA-256 fingerprint of the parent artifact.
        coordinate_space:
            ARTIFACT or ARCHIVE_MEMBER.
        archive_member:
            Member name (e.g. "archive/data.pkl") if inside a container.
        model_format:
            RAW_PICKLE or PYTORCH_ZIP.
        config:
            Optional ScanConfig with allowlists, blocklists, and limits.

        Returns
        -------
        Sequence[Finding]
        """
        active_config = config or ScanConfig()

        # Legacy non-ZIP PyTorch files are a complete pickle stream followed by a
        # raw tensor payload. That payload is data, not opcodes: parsing it would
        # raise spurious trailing-byte / concatenated-stream anomalies. Slice the
        # buffer at the end of the first stream so the opcode analysis below sees
        # exactly the pickle and nothing else.
        if legacy_pytorch:
            probe = self._disassembler.disassemble(data)
            if probe.stop_positions:
                data = data[: probe.stop_positions[0] + 1]

        disassembly = self._disassembler.disassemble(data)
        anomalies = self._structural_engine.evaluate_anomalies(data, disassembly)

        findings: list[Finding] = []
        finding_count = 0

        # --- 1. Evaluate Globals ---
        for g_rec in disassembly.globals:
            category, rationale = classify_global(
                module=g_rec.module,
                name=g_rec.name,
                allowlist=active_config.allowlist_modules,
                blocklist=active_config.blocklist_modules,
            )

            if category == GlobalCategory.DANGEROUS:
                finding_count += 1
                rule_id = "P101"
                meta = _RULE_METADATA[rule_id]
                findings.append(
                    Finding(
                        finding_id=f"FINDING-{rule_id}-{artifact_sha256[:8]}-{finding_count:03d}",
                        rule_id=rule_id,
                        rule_name=meta["name"],
                        severity=Severity.CRITICAL,
                        confidence=Confidence.HIGH,
                        category=meta["category"],
                        format=model_format,
                        artifact_sha256=artifact_sha256,
                        evidence=Evidence(
                            coordinate_space=coordinate_space,
                            byte_offset=g_rec.opcode_record.pos,
                            byte_length=len(g_rec.opcode_record.arg_repr),
                            archive_member=archive_member,
                            matched_value=g_rec.full_name,
                            opcode=g_rec.opcode_record.name,
                            context_snippet=rationale,
                        ),
                        description=f"{meta['description']} Symbol: {g_rec.full_name!r}",
                        remediation=meta["remediation"],
                    )
                )
            elif category == GlobalCategory.SUSPICIOUS:
                finding_count += 1
                rule_id = "P102"
                meta = _RULE_METADATA[rule_id]
                findings.append(
                    Finding(
                        finding_id=f"FINDING-{rule_id}-{artifact_sha256[:8]}-{finding_count:03d}",
                        rule_id=rule_id,
                        rule_name=meta["name"],
                        severity=Severity.HIGH,
                        confidence=Confidence.HIGH,
                        category=meta["category"],
                        format=model_format,
                        artifact_sha256=artifact_sha256,
                        evidence=Evidence(
                            coordinate_space=coordinate_space,
                            byte_offset=g_rec.opcode_record.pos,
                            byte_length=len(g_rec.opcode_record.arg_repr),
                            archive_member=archive_member,
                            matched_value=g_rec.full_name,
                            opcode=g_rec.opcode_record.name,
                            context_snippet=rationale,
                        ),
                        description=f"{meta['description']} Symbol: {g_rec.full_name!r}",
                        remediation=meta["remediation"],
                    )
                )

            # STACK_GLOBAL specific finding if applicable
            if g_rec.is_stack_global and category != GlobalCategory.SAFE:
                finding_count += 1
                rule_id = "P103"
                meta = _RULE_METADATA[rule_id]
                findings.append(
                    Finding(
                        finding_id=f"FINDING-{rule_id}-{artifact_sha256[:8]}-{finding_count:03d}",
                        rule_id=rule_id,
                        rule_name=meta["name"],
                        severity=(
                            Severity.HIGH
                            if category == GlobalCategory.DANGEROUS
                            else Severity.MEDIUM
                        ),
                        confidence=Confidence.HIGH,
                        category=meta["category"],
                        format=model_format,
                        artifact_sha256=artifact_sha256,
                        evidence=Evidence(
                            coordinate_space=coordinate_space,
                            byte_offset=g_rec.opcode_record.pos,
                            byte_length=None,
                            archive_member=archive_member,
                            matched_value=g_rec.full_name,
                            opcode="STACK_GLOBAL",
                            context_snippet=f"Resolution status: {g_rec.resolution_status}",
                        ),
                        description=f"{meta['description']} Target: {g_rec.full_name!r}",
                        remediation=meta["remediation"],
                    )
                )

        # --- 2. Evaluate REDUCE calls ---
        for r_rec in disassembly.reduces:
            # Check if callable is dangerous
            target_str = r_rec.callable_target
            # Try to extract module/name if target_str is e.g. "os.system"
            parts = target_str.rsplit(".", 1)
            mod_part = parts[0] if len(parts) == 2 else target_str
            name_part = parts[1] if len(parts) == 2 else ""

            cat, rationale = classify_global(
                module=mod_part,
                name=name_part,
                allowlist=active_config.allowlist_modules,
                blocklist=active_config.blocklist_modules,
            )

            # Escalate specific dynamic-code execution primitives if they are invoked via REDUCE.
            # While their presence as a GLOBAL is only SUSPICIOUS (P102), actively calling them
            # is a CRITICAL code execution payload (P104).
            if cat == GlobalCategory.SUSPICIOUS and mod_part == "runpy":
                cat = GlobalCategory.DANGEROUS
                rationale = (
                    "Suspicious dynamic execution primitive actively invoked "
                    f"via REDUCE: {target_str!r}"
                )

            if cat == GlobalCategory.DANGEROUS:
                finding_count += 1
                rule_id = "P104"
                meta = _RULE_METADATA[rule_id]
                findings.append(
                    Finding(
                        finding_id=f"FINDING-{rule_id}-{artifact_sha256[:8]}-{finding_count:03d}",
                        rule_id=rule_id,
                        rule_name=meta["name"],
                        severity=Severity.CRITICAL,
                        confidence=Confidence.HIGH,
                        category=meta["category"],
                        format=model_format,
                        artifact_sha256=artifact_sha256,
                        evidence=Evidence(
                            coordinate_space=coordinate_space,
                            byte_offset=r_rec.opcode_record.pos,
                            byte_length=None,
                            archive_member=archive_member,
                            matched_value=target_str,
                            opcode="REDUCE",
                            context_snippet=f"Args: {r_rec.args_repr}",
                        ),
                        description=f"{meta['description']} Target callable: {target_str!r}",
                        remediation=meta["remediation"],
                    )
                )
            elif cat == GlobalCategory.SUSPICIOUS:
                finding_count += 1
                rule_id = "P105"
                meta = _RULE_METADATA[rule_id]
                findings.append(
                    Finding(
                        finding_id=f"FINDING-{rule_id}-{artifact_sha256[:8]}-{finding_count:03d}",
                        rule_id=rule_id,
                        rule_name=meta["name"],
                        severity=Severity.HIGH,
                        confidence=Confidence.HIGH,
                        category=meta["category"],
                        format=model_format,
                        artifact_sha256=artifact_sha256,
                        evidence=Evidence(
                            coordinate_space=coordinate_space,
                            byte_offset=r_rec.opcode_record.pos,
                            byte_length=None,
                            archive_member=archive_member,
                            matched_value=target_str,
                            opcode="REDUCE",
                            context_snippet=f"Args: {r_rec.args_repr}",
                        ),
                        description=f"{meta['description']} Target callable: {target_str!r}",
                        remediation=meta["remediation"],
                    )
                )

        # --- 3. Evaluate BUILD records (P106) ---
        # Fire P106 for dangerous state keys, or when the target is a genuinely
        # non-SAFE global. Ordinary nn.Module attribute names ("out_features",
        # "return_indices") and unresolved numeric operands are NOT globals and
        # are skipped - flagging them made P106 fire on every standard
        # checkpoint. Never execute __setstate__, never instantiate, never import.
        for b_rec in disassembly.builds:
            target = b_rec.target_repr

            if is_dangerous_state_key(target):
                finding_count += 1
                rule_id = "P106"
                meta = _RULE_METADATA[rule_id]
                findings.append(
                    Finding(
                        finding_id=(
                            f"FINDING-{rule_id}-{artifact_sha256[:8]}-{finding_count:03d}"
                        ),
                        rule_id=rule_id,
                        rule_name=meta["name"],
                        severity=Severity.HIGH,
                        confidence=Confidence.HIGH,
                        category=meta["category"],
                        format=model_format,
                        artifact_sha256=artifact_sha256,
                        evidence=Evidence(
                            coordinate_space=coordinate_space,
                            byte_offset=b_rec.opcode_record.pos,
                            byte_length=None,
                            archive_member=archive_member,
                            matched_value=target,
                            opcode="BUILD",
                            context_snippet=(
                                f"BUILD mutates deserialization-critical state key: "
                                f"{target!r}"
                            ),
                        ),
                        description=(
                            f"{meta['description']} Target: {target!r}"
                        ),
                        remediation=meta["remediation"],
                    )
                )
                continue

            if is_plain_state_key(target):
                # Ordinary tensor/module attribute name - expected in every
                # checkpoint state dict. Not a global import; no finding.
                continue
            # Attempt to parse "module.ClassName" from the target repr string.
            # Target strings are either "os.system", "NEWOBJ(cls)", "REDUCE(...)", or
            # "<unknown_target>" — classify what we can, default to UNKNOWN.
            if target in ("<unknown_target>", ""):
                # Unresolvable target — generate finding with MEDIUM confidence.
                finding_count += 1
                rule_id = "P106"
                meta = _RULE_METADATA[rule_id]
                findings.append(
                    Finding(
                        finding_id=(
                            f"FINDING-{rule_id}-{artifact_sha256[:8]}-{finding_count:03d}"
                        ),
                        rule_id=rule_id,
                        rule_name=meta["name"],
                        severity=Severity.MEDIUM,
                        confidence=Confidence.MEDIUM,
                        category=meta["category"],
                        format=model_format,
                        artifact_sha256=artifact_sha256,
                        evidence=Evidence(
                            coordinate_space=coordinate_space,
                            byte_offset=b_rec.opcode_record.pos,
                            byte_length=None,
                            archive_member=archive_member,
                            matched_value=target,
                            opcode="BUILD",
                            context_snippet=(
                                f"State: {b_rec.state_repr!r} — "
                                "target class could not be resolved statically"
                            ),
                        ),
                        description=(
                            f"{meta['description']} Target: {target!r}"
                        ),
                        remediation=meta["remediation"],
                    )
                )
            else:
                # Classify the target class using the same global catalog.
                parts = target.rsplit(".", 1)
                mod_part = parts[0] if len(parts) == 2 else target
                name_part = parts[1] if len(parts) == 2 else ""
                cat, rationale = classify_global(
                    module=mod_part,
                    name=name_part,
                    allowlist=active_config.allowlist_modules,
                    blocklist=active_config.blocklist_modules,
                )
                if cat != GlobalCategory.SAFE:
                    finding_count += 1
                    rule_id = "P106"
                    meta = _RULE_METADATA[rule_id]
                    severity = (
                        Severity.HIGH
                        if cat == GlobalCategory.DANGEROUS
                        else Severity.MEDIUM
                    )
                    confidence = (
                        Confidence.HIGH
                        if cat in (GlobalCategory.DANGEROUS, GlobalCategory.SUSPICIOUS)
                        else Confidence.LOW
                    )
                    findings.append(
                        Finding(
                            finding_id=(
                                f"FINDING-{rule_id}-{artifact_sha256[:8]}-{finding_count:03d}"
                            ),
                            rule_id=rule_id,
                            rule_name=meta["name"],
                            severity=severity,
                            confidence=confidence,
                            category=meta["category"],
                            format=model_format,
                            artifact_sha256=artifact_sha256,
                            evidence=Evidence(
                                coordinate_space=coordinate_space,
                                byte_offset=b_rec.opcode_record.pos,
                                byte_length=None,
                                archive_member=archive_member,
                                matched_value=target,
                                opcode="BUILD",
                                context_snippet=rationale,
                            ),
                            description=(
                                f"{meta['description']} Target: {target!r}"
                            ),
                            remediation=meta["remediation"],
                        )
                    )

        # --- 4. Evaluate NEWOBJ / NEWOBJ_EX records (P107) ---
        # Fire P107 when the class being instantiated is not definitively SAFE.
        # Never instantiate classes, never import referenced symbols.
        for n_rec in disassembly.newobjs:
            cls_repr = n_rec.cls_repr
            if cls_repr in ("<unknown_cls>", ""):
                # Unresolvable class — generate finding with MEDIUM confidence.
                finding_count += 1
                rule_id = "P107"
                meta = _RULE_METADATA[rule_id]
                findings.append(
                    Finding(
                        finding_id=(
                            f"FINDING-{rule_id}-{artifact_sha256[:8]}-{finding_count:03d}"
                        ),
                        rule_id=rule_id,
                        rule_name=meta["name"],
                        severity=Severity.MEDIUM,
                        confidence=Confidence.MEDIUM,
                        category=meta["category"],
                        format=model_format,
                        artifact_sha256=artifact_sha256,
                        evidence=Evidence(
                            coordinate_space=coordinate_space,
                            byte_offset=n_rec.opcode_record.pos,
                            byte_length=None,
                            archive_member=archive_member,
                            matched_value=cls_repr,
                            opcode=(
                                "NEWOBJ_EX" if n_rec.is_newobj_ex else "NEWOBJ"
                            ),
                            context_snippet=(
                                f"Args: {n_rec.args_repr!r} — "
                                "class could not be resolved statically"
                            ),
                        ),
                        description=(
                            f"{meta['description']} Class: {cls_repr!r}"
                        ),
                        remediation=meta["remediation"],
                    )
                )
            else:
                parts = cls_repr.rsplit(".", 1)
                mod_part = parts[0] if len(parts) == 2 else cls_repr
                name_part = parts[1] if len(parts) == 2 else ""
                cat, rationale = classify_global(
                    module=mod_part,
                    name=name_part,
                    allowlist=active_config.allowlist_modules,
                    blocklist=active_config.blocklist_modules,
                )
                if cat != GlobalCategory.SAFE:
                    finding_count += 1
                    rule_id = "P107"
                    meta = _RULE_METADATA[rule_id]
                    severity = (
                        Severity.HIGH
                        if cat == GlobalCategory.DANGEROUS
                        else Severity.MEDIUM
                    )
                    confidence = (
                        Confidence.HIGH
                        if cat in (GlobalCategory.DANGEROUS, GlobalCategory.SUSPICIOUS)
                        else Confidence.LOW
                    )
                    findings.append(
                        Finding(
                            finding_id=(
                                f"FINDING-{rule_id}-{artifact_sha256[:8]}-{finding_count:03d}"
                            ),
                            rule_id=rule_id,
                            rule_name=meta["name"],
                            severity=severity,
                            confidence=confidence,
                            category=meta["category"],
                            format=model_format,
                            artifact_sha256=artifact_sha256,
                            evidence=Evidence(
                                coordinate_space=coordinate_space,
                                byte_offset=n_rec.opcode_record.pos,
                                byte_length=None,
                                archive_member=archive_member,
                                matched_value=cls_repr,
                                opcode=(
                                    "NEWOBJ_EX" if n_rec.is_newobj_ex else "NEWOBJ"
                                ),
                                context_snippet=rationale,
                            ),
                            description=(
                                f"{meta['description']} Class: {cls_repr!r}"
                            ),
                            remediation=meta["remediation"],
                        )
                    )

        # --- 5. Evaluate Structural Anomalies ---
        for anom in anomalies:
            rule_id = anom.rule_id
            if rule_id in _RULE_METADATA:
                finding_count += 1
                meta = _RULE_METADATA[rule_id]
                severity = Severity.HIGH
                if rule_id in ("P202", "P206", "P207"):
                    severity = Severity.LOW
                elif rule_id in ("P203", "P205"):
                    severity = Severity.MEDIUM

                findings.append(
                    Finding(
                        finding_id=f"FINDING-{rule_id}-{artifact_sha256[:8]}-{finding_count:03d}",
                        rule_id=rule_id,
                        rule_name=meta["name"],
                        severity=severity,
                        confidence=Confidence.HIGH if anom.pos is not None else Confidence.MEDIUM,
                        category=meta["category"],
                        format=model_format,
                        artifact_sha256=artifact_sha256,
                        evidence=Evidence(
                            coordinate_space=coordinate_space,
                            byte_offset=anom.pos,
                            byte_length=None,
                            archive_member=archive_member,
                            matched_value=anom.anomaly_type,
                            opcode=None,
                            context_snippet=anom.description,
                        ),
                        description=f"{meta['description']} ({anom.description})",
                        remediation=meta["remediation"],
                    )
                )

        return findings

    @staticmethod
    def _read_artifact_bytes(artifact: Artifact) -> bytes:
        """Safely read all bytes of an Artifact without loading extra memory."""
        if artifact.local_path is not None:
            return artifact.local_path.read_bytes()
        elif artifact.stream is not None:
            pos = artifact.stream.tell() if artifact.stream.seekable() else 0
            data = artifact.stream.read()
            if artifact.stream.seekable():
                artifact.stream.seek(pos)
            return data
        return b""


__all__ = ["PickleAnalyzer"]
