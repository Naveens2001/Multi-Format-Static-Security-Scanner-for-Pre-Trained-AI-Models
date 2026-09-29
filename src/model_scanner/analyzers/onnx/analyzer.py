"""Static security analyzer for ONNX model files."""
from __future__ import annotations
from collections.abc import Sequence
from model_scanner.analyzers.base import BaseFormatAnalyzer
from model_scanner.core.models import (
    Artifact, Confidence, CoordinateSpace, Evidence, Finding, ModelFormat, ScanConfig, Severity,
)

_STANDARD_ONNX_DOMAINS: frozenset[str] = frozenset([
    "", "ai.onnx", "ai.onnx.ml", "ai.onnx.training",
    "ai.onnx.preview.training", "com.microsoft",
])

# Path components that indicate traversal attempts in external data locations.
_TRAVERSAL_COMPONENTS: tuple[str, ...] = ("..", "../", "..\\")

# uint64 boundary: offset + length >= 2^64 wraps in 64-bit unsigned arithmetic.
_UINT64_MOD: int = 1 << 64

_RULE_SEVERITY = {
    "ON101": Severity.MEDIUM, "ON102": Severity.MEDIUM,
    "ON103": Severity.HIGH, "ON104": Severity.HIGH,
    "WI101": Severity.HIGH, "WI102": Severity.HIGH, "WI103": Severity.MEDIUM,
}
_RULE_CONFIDENCE = {
    "ON101": Confidence.HIGH, "ON102": Confidence.HIGH,
    "ON103": Confidence.HIGH, "ON104": Confidence.HIGH,
    "WI101": Confidence.HIGH, "WI102": Confidence.HIGH, "WI103": Confidence.MEDIUM,
}
_RULE_NAMES = {
    "ON101": "External or Custom Operator in ONNX Graph",
    "ON102": "Malformed ONNX Graph Structure",
    "ON103": "Path Traversal in ONNX External Data Location",
    "ON104": "External Data Offset/Length Overflow",
    "WI101": "NaN Values Detected in Tensor Weights",
    "WI102": "Infinite Values Detected in Tensor Weights",
    "WI103": "Extreme Finite Values Detected in Tensor Weights",
}
_RULE_CATEGORY = {
    "ON101": "structural_anomaly", "ON102": "structural_anomaly",
    "ON103": "structural_anomaly", "ON104": "structural_anomaly",
    "WI101": "weight_integrity", "WI102": "weight_integrity", "WI103": "weight_integrity",
}
_RULE_REMEDIATION = {
    "ON101": (
        "Audit the custom operator implementation for malicious behavior. "
        "Prefer models using only standard ONNX operators."
    ),
    "ON102": (
        "Re-export the model from a trusted ONNX-compatible framework. "
        "Use onnx.checker.check_model() to validate before deployment."
    ),
    "ON103": (
        "Do not load this model with ONNX Runtime. Re-acquire the model from a trusted source "
        "and verify that all initializer external_data locations are relative paths without "
        "traversal sequences (no '..' components or absolute paths)."
    ),
    "ON104": (
        "Do not load this model with any ONNX runtime. Re-export the model from a trusted "
        "ONNX-compatible framework with correctly bounded external_data offset and length values. "
        "Verify that no initializer offset + length reaches or exceeds 2^64."
    ),
    "WI101": "Re-acquire the model from a trusted source. NaN weights indicate corruption or tampering.",
    "WI102": "Re-acquire the model from a trusted source. Infinite weights indicate numerical corruption or tampering.",
    "WI103": (
        "Investigate the origin of extreme-magnitude weights. Values at the maximum finite "
        "representable magnitude suggest numerical instability, overflow during training, or tampering."
    ),
}

_KIND_TO_RULE = {"nan": "WI101", "inf": "WI102", "extreme": "WI103"}


class ONNXAnalyzer(BaseFormatAnalyzer):
    """Static security analyzer for ONNX model files.

    Requires 'onnx' optional dependency. Degrades gracefully if absent.
    SECURITY: Uses onnx.load_model_from_string() which is pure protobuf
    parsing — it does NOT execute model code or inference.
    """

    model_structure: dict[str, object] | None = None

    def analyze(self, artifact: Artifact, config: ScanConfig | None = None) -> Sequence[Finding]:
        """Perform static analysis on an ONNX artifact."""
        try:
            import onnx  # optional dep
        except ImportError as exc:
            raise ImportError(
                "onnx is required to analyze ONNX model files. "
                "Install it with: pip install onnx"
            ) from exc

        data = self._read_artifact_bytes(artifact)
        if not data:
            return []

        findings: list[Finding] = []
        counter = [0]

        def _next_id(rule_id: str) -> str:
            counter[0] += 1
            return f"FINDING-{rule_id}-{artifact.sha256[:8]}-{counter[0]:03d}"

        def _make_finding(
            rule_id: str,
            byte_offset: int | None,
            matched_value: str,
            description: str,
        ) -> Finding:
            return Finding(
                finding_id=_next_id(rule_id),
                rule_id=rule_id,
                rule_name=_RULE_NAMES[rule_id],
                severity=_RULE_SEVERITY[rule_id],
                confidence=_RULE_CONFIDENCE[rule_id],
                category=_RULE_CATEGORY[rule_id],
                format=ModelFormat.ONNX,
                artifact_sha256=artifact.sha256,
                evidence=Evidence(
                    coordinate_space=CoordinateSpace.ARTIFACT,
                    byte_offset=byte_offset,
                    byte_length=None,
                    archive_member=None,
                    matched_value=matched_value[:512],
                ),
                description=description,
                remediation=_RULE_REMEDIATION[rule_id],
            )

        # Parse ONNX protobuf — pure data deserialization, no model execution
        try:
            model = onnx.load_model_from_string(data)
        except Exception as exc:
            findings.append(_make_finding(
                "ON102", 0, "PARSE_ERROR",
                f"ONNX file could not be parsed as a valid ModelProto: {exc}",
            ))
            return findings

        # ON102: validate graph exists
        if not model.HasField("graph"):
            findings.append(_make_finding(
                "ON102", 0, "MISSING_GRAPH",
                "ONNX model has no graph definition.",
            ))
            return findings

        graph = model.graph

        # ON101: check for external/custom operators
        seen_custom_domains: set[str] = set()
        for node in graph.node:
            domain = node.domain
            if domain not in _STANDARD_ONNX_DOMAINS and domain not in seen_custom_domains:
                seen_custom_domains.add(domain)
                findings.append(_make_finding(
                    "ON101", None,
                    f"op={node.op_type!r} domain={domain!r}",
                    f"ONNX graph node '{node.op_type}' uses non-standard domain '{domain}'. "
                    f"Custom/external operators may execute native code at inference time.",
                ))

        # ON103: check initializer external_data locations for path traversal
        for initializer in graph.initializer:
            location = ""
            for seg in initializer.external_data:
                if seg.key == "location":
                    location = seg.value
                    break
            if not location:
                continue
            # Detect traversal: '..' components or absolute paths
            is_traversal = (
                ".." in location.replace("\\", "/").split("/")
                or location.startswith("/")
                or (len(location) >= 3 and location[1] == ":" and location[2] in ("/", "\\"))
            )
            if is_traversal:
                findings.append(_make_finding(
                    "ON103", None,
                    f"initializer={initializer.name!r} location={location!r}",
                    f"ONNX initializer '{initializer.name}' references external tensor data at "
                    f"path '{location}', which contains a path-traversal sequence. Loading this "
                    f"model with ONNX Runtime would open an attacker-controlled file path.",
                ))

        # ON104: check external_data offset + length for uint64 wraparound.
        # Detection condition: offset + length >= 2^64 (Python int arithmetic is
        # exact, so there is no silent overflow — we compare against _UINT64_MOD).
        # The vulnerability: runtimes (e.g. OpenVINO) that store offset and length
        # as uint64 will compute a wrapped-to-zero sum, bypassing the bounds check
        # "m_offset + m_data_length > file_size" and accessing an attacker-controlled
        # memory location. The offset/length fields are protobuf string-encoded
        # integers, so we parse them with int() and catch malformed values safely.
        for initializer in graph.initializer:
            ext_offset: int | None = None
            ext_length: int | None = None
            ext_location: str = ""
            for seg in initializer.external_data:
                try:
                    if seg.key == "offset":
                        ext_offset = int(seg.value)
                    elif seg.key == "length":
                        ext_length = int(seg.value)
                    elif seg.key == "location":
                        ext_location = seg.value
                except (ValueError, TypeError):
                    pass
            if ext_offset is None or ext_length is None:
                continue
            if ext_offset < 0 or ext_length < 0:
                continue
            overflow_sum = ext_offset + ext_length
            if overflow_sum >= _UINT64_MOD:
                wrapped = overflow_sum % _UINT64_MOD
                findings.append(_make_finding(
                    "ON104", None,
                    (
                        f"initializer={initializer.name!r} location={ext_location!r} "
                        f"offset={ext_offset} length={ext_length} "
                        f"offset+length={overflow_sum} wrapped_sum={wrapped}"
                    ),
                    (
                        f"ONNX initializer '{initializer.name}' has external_data "
                        f"offset ({ext_offset}) + length ({ext_length}) = {overflow_sum}, "
                        f"which reaches or exceeds 2^64 ({_UINT64_MOD}). A runtime performing "
                        f"this addition in uint64 computes wrapped_sum={wrapped}, bypassing "
                        f"file-bounds validation and enabling out-of-bounds memory access."
                    ),
                ))

        # WI101/WI102/WI103: check initializer weights for NaN/Inf/extreme
        from model_scanner.analyzers._weight_utils import check_tensor

        # ONNX data type enum: FLOAT=1, DOUBLE=11, FLOAT16=10, BFLOAT16=16
        _onnx_dtype_map: dict[int, str] = {1: "F32", 11: "F64", 10: "F16", 16: "BF16"}

        # Build structural inventory while iterating initializers
        _MAX_INVENTORY = 1000
        tensor_inventory: list[dict[str, object]] = []

        for initializer in graph.initializer:
            dtype_str = _onnx_dtype_map.get(initializer.data_type)
            shape = list(initializer.dims) if initializer.dims else []

            if len(tensor_inventory) < _MAX_INVENTORY:
                tensor_inventory.append({
                    "name": initializer.name,
                    "dtype": dtype_str or f"onnx_type_{initializer.data_type}",
                    "shape": shape,
                })

            if dtype_str is None:
                continue
            raw_data = initializer.raw_data
            if not raw_data:
                continue
            hits = check_tensor(raw_data, dtype_str)
            for hit in hits:
                rule_id = _KIND_TO_RULE[hit.kind]
                findings.append(_make_finding(
                    rule_id, None,
                    f"initializer={initializer.name!r} dtype={dtype_str} {hit.kind}_count={hit.count}",
                    f"ONNX initializer '{initializer.name}' ({dtype_str}) contains {hit.count} "
                    f"{hit.kind.upper()} value(s).",
                ))

        self.model_structure = {
            "tensor_count": len(list(graph.initializer)),
            "tensors": tensor_inventory,
        }

        return findings

    @staticmethod
    def _read_artifact_bytes(artifact: Artifact) -> bytes:
        """Read all bytes from an Artifact's local_path or stream."""
        if artifact.local_path is not None:
            return artifact.local_path.read_bytes()
        if artifact.stream is not None:
            pos = artifact.stream.tell() if artifact.stream.seekable() else 0
            data = artifact.stream.read()
            if artifact.stream.seekable():
                artifact.stream.seek(pos)
            return data
        return b""


__all__ = ["ONNXAnalyzer"]
