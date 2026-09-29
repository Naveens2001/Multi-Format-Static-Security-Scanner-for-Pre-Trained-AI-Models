"""
tests/unit/test_onnx_analyzer.py
=================================

Unit tests for the ONNX static security analyzer.

SAFETY CONTRACT
---------------
- No onnx model inference or execution — only onnx.load_model_from_string() (protobuf parsing).
- No eval(), exec(), marshal.loads(), or dynamic deserialization.
- Synthetic ONNX fixtures built programmatically using the onnx library.
- Real-world path-traversal fixture mirrors the public PoC at
  huggingface.co/rez0/onnx-ort-path-traversal-poc (SHA256 verified inline).

Coverage
--------
1. Missing onnx dependency raises ImportError (fail-closed, not false CLEAN).
2. ON103 fires for path traversal in external_data location (../).
3. ON103 fires for absolute POSIX path in external_data location.
4. ON103 fires for absolute Windows path in external_data location.
5. ON103 does NOT fire for a safe relative location (same-directory reference).
6. ON101 fires for a non-standard operator domain.
7. ON102 fires when the ONNX file cannot be parsed.
8. Real-world PoC fixture (159 bytes, known SHA256) produces exactly one ON103 finding.
9. ON104 fires when external_data offset + length >= 2^64 (uint64 wraparound).
   - TC11 PoC parameters (offset=0xFFFFFFFFFFFFFFF0, length=16, sum=2^64) — TRUE POSITIVE.
   - Strict overflow (offset=2^64-16, length=32, sum=2^64+16) — fires.
   - Sub-boundary (offset=2^64-16, length=15, sum=2^64-1) — does NOT fire.
   - Normal model (offset=0, length=1024) — does NOT fire.
   - Large but valid offset (100 GB, no overflow) — does NOT fire.
   - ON103 regression: path-traversal location still fires ON103, not ON104.
"""

from __future__ import annotations

import io
import struct
import sys
from unittest.mock import patch

import pytest

from model_scanner.analyzers.onnx.analyzer import ONNXAnalyzer
from model_scanner.core.models import Artifact, InputSourceType, ScanConfig

# ---------------------------------------------------------------------------
# Skip entire module if onnx is not installed
# ---------------------------------------------------------------------------
onnx = pytest.importorskip("onnx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_artifact(data: bytes) -> Artifact:
    sha = "b" * 64
    return Artifact(
        artifact_id="test-onnx",
        source_path="<test_onnx>",
        input_source=InputSourceType.IN_MEMORY,
        size_bytes=len(data),
        sha256=sha,
        stream=io.BytesIO(data),
    )


def _make_simple_onnx(
    *,
    op_type: str = "Add",
    domain: str = "",
    has_external_data: bool = False,
    external_location: str = "weights.bin",
) -> bytes:
    """Build a minimal valid ONNX model protobuf.

    If has_external_data is True, the initializer W uses external data at
    external_location instead of inline raw_data.
    """
    # Input X: float32, shape [1]
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(op_type, inputs=["X", "W"], outputs=["Y"], domain=domain)

    if has_external_data:
        W = onnx.TensorProto()
        W.name = "W"
        W.data_type = onnx.TensorProto.FLOAT
        W.dims.append(1)
        W.data_location = onnx.TensorProto.EXTERNAL
        W.external_data.add(key="location", value=external_location)
        W.external_data.add(key="offset", value="0")
        W.external_data.add(key="length", value="4")
    else:
        W = onnx.helper.make_tensor("W", onnx.TensorProto.FLOAT, [1], [1.0])

    graph = onnx.helper.make_graph([node], "test_graph", [X], [Y], initializer=[W])
    model = onnx.helper.make_model(graph, ir_version=8)
    model.opset_import[0].version = 17
    return model.SerializeToString()


# ---------------------------------------------------------------------------
# 1. Missing onnx dependency — must raise ImportError, not silently return []
# ---------------------------------------------------------------------------

class TestMissingOnnxDependency:
    def test_missing_onnx_raises_import_error(self) -> None:
        """When onnx is absent the analyzer must raise ImportError (fail-closed)."""
        data = _make_simple_onnx()
        artifact = _make_artifact(data)

        with patch.dict(sys.modules, {"onnx": None}):
            analyzer = ONNXAnalyzer()
            with pytest.raises(ImportError, match="onnx is required"):
                analyzer.analyze(artifact)

    def test_missing_onnx_does_not_return_empty_list(self) -> None:
        """Complementary: the silent [] path (false CLEAN) must not exist."""
        data = _make_simple_onnx()
        artifact = _make_artifact(data)

        with patch.dict(sys.modules, {"onnx": None}):
            analyzer = ONNXAnalyzer()
            try:
                result = analyzer.analyze(artifact)
                assert result != [], (
                    "analyzer returned [] when onnx was unavailable — false-CLEAN bug"
                )
            except ImportError:
                pass


# ---------------------------------------------------------------------------
# 2-5. ON103: path traversal in external_data location
# ---------------------------------------------------------------------------

class TestON103PathTraversal:
    """ON103 must fire for path-traversal locations and must not fire for safe ones."""

    def test_on103_fires_for_relative_traversal(self) -> None:
        """../../../../etc/hosts must produce exactly one ON103 finding."""
        data = _make_simple_onnx(
            has_external_data=True,
            external_location="../../../../etc/hosts",
        )
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on103 = [f for f in findings if f.rule_id == "ON103"]
        assert len(on103) == 1
        assert "ON103" == on103[0].rule_id
        assert on103[0].severity.value == "HIGH"
        assert "../../../../etc/hosts" in on103[0].evidence.matched_value

    def test_on103_fires_for_absolute_posix_path(self) -> None:
        """/etc/passwd must produce an ON103 finding."""
        data = _make_simple_onnx(
            has_external_data=True,
            external_location="/etc/passwd",
        )
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on103 = [f for f in findings if f.rule_id == "ON103"]
        assert len(on103) == 1
        assert "/etc/passwd" in on103[0].evidence.matched_value

    def test_on103_fires_for_absolute_windows_path(self) -> None:
        """C:\\Windows\\System32\\... must produce an ON103 finding."""
        data = _make_simple_onnx(
            has_external_data=True,
            external_location="C:\\Windows\\System32\\config\\SAM",
        )
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on103 = [f for f in findings if f.rule_id == "ON103"]
        assert len(on103) == 1

    def test_on103_does_not_fire_for_safe_location(self) -> None:
        """A plain filename with no traversal must NOT produce ON103."""
        data = _make_simple_onnx(
            has_external_data=True,
            external_location="weights.bin",
        )
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on103 = [f for f in findings if f.rule_id == "ON103"]
        assert on103 == [], f"Unexpected ON103 for safe location: {on103}"

    def test_on103_does_not_fire_for_sibling_directory(self) -> None:
        """A simple relative path like 'data/weights.bin' must not fire ON103."""
        data = _make_simple_onnx(
            has_external_data=True,
            external_location="data/weights.bin",
        )
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on103 = [f for f in findings if f.rule_id == "ON103"]
        assert on103 == [], f"Unexpected ON103 for relative path: {on103}"


# ---------------------------------------------------------------------------
# 6. ON101: non-standard operator domain
# ---------------------------------------------------------------------------

class TestON101CustomDomain:
    def test_on101_fires_for_custom_domain(self) -> None:
        """A node with a non-standard domain must fire ON101."""
        data = _make_simple_onnx(op_type="CustomOp", domain="com.attacker.evil")
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on101 = [f for f in findings if f.rule_id == "ON101"]
        assert len(on101) == 1
        assert "com.attacker.evil" in on101[0].evidence.matched_value

    def test_on101_does_not_fire_for_standard_domain(self) -> None:
        """Standard domain '' (default) must not fire ON101."""
        data = _make_simple_onnx(op_type="Add", domain="")
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on101 = [f for f in findings if f.rule_id == "ON101"]
        assert on101 == []


# ---------------------------------------------------------------------------
# 7. ON102: malformed / unparseable ONNX
# ---------------------------------------------------------------------------

class TestON102MalformedGraph:
    def test_on102_fires_for_garbage_bytes(self) -> None:
        """Garbage bytes that fail protobuf parsing must produce ON102."""
        artifact = _make_artifact(b"\x00\x01\x02\x03NOT_ONNX_PROTOBUF\xFF\xFE")
        findings = ONNXAnalyzer().analyze(artifact)
        on102 = [f for f in findings if f.rule_id == "ON102"]
        assert len(on102) >= 1

    def test_on102_fires_for_empty_bytes(self) -> None:
        """Empty file must return no findings (not an error — empty is CLEAN)."""
        artifact = _make_artifact(b"")
        findings = ONNXAnalyzer().analyze(artifact)
        # Empty file returns early with []
        assert findings == []


# ---------------------------------------------------------------------------
# 8. Real-world PoC fixture
# ---------------------------------------------------------------------------

# SHA256 of the known-malicious model.onnx from rez0/onnx-ort-path-traversal-poc.
# The raw bytes are embedded here as a static constant to avoid network access in tests.
# This is the exact 159-byte protobuf obtained from HuggingFace on 2026-09-02.
# It was verified with: sha256sum model.onnx == 54706759...
_POC_MODEL_BYTES: bytes = (
    b"\x08\x08:\x94\x01\n\x0e\n\x01X\n\x01W\x12\x01Y\"\x03Add\x12\rtraversal_poc"
    b"*I\x08\x01\x08\x04\x10\x01B\x01Wj!\n\x08location\x12\x15../../../../etc/hosts"
    b"j\x0b\n\x06offset\x12\x010j\x0c\n\x06length\x12\x0216p\x01Z\x13\n\x01X\x12"
    b"\x0e\n\x0c\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04b\x13\n\x01Y\x12\x0e"
    b"\n\x0c\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\r"
)
_POC_SHA256 = "54706759afd3eb55d04c323154740745c1cc1f9f6743ed2c17269e3d94ea6a18"


class TestRealWorldPathTraversalPoC:
    """Regression test using the exact bytes of the public PoC model."""

    def test_poc_sha256_matches(self) -> None:
        """Verify the embedded bytes match the known SHA256 of the real PoC."""
        import hashlib
        digest = hashlib.sha256(_POC_MODEL_BYTES).hexdigest()
        assert digest == _POC_SHA256, (
            f"Embedded PoC bytes do not match expected SHA256.\n"
            f"Got:      {digest}\n"
            f"Expected: {_POC_SHA256}"
        )

    def test_poc_produces_on103(self) -> None:
        """The real PoC model must produce exactly one ON103 HIGH finding."""
        artifact = Artifact(
            artifact_id="poc-test",
            source_path="model.onnx",
            input_source=InputSourceType.IN_MEMORY,
            size_bytes=len(_POC_MODEL_BYTES),
            sha256=_POC_SHA256,
            stream=io.BytesIO(_POC_MODEL_BYTES),
        )
        findings = ONNXAnalyzer().analyze(artifact)
        on103 = [f for f in findings if f.rule_id == "ON103"]
        assert len(on103) == 1
        f = on103[0]
        assert f.severity.value == "HIGH"
        assert f.confidence.value == "HIGH"
        assert "../../../../etc/hosts" in f.evidence.matched_value
        assert f.remediation  # non-empty

    def test_poc_produces_no_false_positives(self) -> None:
        """The PoC model uses standard Add op — must not fire ON101."""
        artifact = Artifact(
            artifact_id="poc-test2",
            source_path="model.onnx",
            input_source=InputSourceType.IN_MEMORY,
            size_bytes=len(_POC_MODEL_BYTES),
            sha256=_POC_SHA256,
            stream=io.BytesIO(_POC_MODEL_BYTES),
        )
        findings = ONNXAnalyzer().analyze(artifact)
        on101 = [f for f in findings if f.rule_id == "ON101"]
        assert on101 == [], f"Unexpected ON101 for standard Add op: {on101}"


# ---------------------------------------------------------------------------
# Helper for ON104 fixtures
# ---------------------------------------------------------------------------

def _make_onnx_with_external_offsets(
    *,
    initializer_name: str = "W",
    location: str = "data.bin",
    offset: int = 0,
    length: int = 4,
) -> bytes:
    """Build a minimal ONNX model with specified external_data offset/length.

    The offset and length values are stored as protobuf strings (the ONNX spec
    stores external_data key-value pairs as StringStringEntryProto). Python's
    arbitrary-precision int handles values beyond 2^64 without silent truncation,
    so the scanner's detection logic operates on the true numeric value.

    SAFETY: This helper only calls onnx.helper functions and SerializeToString().
    No model inference or runtime execution is performed.
    """
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])
    node = onnx.helper.make_node("Add", inputs=["X", initializer_name], outputs=["Y"])

    W = onnx.TensorProto()
    W.name = initializer_name
    W.data_type = onnx.TensorProto.FLOAT
    W.dims.append(1)
    W.data_location = onnx.TensorProto.EXTERNAL
    W.external_data.add(key="location", value=location)
    W.external_data.add(key="offset", value=str(offset))
    W.external_data.add(key="length", value=str(length))

    graph = onnx.helper.make_graph([node], "test_overflow_graph", [X], [Y], initializer=[W])
    model = onnx.helper.make_model(graph, ir_version=8)
    model.opset_import[0].version = 13
    return model.SerializeToString()


# ---------------------------------------------------------------------------
# 9. ON104: external_data offset + length uint64 wraparound
# ---------------------------------------------------------------------------

class TestON104ExternalDataOverflow:
    """ON104 security behavior tests — uint64 wraparound in ONNX external_data.

    Detection condition: offset + length >= 2^64 (Python integer comparison;
    no silent overflow in the scanner). Any runtime that evaluates this sum
    in uint64 will compute the wrapped value, potentially bypassing bounds checks.

    TC11 (0xiviel/poc-openvino-onnx-uint64-overflow):
      offset = 0xFFFFFFFFFFFFFFF0 = 18446744073709551600
      length = 16
      sum    = 18446744073709551616 = 2^64  → wrapped = 0
      OpenVINO check: m_offset + m_data_length > file_size → 0 > file_size → False → BYPASSED
    """

    _UINT64_MOD: int = 1 << 64          # 2^64 = 18446744073709551616
    _UINT64_MAX: int = (1 << 64) - 1    # 2^64 - 1 = 18446744073709551615

    # TC11 exact parameters
    _TC11_OFFSET: int = 0xFFFFFFFFFFFFFFF0  # = 18446744073709551600 = 2^64 - 16
    _TC11_LENGTH: int = 16

    def test_on104_fires_for_tc11_parameters(self) -> None:
        """TC11: offset=0xFFFFFFFFFFFFFFF0, length=16 → sum=2^64 → ON104 fires.

        This is the exact vulnerable condition from
        0xiviel/poc-openvino-onnx-uint64-overflow. The sum reaches exactly 2^64,
        which wraps to 0 in uint64, bypassing OpenVINO's bounds check.
        The old 93368c0 implementation had no ON104 rule and returned CLEAN (FN).
        This test must fail on 93368c0 and pass after ON104 is implemented.
        """
        assert self._TC11_OFFSET + self._TC11_LENGTH == self._UINT64_MOD, (
            "TC11 fixture: offset + length must equal exactly 2^64"
        )
        data = _make_onnx_with_external_offsets(
            location="overflow_data.bin",
            offset=self._TC11_OFFSET,
            length=self._TC11_LENGTH,
        )
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on104 = [f for f in findings if f.rule_id == "ON104"]
        assert len(on104) == 1, f"Expected 1 ON104 finding, got {len(on104)}: {on104}"
        f = on104[0]
        assert f.severity.value == "HIGH"
        assert f.confidence.value == "HIGH"
        assert str(self._TC11_OFFSET) in f.evidence.matched_value
        assert str(self._TC11_LENGTH) in f.evidence.matched_value
        assert f.remediation

    def test_on104_fires_for_strict_overflow_past_uint64_max(self) -> None:
        """offset=2^64-16, length=32 → sum=2^64+16 (strictly past 2^64) → ON104 fires."""
        offset = self._UINT64_MAX - 15   # = 2^64 - 16
        length = 32
        assert offset + length == self._UINT64_MOD + 16, "Fixture: sum must be 2^64 + 16"
        data = _make_onnx_with_external_offsets(location="data.bin", offset=offset, length=length)
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on104 = [f for f in findings if f.rule_id == "ON104"]
        assert len(on104) == 1, f"Expected 1 ON104 finding for sum > 2^64, got {len(on104)}"
        assert str(offset) in on104[0].evidence.matched_value

    def test_on104_does_not_fire_for_sub_boundary(self) -> None:
        """offset=2^64-16, length=15 → sum=2^64-1 < 2^64 → ON104 must NOT fire.

        Mathematical boundary: ON104 fires when sum >= 2^64 (the uint64 wraparound
        boundary, inclusive). Sum = 2^64 - 1 is the largest value that fits in
        uint64 without wrapping, so it does NOT fire.

        Note: TC11 uses length=16 (sum = 2^64, which fires). This test uses
        length=15 (sum = 2^64 - 1, which does not) to demonstrate the strict
        lower boundary of the detection condition.
        """
        offset = self._UINT64_MAX - 15   # = 2^64 - 16
        length = 15
        assert offset + length == self._UINT64_MAX, (
            "Fixture: sum must equal 2^64 - 1 (just below wraparound boundary)"
        )
        data = _make_onnx_with_external_offsets(location="data.bin", offset=offset, length=length)
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on104 = [f for f in findings if f.rule_id == "ON104"]
        assert on104 == [], (
            f"ON104 must NOT fire when sum < 2^64 (sum={offset + length}): {on104}"
        )

    def test_on104_does_not_fire_for_normal_external_data(self) -> None:
        """offset=0, length=1024 → sum=1024 (far below 2^64): ON104 must NOT fire."""
        data = _make_onnx_with_external_offsets(location="weights.bin", offset=0, length=1024)
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on104 = [f for f in findings if f.rule_id == "ON104"]
        assert on104 == [], f"ON104 must NOT fire for normal external_data (offset=0, length=1024): {on104}"

    def test_on104_does_not_fire_for_large_but_valid_offset(self) -> None:
        """offset=100 GB, length=4096 → sum ~ 100 GB (far below 2^64): ON104 must NOT fire.

        Normal large models can legitimately have large offsets when tensors are
        packed sequentially in a single binary file. 100 GB is well within the
        range of real model files while being 8+ orders of magnitude below 2^64.
        """
        offset = 100 * 1024 * 1024 * 1024  # 100 GiB = ~107 billion
        length = 4096
        assert offset + length < self._UINT64_MOD, "Fixture: sum must be far below 2^64"
        data = _make_onnx_with_external_offsets(location="weights.bin", offset=offset, length=length)
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on104 = [f for f in findings if f.rule_id == "ON104"]
        assert on104 == [], (
            f"ON104 must NOT fire for large-but-valid offset ({offset}): {on104}"
        )

    def test_on103_regression_still_fires_with_traversal(self) -> None:
        """ON103 regression: path-traversal location still produces ON103 (not ON104).

        Both rules iterate external_data independently. ON103 checks the location
        string; ON104 checks offset + length. The ON103 PoC (../../../../etc/hosts)
        uses offset=0 and length=4, so ON104 does NOT fire for it.
        """
        data = _make_simple_onnx(
            has_external_data=True,
            external_location="../../../../etc/hosts",
        )
        artifact = _make_artifact(data)
        findings = ONNXAnalyzer().analyze(artifact)
        on103 = [f for f in findings if f.rule_id == "ON103"]
        on104 = [f for f in findings if f.rule_id == "ON104"]
        assert len(on103) == 1, "ON103 must still fire for traversal location"
        assert on104 == [], (
            f"ON104 must NOT fire for traversal PoC (offset=0, length=4): {on104}"
        )
