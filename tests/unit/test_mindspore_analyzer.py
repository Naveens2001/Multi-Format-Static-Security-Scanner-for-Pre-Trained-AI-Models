"""
tests.unit.test_mindspore_analyzer
===================================

Adversarial unit tests for the MindSpore checkpoint format detector and
analyzer (MS101 — CWE-789 Uncontrolled Memory Allocation).

Security constraints enforced by these tests:
- The scanner must NOT allocate memory proportional to declared tensor dims.
- Malformed or attacker-controlled bytes must not crash the analyzer.
- No hard-coded repo, filename, or specific shape — all patterns are general.
"""

from __future__ import annotations

import io
import sys
import uuid

import pytest

from model_scanner.analyzers.mindspore.analyzer import (
    MindSporeAnalyzer,
    _1_GB,
    _10_GB,
    _100_GB,
)
from model_scanner.analyzers.mindspore.proto_reader import (
    MalformedProtoError,
    TensorMeta,
    _read_varint,
    parse_checkpoint_metadata,
)
from model_scanner.core.models import (
    Artifact,
    InputSourceType,
    ModelFormat,
    Severity,
)
from model_scanner.detection.detector import FormatDispatcher
from model_scanner.detection.magic import is_mindspore_ckpt_header


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(data: bytes, name: str = "test.ckpt") -> Artifact:
    import hashlib
    sha256 = hashlib.sha256(data).hexdigest()
    return Artifact(
        artifact_id=str(uuid.uuid4()),
        source_path=name,
        input_source=InputSourceType.LOCAL_FILE,
        size_bytes=len(data),
        sha256=sha256,
        stream=io.BytesIO(data),
    )


def _encode_varint(value: int) -> bytes:
    """Encode a non-negative integer as a protobuf LEB128 varint."""
    result = []
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            result.append(byte | 0x80)
        else:
            result.append(byte)
            break
    return bytes(result)


def _build_mindspore_ckpt(
    name: str,
    dims: list[int],
    data_type: str = "Float32",
    tensor_data: bytes = b"\x00",
) -> bytes:
    """Build a minimal MindSpore protobuf checkpoint in memory.

    Does NOT allocate an array proportional to ``dims``. The ``tensor_data``
    argument is written verbatim as the tensor_content field.
    Wire format: Checkpoint(field1=Parameter(field1=name, field2=TensorProto))
    """
    # TensorProto: field 1=dims (VARINT), field 2=data_type (LEN), field 3=tensor_content (LEN)
    tensor_proto = b""
    for dim in dims:
        tensor_proto += b"\x08" + _encode_varint(dim)
    dt_bytes = data_type.encode("utf-8")
    tensor_proto += b"\x12" + _encode_varint(len(dt_bytes)) + dt_bytes
    tensor_proto += b"\x1a" + _encode_varint(len(tensor_data)) + tensor_data

    # Parameter: field 1=name (LEN), field 2=tensor (LEN)
    name_bytes = name.encode("utf-8")
    parameter = b"\x0a" + _encode_varint(len(name_bytes)) + name_bytes
    parameter += b"\x12" + _encode_varint(len(tensor_proto)) + tensor_proto

    # Checkpoint: field 1=parameter (LEN)
    return b"\x0a" + _encode_varint(len(parameter)) + parameter


def _build_two_tensor_ckpt(
    name1: str, dims1: list[int], data1: bytes,
    name2: str, dims2: list[int], data2: bytes,
    dtype: str = "Float32",
) -> bytes:
    """Build a two-parameter checkpoint."""
    def _mk_param(name: str, dims: list[int], tensor_data: bytes) -> bytes:
        tensor_proto = b""
        for d in dims:
            tensor_proto += b"\x08" + _encode_varint(d)
        dt_bytes = dtype.encode()
        tensor_proto += b"\x12" + _encode_varint(len(dt_bytes)) + dt_bytes
        tensor_proto += b"\x1a" + _encode_varint(len(tensor_data)) + tensor_data
        name_bytes = name.encode()
        param = b"\x0a" + _encode_varint(len(name_bytes)) + name_bytes
        param += b"\x12" + _encode_varint(len(tensor_proto)) + tensor_proto
        return b"\x0a" + _encode_varint(len(param)) + param

    return _mk_param(name1, dims1, data1) + _mk_param(name2, dims2, data2)


# ---------------------------------------------------------------------------
# Test 1: Format detection — MindSpore protobuf header → MINDSPORE_CKPT
# ---------------------------------------------------------------------------


def test_format_detection_mindspore_magic() -> None:
    """MindSpore checkpoint bytes are correctly detected as MINDSPORE_CKPT."""
    data = _build_mindspore_ckpt("layer.weight", [10, 20], "Float32", b"\x00" * 800)
    dispatcher = FormatDispatcher()
    result = dispatcher.detect_from_bytes(
        header=data[:64],
        filename_hint="model.ckpt",
        total_size=len(data),
    )
    assert result.detected_format == ModelFormat.MINDSPORE_CKPT
    assert result.magic_matched is True
    assert result.format_mismatch is False


def test_format_detection_bomb_bytes_are_mindspore() -> None:
    """Large-dims MindSpore checkpoint (400 GB shape) is detected as MINDSPORE_CKPT."""
    data = _build_mindspore_ckpt("w", [10_000, 10_000, 1_000], "Float32", b"\x00")
    dispatcher = FormatDispatcher()
    result = dispatcher.detect_from_bytes(
        header=data[:64],
        filename_hint="dos_minimal.ckpt",
        total_size=len(data),
    )
    assert result.detected_format == ModelFormat.MINDSPORE_CKPT
    assert result.magic_matched is True


# ---------------------------------------------------------------------------
# Test 2: Regression — PyTorch ZIP .ckpt still detected correctly
# ---------------------------------------------------------------------------


def test_format_detection_pytorch_zip_ckpt_regression() -> None:
    """A genuine PyTorch ZIP .ckpt (starts with PK magic) is still PYTORCH_ZIP."""
    pytorch_zip_magic = b"PK\x03\x04" + b"\x00" * 60
    dispatcher = FormatDispatcher()
    result = dispatcher.detect_from_bytes(
        header=pytorch_zip_magic,
        filename_hint="model.ckpt",
        total_size=1024,
    )
    assert result.detected_format == ModelFormat.PYTORCH_ZIP


# ---------------------------------------------------------------------------
# Test 3: Regression — legacy PyTorch pickle .ckpt still detected correctly
# ---------------------------------------------------------------------------


def test_format_detection_legacy_pytorch_ckpt_regression() -> None:
    """A pickle-based .ckpt (starts with \\x80\\x02) is still LEGACY_PYTORCH."""
    pickle_magic = b"\x80\x02" + b"\x00" * 62
    dispatcher = FormatDispatcher()
    result = dispatcher.detect_from_bytes(
        header=pickle_magic,
        filename_hint="model.ckpt",
        total_size=1024,
    )
    assert result.detected_format == ModelFormat.LEGACY_PYTORCH


# ---------------------------------------------------------------------------
# Test 4: Random bytes with .ckpt extension → PYTORCH_ZIP (extension fallback)
# ---------------------------------------------------------------------------


def test_format_detection_random_bytes_ckpt_fallback() -> None:
    """Random bytes that do not match any magic fall back to PYTORCH_ZIP via extension."""
    random_bytes = bytes([0xFF, 0xFE, 0x00, 0x01] + [0x42] * 60)
    dispatcher = FormatDispatcher()
    result = dispatcher.detect_from_bytes(
        header=random_bytes,
        filename_hint="unknown.ckpt",
        total_size=len(random_bytes),
    )
    assert result.detected_format == ModelFormat.PYTORCH_ZIP
    assert result.magic_matched is False


# ---------------------------------------------------------------------------
# Tests 5–7: MS101 fires for allocation bombs
# ---------------------------------------------------------------------------


def test_ms101_fires_for_400gb_bomb() -> None:
    """10000×10000×1000 Float32 = 400 GB declared → MS101 HIGH."""
    data = _build_mindspore_ckpt("bomb.weight", [10_000, 10_000, 1_000], "Float32", b"\x00")
    artifact = _make_artifact(data, "dos_minimal.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == "MS101"
    assert f.severity == Severity.HIGH
    assert "bomb.weight" in f.description
    assert "CWE-789" in f.cwe_ids


def test_ms101_fires_for_40gb_bomb() -> None:
    """100000×100000×1 Float32 = 40 GB declared → MS101 HIGH (>=10 GB, sparse)."""
    data = _build_mindspore_ckpt("bomb.weight", [100_000, 100_000, 1], "Float32", b"\x00")
    artifact = _make_artifact(data, "dos_moderate.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    assert len(findings) == 1
    assert findings[0].rule_id == "MS101"
    assert findings[0].severity == Severity.HIGH


def test_ms101_fires_for_4tb_bomb() -> None:
    """100000×100000×100 Float32 = 4 TB declared → MS101 HIGH (>=100 GB threshold)."""
    data = _build_mindspore_ckpt("bomb.weight", [100_000, 100_000, 100], "Float32", b"\x00")
    artifact = _make_artifact(data, "dos_extreme.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    assert len(findings) == 1
    assert findings[0].rule_id == "MS101"
    assert findings[0].severity == Severity.HIGH


# ---------------------------------------------------------------------------
# Test 8: No MS101 for small tensor (below 1 GB threshold)
# ---------------------------------------------------------------------------


def test_no_ms101_small_tensor() -> None:
    """100×100 Float32 = 40 KB — well below 1 GB threshold → no finding."""
    data = _build_mindspore_ckpt("layer.weight", [100, 100], "Float32", b"\x00" * 40_000)
    artifact = _make_artifact(data, "small_model.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    assert not any(f.rule_id == "MS101" for f in findings)


# ---------------------------------------------------------------------------
# Test 9: No MS101 for tensor whose declared size is below 1 GB (even with no data)
# ---------------------------------------------------------------------------


def test_no_ms101_tensor_below_1gb() -> None:
    """1000×1000×100 Float32 = 400 MB < 1 GB → no finding even with zero data."""
    data = _build_mindspore_ckpt("w", [1_000, 1_000, 100], "Float32", b"")
    artifact = _make_artifact(data, "model.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    assert not any(f.rule_id == "MS101" for f in findings)


# ---------------------------------------------------------------------------
# Test 10: Truncated checkpoint — no crash
# ---------------------------------------------------------------------------


def test_truncated_checkpoint_no_crash() -> None:
    """Truncated checkpoint bytes must not raise an exception."""
    full = _build_mindspore_ckpt("w", [10_000, 10_000, 1_000], "Float32", b"\x00")
    artifact = _make_artifact(full[:10], "truncated.ckpt")
    _ = list(MindSporeAnalyzer().analyze(artifact))  # must not raise


def test_truncated_to_3_bytes_no_crash() -> None:
    """3-byte truncation must not crash."""
    full = _build_mindspore_ckpt("w", [10_000, 10_000, 1_000], "Float32", b"\x00")
    artifact = _make_artifact(full[:3], "truncated.ckpt")
    _ = list(MindSporeAnalyzer().analyze(artifact))


# ---------------------------------------------------------------------------
# Test 11: Invalid/random bytes — no crash, no MS101
# ---------------------------------------------------------------------------


def test_random_bytes_no_crash() -> None:
    """Completely random bytes must not crash the analyzer."""
    data = bytes(range(256)) * 2  # 512 bytes of structured garbage
    artifact = _make_artifact(data, "garbage.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    assert not any(f.rule_id == "MS101" for f in findings)


# ---------------------------------------------------------------------------
# Test 12: Empty file — no crash, no findings
# ---------------------------------------------------------------------------


def test_empty_file_no_findings() -> None:
    """Empty file produces no findings and does not raise."""
    artifact = _make_artifact(b"", "empty.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    assert findings == []


# ---------------------------------------------------------------------------
# Test 13: MS101 fires correctly for Float64 dtype (8 bytes/element)
# ---------------------------------------------------------------------------


def test_ms101_fires_float64() -> None:
    """50000×50000 Float64 = 20 GB → MS101 HIGH (element size correctly 8 bytes)."""
    data = _build_mindspore_ckpt("w", [50_000, 50_000], "Float64", b"\x00")
    artifact = _make_artifact(data, "bomb_float64.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    ms101 = [f for f in findings if f.rule_id == "MS101"]
    assert len(ms101) == 1
    assert ms101[0].severity == Severity.HIGH


# ---------------------------------------------------------------------------
# Test 14: Multiple tensors — only the bomb tensor generates MS101
# ---------------------------------------------------------------------------


def test_ms101_fires_only_for_bomb_tensor() -> None:
    """Two tensors: one legitimate (small), one bomb → exactly one MS101 finding."""
    data = _build_two_tensor_ckpt(
        "safe_layer", [10, 10], b"\x00" * 400,
        "bomb_layer", [100_000, 100_000], b"\x00",
    )
    artifact = _make_artifact(data, "mixed.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    ms101 = [f for f in findings if f.rule_id == "MS101"]
    assert len(ms101) == 1
    assert "bomb_layer" in ms101[0].description


def test_ms101_fires_for_each_bomb_tensor() -> None:
    """Two bomb tensors → two MS101 findings (one per tensor)."""
    data = _build_two_tensor_ckpt(
        "bomb_a", [100_000, 100_000], b"\x00",
        "bomb_b", [100_000, 100_000], b"\x00",
    )
    artifact = _make_artifact(data, "two_bombs.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    ms101 = [f for f in findings if f.rule_id == "MS101"]
    assert len(ms101) == 2


# ---------------------------------------------------------------------------
# Tests 15–18: _read_varint correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected_bytes", [
    (0, b"\x00"),
    (1, b"\x01"),
    (127, b"\x7f"),
    (128, b"\x80\x01"),
    (300, b"\xac\x02"),
    (10_000, b"\x90\x4e"),
    (100_000, b"\xa0\x8d\x06"),
    (1_000_000, b"\xc0\x84\x3d"),
])
def test_read_varint_roundtrip(value: int, expected_bytes: bytes) -> None:
    """_read_varint correctly decodes varints produced by _encode_varint."""
    encoded = _encode_varint(value)
    assert encoded == expected_bytes, (
        f"Expected {expected_bytes.hex()} for {value}, got {encoded.hex()}"
    )
    decoded, new_offset = _read_varint(encoded, 0)
    assert decoded == value
    assert new_offset == len(encoded)


def test_read_varint_truncated_raises() -> None:
    """_read_varint raises MalformedProtoError when continuation bit is set but data ends."""
    with pytest.raises(MalformedProtoError, match="Truncated varint"):
        _read_varint(b"\x80", 0)


def test_read_varint_11_bytes_raises() -> None:
    """An 11-byte varint (beyond protobuf spec) raises MalformedProtoError."""
    bad_varint = bytes([0x80] * 11)
    with pytest.raises(MalformedProtoError, match="Varint too long"):
        _read_varint(bad_varint, 0)


def test_read_varint_at_offset() -> None:
    """_read_varint reads from the specified offset, ignoring preceding bytes."""
    data = b"\xDE\xAD" + _encode_varint(300)  # 2 noise bytes + b"\xac\x02"
    value, offset = _read_varint(data, 2)
    assert value == 300
    assert offset == 4  # 2 noise + 2 varint bytes


# ---------------------------------------------------------------------------
# Test 19: Astronomically large dim varint — no crash, no allocation
# ---------------------------------------------------------------------------


def test_large_dim_varint_no_crash() -> None:
    """A dim encoded as a very large uint64 is handled safely without allocation."""
    # (2^32 - 1) Float32 = ~17.2 GB → MS101 should fire
    huge_dim = (1 << 32) - 1
    data = _build_mindspore_ckpt("huge", [huge_dim], "Float32", b"\x00")
    metas = parse_checkpoint_metadata(data)
    assert len(metas) == 1
    _ = metas[0].declared_allocation_bytes  # Python int arithmetic — no array allocation

    artifact = _make_artifact(data, "huge_dim.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    ms101 = [f for f in findings if f.rule_id == "MS101"]
    assert len(ms101) == 1  # 17.2 GB >= 10 GB threshold


# ---------------------------------------------------------------------------
# Test 20 (SECURITY): Scanner does NOT allocate declared tensor memory
# ---------------------------------------------------------------------------


def test_scanner_does_not_allocate_tensor_memory() -> None:
    """SECURITY: analyzing a 4 TB allocation bomb must not allocate significant memory."""
    data = _build_mindspore_ckpt("bomb.weight", [100_000, 100_000, 100], "Float32", b"\x00")

    if sys.platform == "win32":
        # On Windows verify the function completes and fires MS101 (no OOM)
        artifact = _make_artifact(data, "dos_extreme.ckpt")
        findings = list(MindSporeAnalyzer().analyze(artifact))
        assert len(findings) == 1
        assert findings[0].rule_id == "MS101"
        return

    import resource  # Unix-only

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    artifact = _make_artifact(data, "dos_extreme.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    scale = 1024 if sys.platform == "linux" else 1  # KB on Linux, bytes on macOS
    increase_bytes = (after - before) * scale
    assert increase_bytes < 100 * 1024 * 1024, (
        f"Memory increase of {increase_bytes / 1e6:.1f} MB suggests tensor allocation"
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "MS101"


# ---------------------------------------------------------------------------
# Test 21: No G001 for MINDSPORE_CKPT in .ckpt file (format_mismatch=False)
# ---------------------------------------------------------------------------


def test_no_g001_for_mindspore_ckpt() -> None:
    """Detecting MINDSPORE_CKPT in a .ckpt file must not set format_mismatch=True."""
    data = _build_mindspore_ckpt("w", [10_000, 10_000, 1_000], "Float32", b"\x00")
    dispatcher = FormatDispatcher()
    result = dispatcher.detect_from_bytes(
        header=data[:64],
        filename_hint="dos_minimal.ckpt",
        total_size=len(data),
    )
    assert result.detected_format == ModelFormat.MINDSPORE_CKPT
    assert result.format_mismatch is False


# ---------------------------------------------------------------------------
# Test 22: Empty protobuf checkpoint (no parameters)
# ---------------------------------------------------------------------------


def test_empty_checkpoint_metadata_returns_empty_list() -> None:
    """Empty bytes → parse_checkpoint_metadata returns []."""
    assert parse_checkpoint_metadata(b"") == []


def test_empty_file_produces_no_findings() -> None:
    """Analyzer on an empty stream returns no findings."""
    artifact = _make_artifact(b"", "empty.ckpt")
    assert list(MindSporeAnalyzer().analyze(artifact)) == []


# ---------------------------------------------------------------------------
# Tests 23–26: TensorMeta arithmetic
# ---------------------------------------------------------------------------


def test_tensor_meta_4tb_arithmetic() -> None:
    """declared_allocation_bytes uses Python int — no overflow for 4 TB."""
    meta = TensorMeta(
        name="bomb",
        dims=[100_000, 100_000, 100],
        data_type="Float32",
        actual_data_bytes=1,
        tensor_field_offset=0,
        tensor_field_length=10,
    )
    assert meta.declared_allocation_bytes == 100_000 * 100_000 * 100 * 4  # 4 TB
    assert meta.declared_element_count == 1_000_000_000_000


def test_tensor_meta_negative_dim_returns_zero() -> None:
    """A negative dimension makes declared_element_count == 0 (nonsensical shape)."""
    meta = TensorMeta(
        name="w", dims=[-1, 100], data_type="Float32",
        actual_data_bytes=0, tensor_field_offset=0, tensor_field_length=10,
    )
    assert meta.declared_element_count == 0
    assert meta.declared_allocation_bytes == 0


def test_tensor_meta_zero_dim_returns_zero() -> None:
    """A zero dimension makes declared_element_count == 0."""
    meta = TensorMeta(
        name="w", dims=[0, 100], data_type="Float32",
        actual_data_bytes=0, tensor_field_offset=0, tensor_field_length=10,
    )
    assert meta.declared_element_count == 0
    assert meta.declared_allocation_bytes == 0


def test_tensor_meta_unknown_dtype_defaults_4_bytes() -> None:
    """Unknown dtype defaults to 4 bytes per element."""
    meta = TensorMeta(
        name="w", dims=[10], data_type="SomeFutureType",
        actual_data_bytes=0, tensor_field_offset=0, tensor_field_length=10,
    )
    assert meta.element_bytes == 4
    assert meta.declared_allocation_bytes == 40


# ---------------------------------------------------------------------------
# Test 27: MS101 evidence contains human-readable allocation summary
# ---------------------------------------------------------------------------


def test_ms101_evidence_contains_summary() -> None:
    """MS101 finding evidence includes declared allocation and actual bytes."""
    data = _build_mindspore_ckpt("evidence.tensor", [100_000, 100_000, 100], "Float32", b"\x00")
    artifact = _make_artifact(data, "dos_extreme.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    assert len(findings) == 1
    f = findings[0]
    assert f.evidence.context_snippet is not None
    assert "Declared allocation" in f.evidence.context_snippet
    assert "Actual data in file" in f.evidence.context_snippet
    assert "evidence.tensor" in f.evidence.matched_value
    assert "dims" in f.evidence.matched_value


# ---------------------------------------------------------------------------
# Tests 28–29: is_mindspore_ckpt_header
# ---------------------------------------------------------------------------


def test_magic_rejects_zip() -> None:
    assert not is_mindspore_ckpt_header(b"PK\x03\x04" + b"\x00" * 60)


def test_magic_rejects_gguf() -> None:
    assert not is_mindspore_ckpt_header(b"GGUF" + b"\x00" * 60)


def test_magic_rejects_pickle() -> None:
    assert not is_mindspore_ckpt_header(b"\x80\x04" + b"\x00" * 60)


def test_magic_rejects_short_input() -> None:
    assert not is_mindspore_ckpt_header(b"\x0a")


def test_magic_rejects_wrong_first_byte() -> None:
    assert not is_mindspore_ckpt_header(b"\x0b\x06\x0a\x01w\x12\x00")


def test_magic_accepts_builder_output() -> None:
    """A header built by _build_mindspore_ckpt is accepted."""
    data = _build_mindspore_ckpt("layer", [10, 10], "Float32", b"\x00" * 400)
    assert is_mindspore_ckpt_header(data[:64])


def test_magic_accepts_minimal_valid_header() -> None:
    """Hand-crafted minimal header: 0A <param_len> 0A <name_len> <name> 12 ..."""
    name = b"w"
    tensor_proto = b"\x12\x00"  # minimal: field 2 empty string
    param = b"\x0a" + _encode_varint(len(name)) + name
    param += b"\x12" + _encode_varint(len(tensor_proto)) + tensor_proto
    header = b"\x0a" + _encode_varint(len(param)) + param
    assert is_mindspore_ckpt_header(header)


# ---------------------------------------------------------------------------
# Bonus: MEDIUM severity for 1–10 GB sparse tensor
# ---------------------------------------------------------------------------


def test_ms101_medium_severity_2gb_sparse() -> None:
    """2 GB declared with 1 byte actual data → MS101 MEDIUM (1 GB <= declared < 10 GB)."""
    # 536_870_912 Float32 elements = 2 GB exactly
    data = _build_mindspore_ckpt("sparse_weight", [536_870_912], "Float32", b"\x00")
    artifact = _make_artifact(data, "sparse.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    ms101 = [f for f in findings if f.rule_id == "MS101"]
    assert len(ms101) == 1
    assert ms101[0].severity == Severity.MEDIUM


def test_ms101_100gb_fires_regardless_of_sparsity() -> None:
    """>=100 GB fires HIGH even when actual_data_bytes is larger than 1e-6 of declared."""
    # 100_000 * 100_000 * 27 * 4 = 108 GB >= 100 GB threshold
    # actual_data = 50_000 bytes → sparsity >> 1e-6, but >=100 GB path fires anyway
    data = _build_mindspore_ckpt(
        "large_weight", [100_000, 100_000, 27], "Float32", b"\x00" * 50_000
    )
    artifact = _make_artifact(data, "large.ckpt")
    findings = list(MindSporeAnalyzer().analyze(artifact))
    ms101 = [f for f in findings if f.rule_id == "MS101"]
    assert len(ms101) == 1
    assert ms101[0].severity == Severity.HIGH
