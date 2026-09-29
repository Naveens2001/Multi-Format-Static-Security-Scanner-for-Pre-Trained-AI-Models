"""
tests/unit/test_gguf_analyzer.py
=================================

Comprehensive regression and unit tests for the Phase 6 GGUF static security
analysis implementation.

SAFETY CONTRACT
---------------
- No tensor weights or model payloads are deserialized or loaded into memory.
- No llama.cpp, transformers, torch, or external GGUF runtime libraries used.
- All fixtures are synthetic binary byte sequences constructed using struct.pack().
- No eval(), exec(), pickle.loads(), or torch.load() called.

Coverage
--------
1. Valid GGUF v1, v2, v3 (no findings)
2. Invalid magic bytes (GU202)
3. Unsupported GGUF version (GU202)
4. Truncated header, counts, KV entries, tensor descriptors (GU203)
5. Resource exhaustion / KV overflow (GU201)
6. Resource exhaustion / Oversized staging buffer (GU104)
7. Invalid metadata value type (GU204)
8. Invalid tensor name / empty name / duplicate tensor name (GU205)
9. Invalid tensor dimensions / negative dimensions / overflow (GU206)
10. Invalid GGML tensor type (GU207)
11. Invalid tensor offset / invalid alignment (GU208)
12. Tensor range outside file bounds (GU209)
13. Overlapping tensor data ranges (GU210)
14. Metadata code injection (GU101)
15. Path traversal sequence in metadata (GU102)
16. Credential exposure in metadata (GU103)
17. Suspicious URL in metadata (GU105)
18. Script injection tag in metadata (GU106)
19. Multiple valid metadata types and multiple tensors
20. Evidence byte offsets and coordinate space
21. Severity, confidence, and non-empty remediation
22. Safety assertions: no forbidden imports or dynamic execution
"""

from __future__ import annotations

import io
import struct
from typing import Any

import pytest

from model_scanner.analyzers.gguf.analyzer import GGUFAnalyzer
from model_scanner.analyzers.gguf.metadata import scan_gguf_metadata
from model_scanner.analyzers.gguf.reader import (
    GGUF_MAGIC,
    GGUFReader,
    GGUFValueType,
)
from model_scanner.analyzers.gguf.tensors import (
    GGMLType,
    compute_tensor_byte_size,
)
from model_scanner.core.models import (
    Artifact,
    Confidence,
    CoordinateSpace,
    InputSourceType,
    ModelFormat,
    ScanConfig,
    Severity,
)

# ---------------------------------------------------------------------------
# Synthetic GGUF Binary Builder
# ---------------------------------------------------------------------------


def _pack_str(s: str, version: int = 3) -> bytes:
    """Pack a length-prefixed string for GGUF."""
    raw = s.encode("utf-8")
    if version == 1:
        return struct.pack("<I", len(raw)) + raw
    return struct.pack("<Q", len(raw)) + raw


def _pack_val(val_type: int, val: Any, version: int = 3) -> bytes:
    """Pack a GGUF metadata value according to its GGUFValueType."""
    if val_type == GGUFValueType.UINT8:
        return struct.pack("<B", int(val))
    if val_type == GGUFValueType.INT8:
        return struct.pack("<b", int(val))
    if val_type == GGUFValueType.UINT16:
        return struct.pack("<H", int(val))
    if val_type == GGUFValueType.INT16:
        return struct.pack("<h", int(val))
    if val_type == GGUFValueType.UINT32:
        return struct.pack("<I", int(val))
    if val_type == GGUFValueType.INT32:
        return struct.pack("<i", int(val))
    if val_type == GGUFValueType.FLOAT32:
        return struct.pack("<f", float(val))
    if val_type == GGUFValueType.BOOL:
        return struct.pack("<B", 1 if val else 0)
    if val_type == GGUFValueType.STRING:
        return _pack_str(str(val), version)
    if val_type == GGUFValueType.UINT64:
        return struct.pack("<Q", int(val))
    if val_type == GGUFValueType.INT64:
        return struct.pack("<q", int(val))
    if val_type == GGUFValueType.FLOAT64:
        return struct.pack("<d", float(val))
    if val_type == GGUFValueType.ARRAY:
        # val is tuple: (elem_type, [elements])
        elem_type, elems = val
        count_fmt = "<I" if version == 1 else "<Q"
        buf = struct.pack("<I", elem_type) + struct.pack(count_fmt, len(elems))
        for el in elems:
            buf += _pack_val(elem_type, el, version)
        return buf
    raise ValueError(f"Unknown value type: {val_type}")


def _build_gguf(
    version: int = 3,
    kv_pairs: list[tuple[str, int, Any]] | None = None,
    tensors: list[dict[str, Any]] | None = None,
    alignment: int = 32,
    data_section_bytes: bytes = b"",
) -> bytes:
    """Build a synthetic valid or custom GGUF binary blob."""
    kv_list = kv_pairs or []
    t_list = tensors or []

    # 1. Header
    magic = GGUF_MAGIC
    ver = struct.pack("<I", version)
    count_fmt = "<II" if version == 1 else "<QQ"
    counts = struct.pack(count_fmt, len(t_list), len(kv_list))
    header_buf = magic + ver + counts

    # 2. KV pairs
    for key, vtype, val in kv_list:
        header_buf += _pack_str(key, version)
        header_buf += struct.pack("<I", vtype)
        header_buf += _pack_val(vtype, val, version)

    # 3. Tensors
    dim_fmt = "<I" if version == 1 else "<Q"
    for t in t_list:
        name = t["name"]
        dims = t["dimensions"]
        ttype = t["type"]
        offset = t["offset"]
        header_buf += _pack_str(name, version)
        header_buf += struct.pack("<I", len(dims))
        for d in dims:
            header_buf += struct.pack(dim_fmt, d)
        header_buf += struct.pack("<I", ttype)
        header_buf += struct.pack("<Q", offset)

    # 4. Alignment padding
    pad_len = (alignment - (len(header_buf) % alignment)) % alignment
    header_buf += bytes(pad_len)

    # 5. Append data section
    return header_buf + data_section_bytes


def _make_artifact(data: bytes, sha_char: str = "0") -> Artifact:
    """Wrap bytes in a standard Artifact."""
    sha = sha_char * 64
    return Artifact(
        artifact_id="test_artifact",
        source_path="<test_gguf>",
        input_source=InputSourceType.IN_MEMORY,
        size_bytes=len(data),
        sha256=sha,
        stream=io.BytesIO(data),
    )


def _rule_ids(findings: Any) -> list[str]:
    return [f.rule_id for f in findings]


# ---------------------------------------------------------------------------
# Test Valid GGUF Files
# ---------------------------------------------------------------------------


class TestValidGGUF:
    """Valid GGUF files must produce zero security findings."""

    def test_valid_gguf_v3_clean(self) -> None:
        data = _build_gguf(
            version=3,
            kv_pairs=[
                ("general.architecture", GGUFValueType.STRING, "llama"),
                ("general.alignment", GGUFValueType.UINT32, 32),
                ("llama.context_length", GGUFValueType.UINT32, 4096),
                ("llama.embedding_length", GGUFValueType.UINT32, 4096),
            ],
            tensors=[
                {
                    "name": "token_embd.weight",
                    "dimensions": [4096, 32000],
                    "type": GGMLType.F16,
                    "offset": 0,
                },
                {
                    "name": "output_norm.weight",
                    "dimensions": [4096],
                    "type": GGMLType.F32,
                    "offset": 4096 * 32000 * 2,
                },
            ],
            data_section_bytes=bytes(4096 * 32000 * 2 + 4096 * 4),
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data))
        assert _rule_ids(findings) == [], f"Expected no findings, got {_rule_ids(findings)}"

    def test_valid_gguf_v2_clean(self) -> None:
        data = _build_gguf(
            version=2,
            kv_pairs=[("general.architecture", GGUFValueType.STRING, "gpt2")],
            tensors=[
                {
                    "name": "weight",
                    "dimensions": [128, 128],
                    "type": GGMLType.F32,
                    "offset": 0,
                }
            ],
            data_section_bytes=bytes(128 * 128 * 4),
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "1"))
        assert _rule_ids(findings) == []

    def test_valid_gguf_v1_clean(self) -> None:
        data = _build_gguf(
            version=1,
            kv_pairs=[("model_name", GGUFValueType.STRING, "test_v1")],
            tensors=[
                {
                    "name": "w",
                    "dimensions": [64],
                    "type": GGMLType.F32,
                    "offset": 0,
                }
            ],
            data_section_bytes=bytes(64 * 4),
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "2"))
        assert _rule_ids(findings) == []

    def test_valid_multiple_metadata_types(self) -> None:
        """All supported scalar and array value types."""
        kv_pairs = [
            ("u8", GGUFValueType.UINT8, 255),
            ("i8", GGUFValueType.INT8, -128),
            ("u16", GGUFValueType.UINT16, 65535),
            ("i16", GGUFValueType.INT16, -32768),
            ("u32", GGUFValueType.UINT32, 100000),
            ("i32", GGUFValueType.INT32, -100000),
            ("f32", GGUFValueType.FLOAT32, 3.14159),
            ("bool_t", GGUFValueType.BOOL, True),
            ("str", GGUFValueType.STRING, "clean_string"),
            ("arr_i32", GGUFValueType.ARRAY, (GGUFValueType.INT32, [1, 2, 3, 4])),
            ("arr_str", GGUFValueType.ARRAY, (GGUFValueType.STRING, ["a", "b", "c"])),
            ("u64", GGUFValueType.UINT64, 12345678901234),
            ("i64", GGUFValueType.INT64, -12345678901234),
            ("f64", GGUFValueType.FLOAT64, 2.718281828459),
        ]
        data = _build_gguf(version=3, kv_pairs=kv_pairs)
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "3"))
        assert _rule_ids(findings) == []


# ---------------------------------------------------------------------------
# GU202: Magic / Version Validation
# ---------------------------------------------------------------------------


class TestGU202MagicAndVersion:
    """Tests for GU202: Invalid Magic or Unsupported GGUF Version."""

    def test_invalid_magic_emits_gu202(self) -> None:
        data = b"XXXX" + struct.pack("<IQQ", 3, 0, 0)
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "4"))
        assert "GU202" in _rule_ids(findings)

    def test_unsupported_version_0_emits_gu202(self) -> None:
        data = GGUF_MAGIC + struct.pack("<IQQ", 0, 0, 0)
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "5"))
        assert "GU202" in _rule_ids(findings)

    def test_unsupported_version_4_emits_gu202(self) -> None:
        data = GGUF_MAGIC + struct.pack("<IQQ", 4, 0, 0)
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "6"))
        assert "GU202" in _rule_ids(findings)

    def test_gu202_severity_and_confidence(self) -> None:
        data = b"BADM" + struct.pack("<IQQ", 3, 0, 0)
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "7"))
        f = next(x for x in findings if x.rule_id == "GU202")
        assert f.severity == Severity.HIGH
        assert f.confidence == Confidence.HIGH
        assert f.remediation != ""


# ---------------------------------------------------------------------------
# GU203: Truncated / Malformed Header
# ---------------------------------------------------------------------------


class TestGU203TruncatedStream:
    """Tests for GU203: Truncated or Malformed GGUF Stream."""

    def test_empty_file_emits_gu203(self) -> None:
        findings = GGUFAnalyzer().analyze(_make_artifact(b"", "8"))
        assert "GU203" in _rule_ids(findings)

    def test_short_file_emits_gu203(self) -> None:
        findings = GGUFAnalyzer().analyze(_make_artifact(b"GGU", "9"))
        assert "GU203" in _rule_ids(findings)

    def test_truncated_version_emits_gu203(self) -> None:
        findings = GGUFAnalyzer().analyze(_make_artifact(GGUF_MAGIC + b"\x01", "a"))
        assert "GU203" in _rule_ids(findings)

    def test_truncated_counts_emits_gu203(self) -> None:
        data = GGUF_MAGIC + struct.pack("<I", 3) + b"\x00" * 4  # needs 16 bytes for counts in v3
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "b"))
        assert "GU203" in _rule_ids(findings)

    def test_truncated_kv_string_len_emits_gu203(self) -> None:
        # Declares 1 KV pair, but stream ends before string bytes
        data = GGUF_MAGIC + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 100) + b"short"
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "c"))
        assert "GU203" in _rule_ids(findings)

    def test_truncated_tensor_descriptor_emits_gu203(self) -> None:
        # Declares 1 tensor, but file ends right after header
        data = GGUF_MAGIC + struct.pack("<IQQ", 3, 1, 0)
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "d"))
        assert "GU203" in _rule_ids(findings)


# ---------------------------------------------------------------------------
# GU201: Resource Limit Enforcement
# ---------------------------------------------------------------------------


class TestGU201ResourceLimits:
    """Tests for GU201: Resource Limits (KV count, tensor count)."""

    def test_oversized_kv_count_emits_gu201(self) -> None:
        huge_kv_count = 50_000
        data = GGUF_MAGIC + struct.pack("<IQQ", 3, 0, huge_kv_count)
        config = ScanConfig(max_gguf_kv_count=10_000)
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "e"), config)
        assert "GU201" in _rule_ids(findings)

    def test_oversized_tensor_count_emits_gu201(self) -> None:
        huge_tensors = 200_000
        data = GGUF_MAGIC + struct.pack("<IQQ", 3, huge_tensors, 0)
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "f"))
        assert "GU201" in _rule_ids(findings)


# ---------------------------------------------------------------------------
# GU104: Oversized Staging Buffer
# ---------------------------------------------------------------------------


class TestGU104OversizedBuffer:
    """Tests for GU104: Oversized Metadata Staging Buffer."""

    def test_oversized_string_value_emits_gu104(self) -> None:
        # String exceeding max_gguf_value_bytes
        oversized_str = "A" * (2000)
        config = ScanConfig(max_gguf_value_bytes=1000)
        data = _build_gguf(
            version=3,
            kv_pairs=[("custom.buffer", GGUFValueType.STRING, oversized_str)],
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "1"), config)
        assert "GU104" in _rule_ids(findings)


# ---------------------------------------------------------------------------
# GU204: Invalid Metadata Value Type
# ---------------------------------------------------------------------------


class TestGU204InvalidValueType:
    """Tests for GU204: Invalid Metadata Value Type."""

    def test_unknown_value_type_emits_gu204(self) -> None:
        # Craft a KV entry with value_type = 99
        header = GGUF_MAGIC + struct.pack("<IQQ", 3, 0, 1)
        key_bytes = _pack_str("some_key", 3)
        bad_type = struct.pack("<I", 99)
        data = header + key_bytes + bad_type + b"\x00" * 32
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "2"))
        assert "GU204" in _rule_ids(findings)


# ---------------------------------------------------------------------------
# GU205: Invalid or Duplicate Tensor Name
# ---------------------------------------------------------------------------


class TestGU205TensorNames:
    """Tests for GU205: Invalid or Duplicate Tensor Name."""

    def test_duplicate_tensor_name_emits_gu205(self) -> None:
        data = _build_gguf(
            version=3,
            tensors=[
                {"name": "weight", "dimensions": [10], "type": GGMLType.F32, "offset": 0},
                {"name": "weight", "dimensions": [10], "type": GGMLType.F32, "offset": 40},
            ],
            data_section_bytes=bytes(80),
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "3"))
        assert "GU205" in _rule_ids(findings)

    def test_empty_tensor_name_emits_gu205(self) -> None:
        data = _build_gguf(
            version=3,
            tensors=[{"name": "", "dimensions": [10], "type": GGMLType.F32, "offset": 0}],
            data_section_bytes=bytes(40),
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "4"))
        assert "GU205" in _rule_ids(findings)


# ---------------------------------------------------------------------------
# GU206: Invalid Tensor Dimensions
# ---------------------------------------------------------------------------


class TestGU206TensorDimensions:
    """Tests for GU206: Invalid Tensor Dimensions."""

    def test_excessive_dimensions_count_emits_gu206(self) -> None:
        data = _build_gguf(
            version=3,
            tensors=[
                {
                    "name": "hyper_tensor",
                    "dimensions": [2] * 20,  # 20 dims > MAX_TENSOR_DIMENSIONS (16)
                    "type": GGMLType.F32,
                    "offset": 0,
                }
            ],
            data_section_bytes=bytes(100),
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "5"))
        assert "GU206" in _rule_ids(findings)


# ---------------------------------------------------------------------------
# GU207: Invalid Tensor Type
# ---------------------------------------------------------------------------


class TestGU207TensorType:
    """Tests for GU207: Invalid Tensor Type."""

    def test_unknown_ggml_tensor_type_emits_gu207(self) -> None:
        data = _build_gguf(
            version=3,
            tensors=[{"name": "t", "dimensions": [4], "type": 999, "offset": 0}],
            data_section_bytes=bytes(64),
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "6"))
        assert "GU207" in _rule_ids(findings)


# ---------------------------------------------------------------------------
# GU208: Invalid Tensor Offset or Alignment
# ---------------------------------------------------------------------------


class TestGU208OffsetAndAlignment:
    """Tests for GU208: Invalid Tensor Offset or Alignment."""

    def test_invalid_alignment_non_power_of_two_emits_gu208(self) -> None:
        data = _build_gguf(
            version=3,
            kv_pairs=[("general.alignment", GGUFValueType.UINT32, 33)],
            alignment=33,  # 33 is not a power of 2
            tensors=[{"name": "t", "dimensions": [4], "type": GGMLType.F32, "offset": 0}],
            data_section_bytes=bytes(16),
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "7"))
        assert "GU208" in _rule_ids(findings)


# ---------------------------------------------------------------------------
# GU209: Tensor Range Outside File Bounds
# ---------------------------------------------------------------------------


class TestGU209TensorRangeOutOfBounds:
    """Tests for GU209: Tensor Range Outside File Bounds."""

    def test_tensor_extends_past_eof_emits_gu209(self) -> None:
        # Tensor needs 16 bytes, but data_section_bytes is only 4
        data = _build_gguf(
            version=3,
            tensors=[{"name": "t", "dimensions": [4], "type": GGMLType.F32, "offset": 0}],
            data_section_bytes=bytes(4),
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "8"))
        assert "GU209" in _rule_ids(findings)

    def test_huge_tensor_offset_emits_gu209(self) -> None:
        data = _build_gguf(
            version=3,
            tensors=[
                {"name": "t", "dimensions": [4], "type": GGMLType.F32, "offset": 10_000_000}
            ],
            data_section_bytes=bytes(16),
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "9"))
        assert "GU209" in _rule_ids(findings)


# ---------------------------------------------------------------------------
# GU210: Overlapping Tensor Ranges
# ---------------------------------------------------------------------------


class TestGU210TensorOverlap:
    """Tests for GU210: Overlapping Tensor Ranges."""

    def test_overlapping_tensors_emits_gu210(self) -> None:
        # Tensor A: offset 0..16, Tensor B: offset 8..24 (overlaps A)
        data = _build_gguf(
            version=3,
            tensors=[
                {"name": "tensor_a", "dimensions": [4], "type": GGMLType.F32, "offset": 0},
                {"name": "tensor_b", "dimensions": [4], "type": GGMLType.F32, "offset": 8},
            ],
            data_section_bytes=bytes(32),
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "a"))
        assert "GU210" in _rule_ids(findings)

    def test_adjacent_tensors_no_overlap(self) -> None:
        # Tensor A: offset 0..16, Tensor B: offset 16..32 (no overlap)
        data = _build_gguf(
            version=3,
            tensors=[
                {"name": "tensor_a", "dimensions": [4], "type": GGMLType.F32, "offset": 0},
                {"name": "tensor_b", "dimensions": [4], "type": GGMLType.F32, "offset": 16},
            ],
            data_section_bytes=bytes(32),
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "b"))
        assert "GU210" not in _rule_ids(findings)


# ---------------------------------------------------------------------------
# GU101: Code Execution Primitives in Metadata
# ---------------------------------------------------------------------------


class TestGU101CodeExecution:
    """Tests for GU101: Code Execution Primitives in Metadata."""

    @pytest.mark.parametrize(
        "val",
        [
            "eval('import os; os.system(\"id\")')",
            "exec('import subprocess')",
            "__import__('os').system('calc.exe')",
            "os.system('id')",
            "subprocess.Popen(['/bin/sh'])",
            "base64.b64decode('aW1wb3J0IG9z')",
            "compile('pass', '', 'exec')",
            "__builtins__.__import__('os')",
        ],
    )
    def test_code_exec_patterns_emit_gu101(self, val: str) -> None:
        data = _build_gguf(
            version=3,
            kv_pairs=[("custom.init_script", GGUFValueType.STRING, val)],
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "c"))
        assert "GU101" in _rule_ids(findings), f"Expected GU101 for {val!r}"

    def test_gu101_severity_is_critical(self) -> None:
        data = _build_gguf(
            version=3,
            kv_pairs=[("cmd", GGUFValueType.STRING, "eval('1+1')")],
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "d"))
        f = next(x for x in findings if x.rule_id == "GU101")
        assert f.severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# GU102: Path Traversal Sequences in Metadata
# ---------------------------------------------------------------------------


class TestGU102PathTraversal:
    """Tests for GU102: Path Traversal Sequences in Metadata."""

    @pytest.mark.parametrize(
        "val",
        [
            "../../../../etc/passwd",
            "..\\..\\Windows\\System32\\cmd.exe",
            "%2e%2e%2f%2e%2e%2fsecret.txt",
            "/etc/shadow",
            "C:\\Windows\\System32\\calc.exe",
        ],
    )
    def test_path_traversal_emits_gu102(self, val: str) -> None:
        data = _build_gguf(
            version=3,
            kv_pairs=[("tokenizer.model_path", GGUFValueType.STRING, val)],
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "e"))
        assert "GU102" in _rule_ids(findings), f"Expected GU102 for {val!r}"


# ---------------------------------------------------------------------------
# GU103: Credential / Secret Exposure in Metadata
# ---------------------------------------------------------------------------


class TestGU103Credentials:
    """Tests for GU103: Credential / Secret Exposure in Metadata."""

    @pytest.mark.parametrize(
        "val",
        [
            "AKIAIOSFODNN7EXAMPLE",
            "ghp_" + "a" * 36,
            "hf_" + "b" * 34,
            "sk-" + "c" * 48,
            "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg==\n-----END PRIVATE KEY-----",
            "api_key = secretpassword123",
        ],
    )
    def test_credential_patterns_emit_gu103(self, val: str) -> None:
        data = _build_gguf(
            version=3,
            kv_pairs=[("auth.token", GGUFValueType.STRING, val)],
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "f"))
        assert "GU103" in _rule_ids(findings), f"Expected GU103 for {val!r}"


# ---------------------------------------------------------------------------
# GU105: Suspicious URLs in Metadata
# ---------------------------------------------------------------------------


class TestGU105SuspiciousURL:
    """Tests for GU105: Suspicious URLs in Metadata."""

    def test_http_url_emits_gu105(self) -> None:
        data = _build_gguf(
            version=3,
            kv_pairs=[("source.repo", GGUFValueType.STRING, "http://c2.attacker.com/weights.bin")],
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "1"))
        assert "GU105" in _rule_ids(findings)

    def test_https_url_emits_gu105(self) -> None:
        data = _build_gguf(
            version=3,
            kv_pairs=[("general.source", GGUFValueType.STRING, "https://huggingface.co/model")],
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "2"))
        assert "GU105" in _rule_ids(findings)


# ---------------------------------------------------------------------------
# GU106: Script Injection Tags in Metadata
# ---------------------------------------------------------------------------


class TestGU106ScriptInjection:
    """Tests for GU106: Script Injection Tags in Metadata."""

    @pytest.mark.parametrize(
        "val",
        [
            "<script>alert('xss')</script>",
            '<iframe src="javascript:alert(1)">',
            "onload=stealCookies()",
            'onerror=fetch("http://attacker.com")',
            "javascript:void(0)",
        ],
    )
    def test_script_injection_emits_gu106(self, val: str) -> None:
        data = _build_gguf(
            version=3,
            kv_pairs=[("general.description", GGUFValueType.STRING, val)],
        )
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "3"))
        assert "GU106" in _rule_ids(findings), f"Expected GU106 for {val!r}"


# ---------------------------------------------------------------------------
# Evidence Properties & Coordinates
# ---------------------------------------------------------------------------


class TestGGUFEvidenceProperties:
    """Validate Evidence fields on GGUF findings."""

    def test_evidence_format_and_coordinates(self) -> None:
        data = b"BADM" + struct.pack("<IQQ", 3, 0, 0)
        findings = GGUFAnalyzer().analyze(_make_artifact(data, "4"))
        assert len(findings) > 0
        f = findings[0]
        assert f.format == ModelFormat.GGUF
        assert f.evidence.coordinate_space == CoordinateSpace.HEADER
        assert f.evidence.byte_offset is not None


# ---------------------------------------------------------------------------
# Submodule Unit Tests
# ---------------------------------------------------------------------------


class TestGGUFSubmodules:
    """Unit tests for GGUF reader, tensors, and metadata modules."""

    def test_compute_tensor_byte_size(self) -> None:
        # F32: 4 bytes per element
        assert compute_tensor_byte_size(GGMLType.F32, (10, 10)) == 400
        # F16: 2 bytes per element
        assert compute_tensor_byte_size(GGMLType.F16, (10, 10)) == 200
        # Q4_0: 32 elements per block, 18 bytes per block
        assert compute_tensor_byte_size(GGMLType.Q4_0, (32,)) == 18
        assert compute_tensor_byte_size(GGMLType.Q4_0, (64,)) == 36

    def test_reader_truncation_handling(self) -> None:
        reader = GGUFReader(b"GGUF\x03\x00\x00\x00")  # truncated counts
        hdr, _kvs, _off = reader.parse_header()
        assert hdr is None
        assert len(reader.issues) > 0

    def test_metadata_scanner_on_empty(self) -> None:
        findings = scan_gguf_metadata([])
        assert findings == []


# ---------------------------------------------------------------------------
# Safety Verification Tests
# ---------------------------------------------------------------------------


class TestGGUFSafetyGuarantees:
    """Verify zero-deserialization and zero dynamic execution in GGUF analyzer."""

    def test_no_torch_imported(self) -> None:
        import model_scanner.analyzers.gguf.analyzer as mod

        assert "torch" not in dir(mod)

    def test_no_llama_cpp_imported(self) -> None:
        import sys

        assert "llama_cpp" not in sys.modules

    def test_no_pickle_in_gguf_analyzer(self) -> None:
        import model_scanner.analyzers.gguf.analyzer as mod

        assert "pickle" not in dir(mod)
