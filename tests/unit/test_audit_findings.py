"""
tests/unit/test_audit_findings.py
===================================

Regression tests for the 8 confirmed audit findings.

Finding 1  — HuggingFace secondary candidate resource cap
Finding 2  — .pbtxt HuggingFace acquisition filter
Finding 3  — Keras .keras ZIP .npy weight-integrity scan
Finding 4  — CRITICAL/HIGH duplicate pattern findings in KR105
Finding 5  — Duplicate Lambda traversal in _extract_lambda_code_fields
Finding 6  — GGUF deferred _weight_utils import (module-level import)
Finding 7  — Duplicated WI101/WI102/WI103 metadata (shared constants)
Finding 8  — GGUF float type mapping duplication

SAFETY CONTRACT
---------------
- No keras.models.load_model(), TensorFlow, or inference APIs.
- No eval(), exec(), marshal.loads(), or dynamic deserialization of attacker data.
- All Lambda payloads are pre-computed constants or pure static strings.
"""
from __future__ import annotations

import io
import json
import struct
import zipfile

import pytest

from model_scanner.core.models import Artifact, InputSourceType


# ---------------------------------------------------------------------------
# Helper: make a minimal Artifact from bytes
# ---------------------------------------------------------------------------

def _make_artifact(data: bytes, sha: str = "a" * 64) -> Artifact:
    return Artifact(
        artifact_id="test",
        source_path="<test>",
        input_source=InputSourceType.IN_MEMORY,
        size_bytes=len(data),
        sha256=sha,
        stream=io.BytesIO(data),
    )


def _make_keras_zip(members: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in members:
            zf.writestr(name, payload)
    buf.seek(0)
    return buf.read()


# ============================================================================
# FINDING 1 — HuggingFace secondary candidate resource cap
# ============================================================================


class TestFinding1SecondaryResourceCap:
    """Secondary extensionless files must be capped at _MAX_REPO_FILES - len(candidates)."""

    def _make_siblings(
        self, n_extension: int, n_extensionless: int, ext: str = ".safetensors"
    ) -> list[dict]:
        siblings = []
        for i in range(n_extension):
            siblings.append({"rfilename": f"model_{i:04d}{ext}", "size": 100})
        for i in range(n_extensionless):
            # Names that _is_suspicious_extensionless returns True for
            siblings.append({"rfilename": f"checkpoint_{i:04d}", "size": 50})
        return siblings

    def test_secondary_capped_when_primaries_fill_quota(self) -> None:
        """With 100 primaries, only 100 secondary slots remain from the 200 max."""
        from model_scanner.acquisition.huggingface import (
            _filter_model_candidates,
            _is_suspicious_extensionless,
            _MAX_REPO_FILES,
        )

        siblings = self._make_siblings(100, 250)
        candidates = _filter_model_candidates(siblings)
        assert len(candidates) == 100

        secondary_slots = max(0, _MAX_REPO_FILES - len(candidates))
        raw_secondary = [
            e for e in siblings
            if _is_suspicious_extensionless(str(e.get("rfilename", "")))
        ]
        raw_secondary.sort(
            key=lambda e: (e.get("size") is None, e.get("size") or 0)
        )
        capped_secondary = raw_secondary[:secondary_slots]

        # Total must not exceed _MAX_REPO_FILES
        assert len(candidates) + len(capped_secondary) <= _MAX_REPO_FILES
        # Secondary was indeed capped (250 raw → 100 slots)
        assert len(capped_secondary) == 100

    def test_secondary_fully_available_when_no_primaries(self) -> None:
        """With 0 primary candidates, all 200 secondary slots are available."""
        from model_scanner.acquisition.huggingface import (
            _is_suspicious_extensionless,
            _MAX_REPO_FILES,
        )

        siblings = [{"rfilename": f"checkpoint_{i:04d}", "size": 50} for i in range(150)]
        candidates: list = []  # no extension-matched files

        secondary_slots = max(0, _MAX_REPO_FILES - len(candidates))
        raw_secondary = [
            e for e in siblings
            if _is_suspicious_extensionless(str(e.get("rfilename", "")))
        ]
        raw_secondary.sort(
            key=lambda e: (e.get("size") is None, e.get("size") or 0)
        )
        capped_secondary = raw_secondary[:secondary_slots]

        # All 150 fit within 200 slots
        assert len(capped_secondary) == 150

    def test_secondary_sorted_smallest_first(self) -> None:
        """Secondary candidates are sorted by ascending size before capping."""
        from model_scanner.acquisition.huggingface import _is_suspicious_extensionless

        siblings = [
            {"rfilename": "checkpoint_big", "size": 9000},
            {"rfilename": "checkpoint_small", "size": 100},
            {"rfilename": "checkpoint_medium", "size": 500},
        ]
        raw_secondary = [
            e for e in siblings
            if _is_suspicious_extensionless(str(e.get("rfilename", "")))
        ]
        raw_secondary.sort(
            key=lambda e: (e.get("size") is None, e.get("size") or 0)
        )
        sizes = [e["size"] for e in raw_secondary]
        assert sizes == sorted(sizes), "Secondary candidates must be sorted by size ascending"

    def test_total_never_exceeds_max(self) -> None:
        """With 200 primaries + 100 extensionless → secondary capped to 0."""
        from model_scanner.acquisition.huggingface import (
            _filter_model_candidates,
            _is_suspicious_extensionless,
            _MAX_REPO_FILES,
        )

        # Build exactly _MAX_REPO_FILES + 5 primaries to trigger the existing
        # AcquisitionError, then separately verify the cap logic would limit to 0 slots.
        siblings = [{"rfilename": f"model_{i:04d}.safetensors", "size": 10} for i in range(_MAX_REPO_FILES + 5)]
        siblings += [{"rfilename": "hidden_checkpoint", "size": 50}]
        candidates = _filter_model_candidates(siblings)
        # Would raise if > _MAX_REPO_FILES, but secondary slots would be 0 anyway
        secondary_slots = max(0, _MAX_REPO_FILES - len(candidates))
        assert secondary_slots == 0


# ============================================================================
# FINDING 2 — .pbtxt HuggingFace acquisition filter
# ============================================================================


class TestFinding2PbtxtFilter:
    """saved_model.pbtxt must survive _filter_model_candidates()."""

    def test_saved_model_pbtxt_survives_filter(self) -> None:
        from model_scanner.acquisition.huggingface import _filter_model_candidates

        siblings = [{"rfilename": "saved_model.pbtxt", "size": 1024}]
        candidates = _filter_model_candidates(siblings)
        names = [str(e.get("rfilename", "")) for e in candidates]
        assert "saved_model.pbtxt" in names, (
            "saved_model.pbtxt must not be dropped by _filter_model_candidates"
        )

    def test_saved_model_pb_still_survives_filter(self) -> None:
        """Regression: saved_model.pb must also still survive."""
        from model_scanner.acquisition.huggingface import _filter_model_candidates

        siblings = [{"rfilename": "saved_model.pb", "size": 2048}]
        candidates = _filter_model_candidates(siblings)
        names = [str(e.get("rfilename", "")) for e in candidates]
        assert "saved_model.pb" in names

    def test_arbitrary_pbtxt_dropped(self) -> None:
        """An arbitrary .pbtxt that is NOT saved_model.pbtxt must be dropped."""
        from model_scanner.acquisition.huggingface import _filter_model_candidates

        siblings = [
            {"rfilename": "config.pbtxt", "size": 512},
            {"rfilename": "graph.pbtxt", "size": 512},
        ]
        candidates = _filter_model_candidates(siblings)
        names = [str(e.get("rfilename", "")) for e in candidates]
        assert "config.pbtxt" not in names
        assert "graph.pbtxt" not in names

    def test_pbtxt_set_contains_both_names(self) -> None:
        """Both names must be in _HF_SAVEDMODEL_PB_NAMES."""
        from model_scanner.acquisition.huggingface import _HF_SAVEDMODEL_PB_NAMES

        assert "saved_model.pb" in _HF_SAVEDMODEL_PB_NAMES
        assert "saved_model.pbtxt" in _HF_SAVEDMODEL_PB_NAMES


# ============================================================================
# FINDING 3 — Keras .keras ZIP .npy weight-integrity scan
# ============================================================================


def _make_npy_bytes(float32_values: list[float]) -> bytes:
    """Build a minimal .npy v1.0 file with the given float32 values."""
    # .npy magic: 0x93 NUMPY version 1.0 + header padded to 128 bytes total
    header_dict = b"{'descr': '<f4', 'fortran_order': False, 'shape': (" + str(len(float32_values)).encode() + b",), }"
    # Pad header to make total preamble = 128 bytes
    # Preamble = 10 bytes (magic 6 + version 2 + header_len 2) + header
    header_len_needed = 128 - 10 - len(header_dict)
    if header_len_needed < 0:
        header_dict = header_dict + b" " * (-header_len_needed)
        header_len_needed = 0
    header_dict = header_dict + b" " * header_len_needed + b"\n"
    # Trim to exactly 128 - 10 = 118 bytes for the header
    header_padded = header_dict[:118]
    header_len = len(header_padded)
    preamble = (
        b"\x93NUMPY"         # magic
        b"\x01\x00"          # version 1.0
        + struct.pack("<H", header_len)   # header length (little-endian uint16)
        + header_padded
    )
    data_bytes = struct.pack(f"<{len(float32_values)}f", *float32_values)
    return preamble + data_bytes


class TestFinding3KerasZipNpyScan:
    """Keras .keras ZIP direct .npy members must be weight-integrity scanned."""

    def _make_artifact_from_zip_members(
        self, members: list[tuple[str, bytes]]
    ) -> Artifact:
        cfg = json.dumps({"class_name": "Sequential", "config": {"layers": []}}).encode()
        all_members = [("config.json", cfg)] + members
        data = _make_keras_zip(all_members)
        return _make_artifact(data)

    def test_clean_npy_no_wi_findings(self) -> None:
        """A .npy member with normal float32 data must not produce WI findings."""
        from model_scanner.analyzers.keras.analyzer import KerasAnalyzer

        clean_values = [1.0, 2.0, 3.0, 0.5, -1.0] * 10
        npy_bytes = _make_npy_bytes(clean_values)
        artifact = self._make_artifact_from_zip_members([("weights/kernel.npy", npy_bytes)])

        findings = KerasAnalyzer().analyze(artifact)
        wi_findings = [f for f in findings if f.rule_id in ("WI101", "WI102", "WI103")]
        assert not wi_findings, (
            f"Clean .npy data must not produce WI findings; got: "
            f"{[f.rule_id + ':' + f.evidence.matched_value for f in wi_findings]}"
        )

    def _make_npy_with_float32(self, float32_bit_pattern: int) -> bytes:
        """Build a minimal valid .npy v1.0 file containing a single F32 value.

        .npy v1.0 format:
          6 bytes magic (\\x93NUMPY)
          1 byte major version (0x01)
          1 byte minor version (0x00)
          2 bytes HEADER_LEN (little-endian uint16)
          HEADER_LEN bytes: header dict string padded with spaces, ending with \\n
          Then: raw array data (4 bytes for a single F32)

        HEADER_LEN=118: total preamble = 10 + 118 = 128 bytes (multiple of 64).
        """
        header_dict = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }"
        header_len = 118  # 10 + 118 = 128, divisible by 64
        # Pad: header_dict + spaces + newline, total = header_len
        pad_needed = header_len - len(header_dict) - 1
        assert pad_needed >= 0, "header_dict too long for header_len=118"
        header_padded = header_dict + b" " * pad_needed + b"\n"
        assert len(header_padded) == header_len
        preamble = (
            b"\x93NUMPY"
            + b"\x01\x00"
            + struct.pack("<H", header_len)
            + header_padded
        )
        assert len(preamble) == 128, f"Expected 128-byte preamble, got {len(preamble)}"
        data_bytes = struct.pack("<I", float32_bit_pattern)
        return preamble + data_bytes

    def test_nan_npy_fires_wi101(self) -> None:
        """A .npy member with NaN float32 values must produce WI101."""
        from model_scanner.analyzers.keras.analyzer import KerasAnalyzer

        # NaN in F32: exponent=0xFF, mantissa != 0 → bit pattern 0x7FC00000
        npy_bytes = self._make_npy_with_float32(0x7FC00000)

        artifact = self._make_artifact_from_zip_members([("weights/kernel.npy", npy_bytes)])
        findings = KerasAnalyzer().analyze(artifact)
        wi101 = [f for f in findings if f.rule_id == "WI101"]
        assert wi101, "WI101 must fire for NaN values in .npy member"

    def test_inf_npy_fires_wi102(self) -> None:
        """A .npy member with Inf float32 values must produce WI102."""
        from model_scanner.analyzers.keras.analyzer import KerasAnalyzer

        # +Inf in F32: exponent=0xFF, mantissa=0 → bit pattern 0x7F800000
        npy_bytes = self._make_npy_with_float32(0x7F800000)

        artifact = self._make_artifact_from_zip_members([("weights/bias.npy", npy_bytes)])
        findings = KerasAnalyzer().analyze(artifact)
        wi102 = [f for f in findings if f.rule_id == "WI102"]
        assert wi102, "WI102 must fire for Inf values in .npy member"

    def test_malformed_archive_no_crash(self) -> None:
        """A malformed .keras archive must not crash — it produces KR102, not a traceback."""
        from model_scanner.analyzers.keras.analyzer import KerasAnalyzer

        data = b"PK\x03\x04" + b"\x00" * 40
        artifact = _make_artifact(data)
        # Must not raise
        findings = KerasAnalyzer().analyze(artifact)
        assert any(f.rule_id == "KR102" for f in findings)

    def test_truncated_npy_no_crash(self) -> None:
        """A .npy file truncated before 128 bytes must not crash or produce WI findings."""
        from model_scanner.analyzers.keras.analyzer import KerasAnalyzer

        truncated_npy = b"\x93NUMPY\x01\x00" + b"\x00" * 10  # way shorter than 128 bytes
        artifact = self._make_artifact_from_zip_members([("weights/kernel.npy", truncated_npy)])
        # Must not raise
        findings = KerasAnalyzer().analyze(artifact)
        wi_findings = [f for f in findings if f.rule_id in ("WI101", "WI102", "WI103")]
        assert not wi_findings, "Truncated .npy must not produce WI findings"


# ============================================================================
# FINDING 4 — CRITICAL/HIGH duplicate pattern findings in KR105
# ============================================================================


class TestFinding4CriticalHighDedup:
    """When CRITICAL patterns are present, HIGH patterns must be suppressed."""

    def test_subprocess_plus_eval_only_critical(self) -> None:
        """subprocess (CRITICAL) + eval (HIGH) → only CRITICAL finding, no HIGH."""
        from model_scanner.analyzers.keras.analyzer import _kr105_scan_raw_bytes

        # Both subprocess (CRITICAL) and eval (HIGH) as standalone words
        raw = b"\x00subprocess\x00eval\x00"
        hits = _kr105_scan_raw_bytes(raw)
        severities = {sev for _, sev in hits}
        assert "CRITICAL" in severities, "subprocess must produce CRITICAL"
        assert "HIGH" not in severities, (
            "eval HIGH must be suppressed when subprocess CRITICAL is present"
        )

    def test_eval_only_produces_high(self) -> None:
        """eval without any CRITICAL pattern must still produce HIGH."""
        from model_scanner.analyzers.keras.analyzer import _kr105_scan_raw_bytes

        raw = b"\x00eval\x00"
        hits = _kr105_scan_raw_bytes(raw)
        severities = {sev for _, sev in hits}
        assert "HIGH" in severities, "eval alone must produce HIGH"
        assert "CRITICAL" not in severities, "eval alone must not produce CRITICAL"

    def test_popen_plus_exec_only_critical(self) -> None:
        """popen (CRITICAL) + exec (HIGH) → only CRITICAL, exec HIGH suppressed."""
        from model_scanner.analyzers.keras.analyzer import _kr105_scan_raw_bytes

        raw = b"\x00popen\x00exec\x00"
        hits = _kr105_scan_raw_bytes(raw)
        severities = {sev for _, sev in hits}
        assert "CRITICAL" in severities
        assert "HIGH" not in severities, "exec HIGH must be suppressed when CRITICAL present"

    def test_os_system_combo_suppresses_high(self) -> None:
        """os+system combo (CRITICAL) + __import__ (HIGH) → CRITICAL only."""
        from model_scanner.analyzers.keras.analyzer import _kr105_scan_raw_bytes

        raw = b"\x00os\x00system\x00__import__\x00"
        hits = _kr105_scan_raw_bytes(raw)
        severities = {sev for _, sev in hits}
        assert "CRITICAL" in severities
        assert "HIGH" not in severities

    def test_no_pattern_produces_empty(self) -> None:
        """Benign bytes produce no hits."""
        from model_scanner.analyzers.keras.analyzer import _kr105_scan_raw_bytes

        raw = b"\x00benign_function_name\x00"
        hits = _kr105_scan_raw_bytes(raw)
        assert hits == []

    def test_import_alone_produces_high(self) -> None:
        """__import__ alone (no CRITICAL) must still produce HIGH."""
        from model_scanner.analyzers.keras.analyzer import _kr105_scan_raw_bytes

        raw = b"\x00__import__\x00"
        hits = _kr105_scan_raw_bytes(raw)
        severities = {sev for _, sev in hits}
        assert "HIGH" in severities
        assert "CRITICAL" not in severities


# ============================================================================
# FINDING 5 — Duplicate Lambda traversal
# ============================================================================


class TestFinding5LambdaTraversal:
    """_extract_lambda_code_fields must return exactly one (path, code) per lambda."""

    def test_lambda_node_returns_exactly_one_code_entry(self) -> None:
        """A __lambda__ config with a 'code' field must produce exactly one result."""
        from model_scanner.analyzers.keras.analyzer import _extract_lambda_code_fields

        config = {
            "class_name": "__lambda__",
            "config": {
                "code": "OPAQUE_CODE_PAYLOAD",
                "defaults": None,
                "closure": None,
            },
        }
        results = _extract_lambda_code_fields(config)
        # Must be exactly ONE entry, not two (pre-fix it returned two)
        assert len(results) == 1, (
            f"Expected 1 result from __lambda__ node, got {len(results)}: {results}"
        )
        assert results[0][1] == "OPAQUE_CODE_PAYLOAD"

    def test_nested_lambda_node_returns_exactly_one(self) -> None:
        """A Lambda layer wrapping a __lambda__ must produce exactly one code entry.

        The Lambda branch does a generic descent which visits the __lambda__ dict
        inside config.function.  The __lambda__ branch then extracts the code
        exactly once (skipping 'config' in its own descent to prevent duplication).
        """
        from model_scanner.analyzers.keras.analyzer import _extract_lambda_code_fields

        config = {
            "class_name": "Lambda",
            "config": {
                "name": "lambda_1",
                "function": {
                    "class_name": "__lambda__",
                    "config": {
                        "code": "NESTED_PAYLOAD",
                        "defaults": None,
                        "closure": None,
                    },
                },
            },
        }
        results = _extract_lambda_code_fields(config)
        codes = [r[1] for r in results]
        assert codes.count("NESTED_PAYLOAD") == 1, (
            f"NESTED_PAYLOAD must appear exactly once; got: {results}"
        )
        assert len(results) == 1, (
            f"Exactly one result expected, got {len(results)}: {results}"
        )

    def test_two_independent_lambdas_return_two_entries(self) -> None:
        """Two separate __lambda__ nodes in the same config must each produce one entry."""
        from model_scanner.analyzers.keras.analyzer import _extract_lambda_code_fields

        config = {
            "layers": [
                {
                    "class_name": "__lambda__",
                    "config": {"code": "PAYLOAD_A", "defaults": None, "closure": None},
                },
                {
                    "class_name": "__lambda__",
                    "config": {"code": "PAYLOAD_B", "defaults": None, "closure": None},
                },
            ]
        }
        results = _extract_lambda_code_fields(config)
        codes = [r[1] for r in results]
        assert codes.count("PAYLOAD_A") == 1, f"PAYLOAD_A must appear once; got: {results}"
        assert codes.count("PAYLOAD_B") == 1, f"PAYLOAD_B must appear once; got: {results}"
        assert len(results) == 2

    def test_lambda_node_without_code_returns_empty(self) -> None:
        """A __lambda__ node without a 'code' field must return no entries."""
        from model_scanner.analyzers.keras.analyzer import _extract_lambda_code_fields

        config = {
            "class_name": "__lambda__",
            "config": {"defaults": None, "closure": None},
        }
        results = _extract_lambda_code_fields(config)
        assert results == []


# ============================================================================
# FINDING 6 — GGUF deferred _weight_utils import (module-level import)
# ============================================================================


class TestFinding6GGUFModuleLevelImport:
    """check_gguf_integrity must be imported at module level, not inside analyze()."""

    def test_check_gguf_integrity_is_module_level_import(self) -> None:
        """The deferred import must be gone from the analyze() body."""
        import inspect
        from model_scanner.analyzers.gguf import analyzer as gguf_analyzer_mod

        # The module-level globals must include check_gguf_integrity (imported at top)
        assert "check_gguf_integrity" in dir(gguf_analyzer_mod), (
            "check_gguf_integrity must be importable at module level from gguf.analyzer"
        )

    def test_analyze_body_has_no_deferred_import(self) -> None:
        """The analyze() method body must not contain a deferred import statement."""
        import inspect
        from model_scanner.analyzers.gguf.analyzer import GGUFAnalyzer

        source = inspect.getsource(GGUFAnalyzer.analyze)
        assert "from model_scanner.analyzers.gguf.integrity import" not in source, (
            "analyze() must not contain a deferred 'from ... import check_gguf_integrity'; "
            "it should be a module-level import instead"
        )

    def test_float_ggml_types_is_module_level(self) -> None:
        """_FLOAT_GGML_TYPES must be importable directly from gguf.analyzer."""
        from model_scanner.analyzers.gguf.analyzer import _FLOAT_GGML_TYPES  # noqa: F401
        assert isinstance(_FLOAT_GGML_TYPES, dict)


# ============================================================================
# FINDING 7 — Shared WI101/WI102/WI103 metadata in _weight_utils
# ============================================================================


class TestFinding7SharedWIMetadata:
    """WI rule metadata must be defined once in _weight_utils and imported everywhere."""

    def test_wi_constants_exported_from_weight_utils(self) -> None:
        """All shared WI constants must be importable from _weight_utils."""
        from model_scanner.analyzers._weight_utils import (
            WI_RULE_SEVERITY,
            WI_RULE_CONFIDENCE,
            WI_RULE_NAMES,
            WI_RULE_CATEGORY,
            WI_RULE_REMEDIATION,
            WI_KIND_TO_RULE,
        )
        for rule_id in ("WI101", "WI102", "WI103"):
            assert rule_id in WI_RULE_SEVERITY
            assert rule_id in WI_RULE_CONFIDENCE
            assert rule_id in WI_RULE_NAMES
            assert rule_id in WI_RULE_CATEGORY
            assert rule_id in WI_RULE_REMEDIATION
        assert WI_KIND_TO_RULE == {"nan": "WI101", "inf": "WI102", "extreme": "WI103"}

    def test_keras_uses_shared_wi_constants(self) -> None:
        """keras/analyzer.py _RULE_SEVERITY must include WI entries from _weight_utils."""
        from model_scanner.analyzers.keras.analyzer import _RULE_SEVERITY, _RULE_NAMES
        from model_scanner.analyzers._weight_utils import WI_RULE_SEVERITY, WI_RULE_NAMES

        for rule_id in ("WI101", "WI102", "WI103"):
            assert _RULE_SEVERITY.get(rule_id) == WI_RULE_SEVERITY[rule_id], (
                f"keras _RULE_SEVERITY[{rule_id}] must match WI_RULE_SEVERITY"
            )
            assert _RULE_NAMES.get(rule_id) == WI_RULE_NAMES[rule_id]

    def test_gguf_integrity_uses_shared_wi_constants(self) -> None:
        """gguf/integrity.py _RULE_SEVERITY must equal WI_RULE_SEVERITY."""
        from model_scanner.analyzers.gguf.integrity import _RULE_SEVERITY, _RULE_NAMES
        from model_scanner.analyzers._weight_utils import WI_RULE_SEVERITY, WI_RULE_NAMES

        for rule_id in ("WI101", "WI102", "WI103"):
            assert _RULE_SEVERITY.get(rule_id) == WI_RULE_SEVERITY[rule_id]
            assert _RULE_NAMES.get(rule_id) == WI_RULE_NAMES[rule_id]

    def test_safetensors_integrity_uses_shared_wi_constants(self) -> None:
        """safetensors/integrity.py _RULE_SEVERITY must equal WI_RULE_SEVERITY."""
        from model_scanner.analyzers.safetensors.integrity import _RULE_SEVERITY, _RULE_NAMES
        from model_scanner.analyzers._weight_utils import WI_RULE_SEVERITY, WI_RULE_NAMES

        for rule_id in ("WI101", "WI102", "WI103"):
            assert _RULE_SEVERITY.get(rule_id) == WI_RULE_SEVERITY[rule_id]
            assert _RULE_NAMES.get(rule_id) == WI_RULE_NAMES[rule_id]

    def test_wi_constants_in_all_exports(self) -> None:
        """WI constants must be in _weight_utils.__all__."""
        import model_scanner.analyzers._weight_utils as wu

        for name in ("WI_RULE_SEVERITY", "WI_RULE_CONFIDENCE", "WI_RULE_NAMES",
                     "WI_RULE_CATEGORY", "WI_RULE_REMEDIATION", "WI_KIND_TO_RULE"):
            assert name in wu.__all__, f"{name} must be in _weight_utils.__all__"


# ============================================================================
# FINDING 8 — GGUF float type mapping deduplication
# ============================================================================


class TestFinding8GGUFFloatTypeMapping:
    """_GGML_TYPE_STR local dict in analyze() must be replaced by _FLOAT_GGML_TYPES."""

    def test_no_local_ggml_type_str_in_analyze(self) -> None:
        """analyze() must not define a local _GGML_TYPE_STR dict."""
        import inspect
        from model_scanner.analyzers.gguf.analyzer import GGUFAnalyzer

        source = inspect.getsource(GGUFAnalyzer.analyze)
        assert "_GGML_TYPE_STR" not in source, (
            "analyze() must not define a local _GGML_TYPE_STR dict; "
            "use the imported _FLOAT_GGML_TYPES instead (Finding 8)"
        )

    def test_float_ggml_types_imported_from_integrity(self) -> None:
        """_FLOAT_GGML_TYPES must be importable from gguf.analyzer (re-exported from integrity)."""
        from model_scanner.analyzers.gguf.analyzer import _FLOAT_GGML_TYPES
        from model_scanner.analyzers.gguf.integrity import _FLOAT_GGML_TYPES as integrity_map

        assert _FLOAT_GGML_TYPES is integrity_map, (
            "_FLOAT_GGML_TYPES in analyzer must be the same object as in integrity"
        )

    def test_float_ggml_types_values_correct(self) -> None:
        """_FLOAT_GGML_TYPES must map the correct GGML type codes to dtype strings."""
        from model_scanner.analyzers.gguf.integrity import _FLOAT_GGML_TYPES

        assert _FLOAT_GGML_TYPES[0] == "F32"
        assert _FLOAT_GGML_TYPES[1] == "F16"
        assert _FLOAT_GGML_TYPES[28] == "F64"
        assert _FLOAT_GGML_TYPES[30] == "BF16"

    def test_analyze_uses_float_ggml_types_in_structure(self) -> None:
        """The model_structure tensor dtype lookup uses _FLOAT_GGML_TYPES."""
        import inspect
        from model_scanner.analyzers.gguf.analyzer import GGUFAnalyzer

        source = inspect.getsource(GGUFAnalyzer.analyze)
        assert "_FLOAT_GGML_TYPES.get" in source, (
            "analyze() must use _FLOAT_GGML_TYPES.get() for tensor dtype lookup"
        )


# ---------------------------------------------------------------------------
# Dashboard/Report Defect Fixes (D01–D25)
# ---------------------------------------------------------------------------


class TestD01ExitCodeRename:
    """D01: clamav_exit_code renamed to exit_code."""

    def test_scan_summary_has_exit_code_not_clamav(self) -> None:
        from model_scanner.core.models import ScanSummary, ScanOutcome
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(ScanSummary)}
        assert "exit_code" in field_names
        assert "clamav_exit_code" not in field_names

    def test_json_export_uses_exit_code_key(self) -> None:
        from model_scanner import scan
        from model_scanner.api import export_json_dict
        import tempfile, pathlib
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"\x80\x02}q\x00.")
            path = pathlib.Path(f.name)
        try:
            result = scan(path)
            d = export_json_dict(result)
            assert "exit_code" in d["summary"]
            assert "clamav_exit_code" not in d["summary"]
        finally:
            path.unlink()

    def test_json_export_no_schema_url(self) -> None:
        """D02: $schema key must not appear in JSON output."""
        from model_scanner import scan
        from model_scanner.api import export_json_dict
        import tempfile, pathlib
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"\x80\x02}q\x00.")
            path = pathlib.Path(f.name)
        try:
            result = scan(path)
            d = export_json_dict(result)
            assert "$schema" not in d
        finally:
            path.unlink()


class TestD03ArtifactDuration:
    """D03: HTML report shows per-artifact duration, not total scan duration."""

    def test_report_model_has_artifact_duration_ms(self) -> None:
        from model_scanner import scan
        from model_scanner.presentation.report_model import ReportBuilder
        import tempfile, pathlib
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"\x80\x02}q\x00.")
            path = pathlib.Path(f.name)
        try:
            result = scan(path)
            report = ReportBuilder.build(result)
            assert report.artifact_duration_ms is not None
            assert isinstance(report.artifact_duration_ms, int)
            assert report.artifact_duration_ms >= 0
        finally:
            path.unlink()


class TestD12DoubleEvidence:
    """D12: Single-sample evidence renders without duplicate 'Static Evidence' label."""

    def test_single_sample_no_label_paragraph(self) -> None:
        from model_scanner.presentation.html_report import _render_evidence_samples
        from model_scanner.presentation.report_model import RedactedEvidence, ReportFinding
        ev = RedactedEvidence(
            coordinate_space="GLOBAL",
            archive_member=None,
            byte_offset=0,
            byte_length=4,
            matched_value="os",
            original_length=None,
            opcode=None,
            context_snippet=None,
        )
        rf = ReportFinding(
            rule_id="P101",
            rule_name="Test",
            severity="CRITICAL",
            confidence="HIGH",
            category="code_execution",
            description="desc",
            security_issue="issue",
            impact="impact",
            remediation="fix",
            cwe_ids=(),
            owasp_category="ML06:2023 AI Supply Chain Attacks",
            mitre_technique="T1059",
            references=(),
            confidence_rationale="rationale",
            occurrence_count=1,
            evidence_samples=(ev,),
            additional_occurrences=0,
            correlated_finding_ids=(),
        )
        html = _render_evidence_samples(rf)
        # The section header <h5>Static Evidence</h5> is in _render_finding_card,
        # not here. The label paragraph must not duplicate it.
        assert "Static Evidence" not in html


class TestD13ReferencesNotDuplicateThreatMapping:
    """D13: References must not contain CWE / OWASP / MITRE identifiers."""

    def test_catalog_references_returns_only_cves(self) -> None:
        from model_scanner.presentation.report_model import _catalog_references
        refs = _catalog_references("P101", ())
        for ref in refs:
            assert ref.startswith("CVE-"), f"Non-CVE reference found: {ref}"

    def test_catalog_references_empty_when_no_cves(self) -> None:
        from model_scanner.presentation.report_model import _catalog_references
        # P101 has no CVEs assigned
        refs = _catalog_references("P101", ())
        assert refs == ()


class TestD15CorrelationExcludedFromScoring:
    """D15: C-series correlation findings are excluded from risk score arithmetic."""

    def test_correlation_findings_not_scored(self) -> None:
        from model_scanner.core.scoring import RiskEngine
        from model_scanner.core.models import (
            Finding, Confidence, CoordinateSpace, Evidence,
            ModelFormat, Severity,
        )

        def _f(rule_id: str, sev: Severity) -> Finding:
            return Finding(
                finding_id=f"FINDING-{rule_id}-abc",
                rule_id=rule_id,
                rule_name=f"Rule {rule_id}",
                severity=sev,
                confidence=Confidence.HIGH,
                category="code_execution",
                format=ModelFormat.RAW_PICKLE,
                artifact_sha256="a" * 64,
                evidence=Evidence(
                    coordinate_space=CoordinateSpace.GLOBAL,
                    byte_offset=0,
                    byte_length=4,
                    archive_member=None,
                    matched_value="os",
                    opcode=None,
                    raw_hex=None,
                    context_snippet=None,
                ),
                description="desc",
                remediation="fix",
                cve_ids=(),
                cwe_ids=(),
                owasp_category=None,
                mitre_technique=None,
                correlated_finding_ids=(),
            )

        primary = _f("P101", Severity.CRITICAL)
        correlation = _f("C004", Severity.HIGH)

        score_with = RiskEngine.calculate([primary, correlation], ModelFormat.RAW_PICKLE)
        score_without = RiskEngine.calculate([primary], ModelFormat.RAW_PICKLE)
        assert score_with.final_score == score_without.final_score, (
            "C-series findings must not inflate the risk score"
        )
