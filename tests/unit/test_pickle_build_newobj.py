"""
tests.unit.test_pickle_build_newobj
====================================

Phase 4 regression tests for:

1. Suspicious BUILD opcode -> P106 finding
2. Benign BUILD opcode (safe class) -> no P106 finding
3. Suspicious NEWOBJ/NEWOBJ_EX opcode -> P107 finding
4. Benign NEWOBJ/NEWOBJ_EX opcode (safe class) -> no P107 finding
5. Evidence byte offsets are preserved in findings
6. Cumulative PyTorch ZIP extraction limit (ResourceLimitError -> P208 finding)
7. Explicit torch loader classification overrides broad torch.* safe classification
8. Protocol-0 PyTorch member detection consistency

SECURITY GUARANTEE
------------------
Zero dynamic execution, zero deserialization, zero class instantiation.
pickle.loads(), pickle.load(), torch.load(), eval(), exec() are NEVER called.
All test payloads are hand-crafted synthetic byte literals or built via the
zipfile standard-library module in write mode only.

Design Reference
----------------
Phase 4 static analysis engine; docs/implementation_design.md Sections 14, 15.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from model_scanner.analyzers.pickle.analyzer import PickleAnalyzer
from model_scanner.analyzers.pickle.catalog import GlobalCategory, classify_global
from model_scanner.analyzers.pickle.disassembler import PickleDisassembler
from model_scanner.analyzers.pytorch.analyzer import PyTorchAnalyzer
from model_scanner.analyzers.pytorch.zip_reader import PyTorchZipReader
from model_scanner.core.models import (
    Artifact,
    Confidence,
    CoordinateSpace,
    InputSourceType,
    Severity,
)
from model_scanner.detection.magic import check_pickle_protocol_0_heuristic

# ---------------------------------------------------------------------------
# Synthetic fixture helpers -- NO pickle.loads / torch.load / eval / exec
# ---------------------------------------------------------------------------


def _make_zip(members: dict[str, bytes]) -> bytes:
    """Build an in-memory ZIP archive from a dict of {filename: content}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, content in members.items():
            zf.writestr(filename, content)
    return buf.getvalue()


def _make_artifact(zip_bytes: bytes, sha256_char: str = "a") -> Artifact:
    """Wrap raw bytes in an in-memory Artifact for PyTorchAnalyzer."""
    sha = sha256_char * 64
    return Artifact(
        artifact_id=f"art-{sha256_char}",
        source_path="synthetic_model.pt",
        input_source=InputSourceType.IN_MEMORY,
        size_bytes=len(zip_bytes),
        sha256=sha,
        stream=io.BytesIO(zip_bytes),
    )


# ---------------------------------------------------------------------------
# Hand-crafted pickle byte streams (Protocol 2 unless stated otherwise)
# ---------------------------------------------------------------------------

# Protocol 2 framing byte prefix
_P2_HEADER = b"\x80\x02"

# BUILD opcode byte
_BUILD = b"b"

# NEWOBJ opcode byte
_NEWOBJ = b"\x81"

# NEWOBJ_EX opcode byte
_NEWOBJ_EX = b"\x92"

# STOP opcode byte
_STOP = b"."

# EMPTY_TUPLE opcode byte (used as args placeholder)
_EMPTY_TUPLE = b")"


def _global_opcode(module: str, name: str) -> bytes:
    """Return the GLOBAL opcode bytes for (module, name)."""
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


def _short_binstring(s: str) -> bytes:
    """Return SHORT_BINSTRING opcode for a short string (Protocol 2)."""
    encoded = s.encode("utf-8")
    assert len(encoded) < 256, "Use BINSTRING for longer strings"
    return b"U" + bytes([len(encoded)]) + encoded


# ---------------------------------------------------------------------------
# P106 -- Suspicious BUILD State Mutation
# ---------------------------------------------------------------------------


class TestBuildP106Suspicious:
    """Regression: suspicious BUILD target -> P106 finding emitted."""

    _SUSPICIOUS_BUILD_PAYLOAD = (
        _P2_HEADER
        + _global_opcode("os", "system")
        + _short_binstring("injected_state")
        + _BUILD
        + _STOP
    )

    def test_suspicious_build_emits_p106(self) -> None:
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._SUSPICIOUS_BUILD_PAYLOAD,
            artifact_sha256="1" * 64,
        )
        rule_ids = [f.rule_id for f in findings]
        assert "P106" in rule_ids, (
            "Expected P106 for suspicious BUILD targeting os.system; "
            f"got rule_ids={rule_ids}"
        )

    def test_suspicious_build_p106_severity_is_high(self) -> None:
        """os.system is DANGEROUS, so P106 severity must be HIGH."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._SUSPICIOUS_BUILD_PAYLOAD,
            artifact_sha256="2" * 64,
        )
        p106 = next(f for f in findings if f.rule_id == "P106")
        assert p106.severity == Severity.HIGH, (
            f"Expected HIGH severity for DANGEROUS BUILD target, got {p106.severity}"
        )

    def test_suspicious_build_p106_confidence_is_high(self) -> None:
        """os.system is unambiguously DANGEROUS -- confidence must be HIGH."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._SUSPICIOUS_BUILD_PAYLOAD,
            artifact_sha256="3" * 64,
        )
        p106 = next(f for f in findings if f.rule_id == "P106")
        assert p106.confidence == Confidence.HIGH

    def test_suspicious_build_p106_opcode_is_build(self) -> None:
        """Evidence opcode field must be BUILD."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._SUSPICIOUS_BUILD_PAYLOAD,
            artifact_sha256="4" * 64,
        )
        p106 = next(f for f in findings if f.rule_id == "P106")
        assert p106.evidence.opcode == "BUILD"

    def test_suspicious_build_p106_byte_offset_is_non_negative(self) -> None:
        """Evidence byte_offset must be a non-negative integer."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._SUSPICIOUS_BUILD_PAYLOAD,
            artifact_sha256="5" * 64,
        )
        p106 = next(f for f in findings if f.rule_id == "P106")
        assert p106.evidence.byte_offset is not None
        assert p106.evidence.byte_offset >= 0

    def test_suspicious_build_p106_coordinate_space_artifact(self) -> None:
        """Default coordinate space is ARTIFACT."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._SUSPICIOUS_BUILD_PAYLOAD,
            artifact_sha256="6" * 64,
        )
        p106 = next(f for f in findings if f.rule_id == "P106")
        assert p106.evidence.coordinate_space == CoordinateSpace.ARTIFACT


class TestBuildP106UnknownTarget:
    """Regression: unresolvable BUILD target (<unknown_target>) -> P106 MEDIUM confidence."""

    _UNKNOWN_BUILD_PAYLOAD = _P2_HEADER + _BUILD + _STOP

    def test_unknown_target_build_emits_p106_medium_confidence(self) -> None:
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._UNKNOWN_BUILD_PAYLOAD,
            artifact_sha256="7" * 64,
        )
        rule_ids = [f.rule_id for f in findings]
        assert "P106" in rule_ids, (
            "Expected P106 for unresolvable BUILD target; "
            f"got rule_ids={rule_ids}"
        )
        p106 = next(f for f in findings if f.rule_id == "P106")
        assert p106.confidence == Confidence.MEDIUM
        assert p106.severity == Severity.MEDIUM


class TestBuildP106Benign:
    """Regression: safe BUILD target -> no P106 finding."""

    _SAFE_BUILD_PAYLOAD = (
        _P2_HEADER
        + _global_opcode("collections", "OrderedDict")
        + _EMPTY_TUPLE
        + _BUILD
        + _STOP
    )

    def test_safe_class_build_does_not_emit_p106(self) -> None:
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._SAFE_BUILD_PAYLOAD,
            artifact_sha256="8" * 64,
        )
        rule_ids = [f.rule_id for f in findings]
        assert "P106" not in rule_ids, (
            "P106 must NOT be emitted for safe BUILD target "
            f"(collections.OrderedDict); got rule_ids={rule_ids}"
        )

    def test_safe_pytorch_build_does_not_emit_p106(self) -> None:
        """torch._utils._rebuild_tensor_v2 via BUILD must not trigger P106."""
        safe_build = (
            _P2_HEADER
            + _global_opcode("torch._utils", "_rebuild_tensor_v2")
            + _EMPTY_TUPLE
            + _BUILD
            + _STOP
        )
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=safe_build,
            artifact_sha256="9" * 64,
        )
        rule_ids = [f.rule_id for f in findings]
        assert "P106" not in rule_ids, (
            "P106 must NOT be emitted for safe torch._utils._rebuild_tensor_v2 BUILD; "
            f"got rule_ids={rule_ids}"
        )


# ---------------------------------------------------------------------------
# P107 -- Suspicious NEWOBJ / NEWOBJ_EX Instantiation
# ---------------------------------------------------------------------------


class TestNewObjP107Suspicious:
    """Regression: suspicious NEWOBJ class -> P107 finding emitted."""

    _SUSPICIOUS_NEWOBJ_PAYLOAD = (
        _P2_HEADER
        + _global_opcode("ctypes", "CDLL")
        + _EMPTY_TUPLE
        + _NEWOBJ
        + _STOP
    )

    def test_suspicious_newobj_emits_p107(self) -> None:
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._SUSPICIOUS_NEWOBJ_PAYLOAD,
            artifact_sha256="a" * 64,
        )
        rule_ids = [f.rule_id for f in findings]
        assert "P107" in rule_ids, (
            "Expected P107 for NEWOBJ with ctypes.CDLL; "
            f"got rule_ids={rule_ids}"
        )

    def test_suspicious_newobj_p107_opcode_field_is_newobj(self) -> None:
        """Evidence opcode must be NEWOBJ (not NEWOBJ_EX)."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._SUSPICIOUS_NEWOBJ_PAYLOAD,
            artifact_sha256="b" * 64,
        )
        p107 = next(f for f in findings if f.rule_id == "P107")
        assert p107.evidence.opcode == "NEWOBJ"

    def test_suspicious_newobj_p107_severity_is_medium(self) -> None:
        """ctypes.CDLL is SUSPICIOUS (not DANGEROUS) -> MEDIUM severity."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._SUSPICIOUS_NEWOBJ_PAYLOAD,
            artifact_sha256="c" * 64,
        )
        p107 = next(f for f in findings if f.rule_id == "P107")
        assert p107.severity == Severity.MEDIUM

    def test_suspicious_newobj_p107_byte_offset_preserved(self) -> None:
        """Evidence byte_offset must be a non-negative integer."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._SUSPICIOUS_NEWOBJ_PAYLOAD,
            artifact_sha256="d" * 64,
        )
        p107 = next(f for f in findings if f.rule_id == "P107")
        assert p107.evidence.byte_offset is not None
        assert p107.evidence.byte_offset >= 0


class TestNewObjExP107Suspicious:
    """Regression: suspicious NEWOBJ_EX class -> P107 with opcode NEWOBJ_EX."""

    _NEWOBJ_EX_PAYLOAD = (
        b"\x80\x04"
        + _global_opcode("subprocess", "Popen")
        + _EMPTY_TUPLE
        + b"}"  # EMPTY_DICT
        + _NEWOBJ_EX
        + _STOP
    )

    def test_suspicious_newobj_ex_emits_p107(self) -> None:
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._NEWOBJ_EX_PAYLOAD,
            artifact_sha256="e" * 64,
        )
        rule_ids = [f.rule_id for f in findings]
        assert "P107" in rule_ids, (
            "Expected P107 for NEWOBJ_EX with subprocess.Popen; "
            f"got rule_ids={rule_ids}"
        )

    def test_newobj_ex_evidence_opcode_is_newobj_ex(self) -> None:
        """Evidence opcode field must be NEWOBJ_EX."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._NEWOBJ_EX_PAYLOAD,
            artifact_sha256="f" * 64,
        )
        p107 = next(f for f in findings if f.rule_id == "P107")
        assert p107.evidence.opcode == "NEWOBJ_EX"

    def test_newobj_ex_p107_dangerous_class_severity_is_high(self) -> None:
        """subprocess.Popen is DANGEROUS -> P107 severity must be HIGH."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._NEWOBJ_EX_PAYLOAD,
            artifact_sha256="0" * 64,
        )
        p107 = next(f for f in findings if f.rule_id == "P107")
        assert p107.severity == Severity.HIGH


class TestNewObjP107UnknownClass:
    """Regression: unresolvable NEWOBJ class (<unknown_cls>) -> P107 MEDIUM confidence."""

    _UNKNOWN_NEWOBJ_PAYLOAD = _P2_HEADER + _NEWOBJ + _STOP

    def test_unknown_class_newobj_emits_p107_medium(self) -> None:
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._UNKNOWN_NEWOBJ_PAYLOAD,
            artifact_sha256="1" * 64,
        )
        rule_ids = [f.rule_id for f in findings]
        assert "P107" in rule_ids, (
            "Expected P107 for unresolvable NEWOBJ class; "
            f"got rule_ids={rule_ids}"
        )
        p107 = next(f for f in findings if f.rule_id == "P107")
        assert p107.confidence == Confidence.MEDIUM
        assert p107.severity == Severity.MEDIUM


class TestNewObjP107Benign:
    """Regression: safe NEWOBJ class -> no P107 finding."""

    _SAFE_NEWOBJ_PAYLOAD = (
        _P2_HEADER
        + _global_opcode("collections", "OrderedDict")
        + _EMPTY_TUPLE
        + _NEWOBJ
        + _STOP
    )

    def test_safe_class_newobj_does_not_emit_p107(self) -> None:
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._SAFE_NEWOBJ_PAYLOAD,
            artifact_sha256="2" * 64,
        )
        rule_ids = [f.rule_id for f in findings]
        assert "P107" not in rule_ids, (
            "P107 must NOT be emitted for safe NEWOBJ with collections.OrderedDict; "
            f"got rule_ids={rule_ids}"
        )

    def test_safe_torch_nn_newobj_does_not_emit_p107(self) -> None:
        """torch.nn.modules.container.Sequential (SAFE) must not trigger P107."""
        safe_newobj = (
            _P2_HEADER
            + _global_opcode("torch.nn.modules.container", "Sequential")
            + _EMPTY_TUPLE
            + _NEWOBJ
            + _STOP
        )
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=safe_newobj,
            artifact_sha256="3" * 64,
        )
        rule_ids = [f.rule_id for f in findings]
        assert "P107" not in rule_ids, (
            "P107 must NOT be emitted for safe torch.nn.modules.container.Sequential NEWOBJ; "
            f"got rule_ids={rule_ids}"
        )


# ---------------------------------------------------------------------------
# Evidence byte offset preservation
# ---------------------------------------------------------------------------


class TestEvidenceByteOffsets:
    """Regression: byte offsets in Evidence are correct and non-negative."""

    def test_p106_byte_offset_matches_build_opcode_position(self) -> None:
        """The BUILD opcode byte offset must be > 0 (after the GLOBAL opcode)."""
        payload = (
            _P2_HEADER
            + _global_opcode("os", "system")
            + _short_binstring("state")
            + _BUILD
            + _STOP
        )
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(data=payload, artifact_sha256="4" * 64)
        p106 = next(f for f in findings if f.rule_id == "P106")
        assert p106.evidence.byte_offset > 0

    def test_p107_byte_offset_matches_newobj_opcode_position(self) -> None:
        """The NEWOBJ opcode byte offset must be > 0 (after the GLOBAL opcode)."""
        payload = (
            _P2_HEADER
            + _global_opcode("ctypes", "CDLL")
            + _EMPTY_TUPLE
            + _NEWOBJ
            + _STOP
        )
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(data=payload, artifact_sha256="5" * 64)
        p107 = next(f for f in findings if f.rule_id == "P107")
        assert p107.evidence.byte_offset > 0

    def test_p106_archive_member_preserved_when_set(self) -> None:
        """archive_member is forwarded to evidence when analyzing an archive stream."""
        payload = (
            _P2_HEADER
            + _global_opcode("os", "system")
            + _short_binstring("state")
            + _BUILD
            + _STOP
        )
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=payload,
            artifact_sha256="6" * 64,
            coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
            archive_member="archive/data.pkl",
        )
        p106 = next(f for f in findings if f.rule_id == "P106")
        assert p106.evidence.archive_member == "archive/data.pkl"
        assert p106.evidence.coordinate_space == CoordinateSpace.ARCHIVE_MEMBER

    def test_p107_archive_member_preserved_when_set(self) -> None:
        """archive_member is forwarded to P107 evidence when set."""
        payload = (
            _P2_HEADER
            + _global_opcode("ctypes", "CDLL")
            + _EMPTY_TUPLE
            + _NEWOBJ
            + _STOP
        )
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=payload,
            artifact_sha256="7" * 64,
            coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
            archive_member="archive/model.pkl",
        )
        p107 = next(f for f in findings if f.rule_id == "P107")
        assert p107.evidence.archive_member == "archive/model.pkl"
        assert p107.evidence.coordinate_space == CoordinateSpace.ARCHIVE_MEMBER


# ---------------------------------------------------------------------------
# Disassembler-level BUILD / NEWOBJ record verification
# ---------------------------------------------------------------------------


class TestDisassemblerBuildNewObjRecords:
    """Regression: PickleDisassembler populates builds / newobjs lists correctly."""

    def test_disassembler_populates_builds_list(self) -> None:
        payload = (
            _P2_HEADER
            + _global_opcode("os", "system")
            + _short_binstring("state_value")
            + _BUILD
            + _STOP
        )
        disassembler = PickleDisassembler()
        output = disassembler.disassemble(payload)
        assert len(output.builds) == 1
        build_rec = output.builds[0]
        assert build_rec.target_repr == "os.system"
        assert "state_value" in build_rec.state_repr

    def test_disassembler_populates_newobjs_list_for_newobj(self) -> None:
        payload = (
            _P2_HEADER
            + _global_opcode("ctypes", "CDLL")
            + _EMPTY_TUPLE
            + _NEWOBJ
            + _STOP
        )
        disassembler = PickleDisassembler()
        output = disassembler.disassemble(payload)
        assert len(output.newobjs) == 1
        newobj_rec = output.newobjs[0]
        assert newobj_rec.cls_repr == "ctypes.CDLL"
        assert not newobj_rec.is_newobj_ex

    def test_disassembler_populates_newobjs_list_for_newobj_ex(self) -> None:
        payload = (
            b"\x80\x04"
            + _global_opcode("subprocess", "Popen")
            + _EMPTY_TUPLE
            + b"}"  # EMPTY_DICT
            + _NEWOBJ_EX
            + _STOP
        )
        disassembler = PickleDisassembler()
        output = disassembler.disassemble(payload)
        assert len(output.newobjs) == 1
        newobj_rec = output.newobjs[0]
        assert newobj_rec.cls_repr == "subprocess.Popen"
        assert newobj_rec.is_newobj_ex

    def test_disassembler_empty_stack_build_unknown_target(self) -> None:
        """BUILD with empty stack records <unknown_target> as target_repr."""
        payload = _P2_HEADER + _BUILD + _STOP
        disassembler = PickleDisassembler()
        output = disassembler.disassemble(payload)
        assert len(output.builds) == 1
        assert output.builds[0].target_repr == "<unknown_target>"

    def test_disassembler_empty_stack_newobj_unknown_cls(self) -> None:
        """NEWOBJ with empty stack records <unknown_cls> as cls_repr."""
        payload = _P2_HEADER + _NEWOBJ + _STOP
        disassembler = PickleDisassembler()
        output = disassembler.disassemble(payload)
        assert len(output.newobjs) == 1
        assert output.newobjs[0].cls_repr == "<unknown_cls>"


# ---------------------------------------------------------------------------
# 6. Cumulative PyTorch ZIP extraction limit -> P208 finding
# ---------------------------------------------------------------------------


class TestCumulativeZipExtractionLimit:
    """Regression: cumulative extraction limit triggers P208 in PyTorchAnalyzer."""

    _BENIGN_PICKLE = b"\x80\x02}q\x00."

    def test_cumulative_limit_zero_raises_p208(self) -> None:
        """Setting cumulative limit to 0 bytes must produce P208 finding."""
        zip_bytes = _make_zip({"archive/data.pkl": self._BENIGN_PICKLE})
        constrained_reader = PyTorchZipReader(max_cumulative_bytes=0)
        art = _make_artifact(zip_bytes, sha256_char="8")
        analyzer = PyTorchAnalyzer()
        analyzer._zip_reader = constrained_reader
        findings = analyzer.analyze(art)
        rule_ids = [f.rule_id for f in findings]
        assert "P208" in rule_ids, (
            "Expected P208 when cumulative extraction limit is 0 bytes; "
            f"got rule_ids={rule_ids}"
        )

    def test_cumulative_limit_p208_error_code_in_evidence(self) -> None:
        """P208 Evidence.matched_value must contain the error_code from the exception."""
        zip_bytes = _make_zip({"archive/data.pkl": self._BENIGN_PICKLE})
        constrained_reader = PyTorchZipReader(max_cumulative_bytes=0)
        art = _make_artifact(zip_bytes, sha256_char="9")
        analyzer = PyTorchAnalyzer()
        analyzer._zip_reader = constrained_reader
        findings = analyzer.analyze(art)
        p208 = next(f for f in findings if f.rule_id == "P208")
        assert p208.evidence.matched_value == "CUMULATIVE_LIMIT_EXCEEDED"

    def test_cumulative_limit_stops_further_member_analysis(self) -> None:
        """After cumulative limit breach, loop must break (exactly one P208)."""
        zip_bytes = _make_zip(
            {
                "archive/data.pkl": self._BENIGN_PICKLE,
                "archive/constants.pkl": self._BENIGN_PICKLE,
            }
        )
        constrained_reader = PyTorchZipReader(max_cumulative_bytes=0)
        art = _make_artifact(zip_bytes, sha256_char="a")
        analyzer = PyTorchAnalyzer()
        analyzer._zip_reader = constrained_reader
        findings = analyzer.analyze(art)
        p208_findings = [f for f in findings if f.rule_id == "P208"]
        assert len(p208_findings) == 1, (
            f"Expected exactly one P208 after cumulative limit breach; "
            f"got {len(p208_findings)}"
        )

    def test_normal_size_archive_no_p208(self) -> None:
        """Normal-sized archive within default limits produces no P208."""
        zip_bytes = _make_zip({"archive/data.pkl": self._BENIGN_PICKLE})
        art = _make_artifact(zip_bytes, sha256_char="b")
        analyzer = PyTorchAnalyzer()
        findings = analyzer.analyze(art)
        rule_ids = [f.rule_id for f in findings]
        assert "P208" not in rule_ids


# ---------------------------------------------------------------------------
# 7. Explicit torch loader classification overrides broad torch.* safe wildcard
# ---------------------------------------------------------------------------


class TestTorchLoaderClassificationOverride:
    """Regression: torch.jit.load / torch.hub.load / torch.serialization.load
    must be classified SUSPICIOUS even though the broad torch.* wildcard is SAFE."""

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("torch.jit", "load"),
            ("torch.hub", "load"),
            ("torch.serialization", "load"),
        ],
    )
    def test_explicit_torch_loader_is_suspicious_not_safe(
        self, module: str, name: str
    ) -> None:
        """Explicit torch loader entries must override the broad torch.* SAFE wildcard."""
        cat, rationale = classify_global(module=module, name=name)
        assert cat == GlobalCategory.SUSPICIOUS, (
            f"classify_global({module!r}, {name!r}) returned {cat!r}; "
            f"expected SUSPICIOUS (not SAFE via torch.* wildcard). "
            f"Rationale: {rationale}"
        )

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("torch.jit", "load"),
            ("torch.hub", "load"),
            ("torch.serialization", "load"),
        ],
    )
    def test_explicit_torch_loader_in_pickle_stream_emits_p102(
        self, module: str, name: str
    ) -> None:
        """Pickle stream using explicit torch loaders must produce P102 (Suspicious Global)."""
        payload = _P2_HEADER + _global_opcode(module, name) + _STOP
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(data=payload, artifact_sha256="c" * 64)
        rule_ids = [f.rule_id for f in findings]
        assert "P102" in rule_ids, (
            f"Expected P102 for {module}.{name} in Pickle stream; "
            f"got rule_ids={rule_ids}"
        )
        assert "P101" not in rule_ids, (
            f"{module}.{name} must be SUSPICIOUS (P102), not DANGEROUS (P101); "
            f"got rule_ids={rule_ids}"
        )

    def test_broad_torch_namespace_is_safe(self) -> None:
        """Regular torch.* symbols (e.g. torch.optim.Adam) remain SAFE."""
        cat, rationale = classify_global(module="torch.optim", name="Adam")
        assert cat == GlobalCategory.SAFE, (
            f"torch.optim.Adam should be SAFE via the broad torch.* wildcard; "
            f"got {cat!r}. Rationale: {rationale}"
        )

    def test_torch_distributed_is_not_safe(self) -> None:
        """torch.distributed is explicitly excluded from the broad torch.* safe wildcard."""
        cat, rationale = classify_global(module="torch.distributed", name="send")
        assert cat == GlobalCategory.UNKNOWN, (
            f"torch.distributed.send should be UNKNOWN (excluded from safe wildcard); "
            f"got {cat!r}. Rationale: {rationale}"
        )

    def test_torch_jit_load_in_pytorch_zip_emits_p102(self) -> None:
        """torch.jit.load embedded in a PyTorch ZIP member must emit P102."""
        payload = _P2_HEADER + _global_opcode("torch.jit", "load") + _STOP
        zip_bytes = _make_zip({"archive/data.pkl": payload})
        art = _make_artifact(zip_bytes, sha256_char="d")
        analyzer = PyTorchAnalyzer()
        findings = analyzer.analyze(art)
        rule_ids = [f.rule_id for f in findings]
        assert "P102" in rule_ids, (
            "P102 must be emitted for torch.jit.load embedded in PyTorch ZIP member; "
            f"got rule_ids={rule_ids}"
        )


# ---------------------------------------------------------------------------
# 8. Protocol-0 PyTorch member detection consistency
# ---------------------------------------------------------------------------


class TestProtocol0PytorchMemberDetection:
    """Regression: Protocol-0 Pickle PyTorch member heuristics behave consistently."""

    _P0_SAFE_PYTORCH = b"ctorch._utils\n_rebuild_tensor_v2\n."
    _P0_DANGEROUS = b"cos\nsystem\n(S\'id\'\ntR."
    _P0_JIT_LOAD = b"ctorch.jit\nload\n(S\'model.pt\'\ntR."

    def test_p0_safe_pytorch_heuristic_matches(self) -> None:
        """check_pickle_protocol_0_heuristic must match safe Protocol-0 torch payload."""
        matched, reason = check_pickle_protocol_0_heuristic(self._P0_SAFE_PYTORCH)
        assert matched, (
            f"Protocol-0 heuristic must match safe PyTorch stream; reason={reason!r}"
        )

    def test_p0_dangerous_heuristic_matches(self) -> None:
        """check_pickle_protocol_0_heuristic must match Protocol-0 os.system stream."""
        matched, reason = check_pickle_protocol_0_heuristic(self._P0_DANGEROUS)
        assert matched, (
            f"Protocol-0 heuristic must match dangerous os.system stream; reason={reason!r}"
        )

    def test_p0_safe_pytorch_no_danger_findings(self) -> None:
        """PickleAnalyzer must not emit P101/P102/P104 for safe Protocol-0 torch stream."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._P0_SAFE_PYTORCH,
            artifact_sha256="e" * 64,
        )
        rule_ids = [f.rule_id for f in findings]
        assert "P101" not in rule_ids
        assert "P102" not in rule_ids
        assert "P104" not in rule_ids

    def test_p0_dangerous_emits_p101_and_p104(self) -> None:
        """PickleAnalyzer must emit P101 + P104 for Protocol-0 os.system stream."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._P0_DANGEROUS,
            artifact_sha256="f" * 64,
        )
        rule_ids = [f.rule_id for f in findings]
        assert "P101" in rule_ids, "P101 expected for Protocol-0 os.system GLOBAL"
        assert "P104" in rule_ids, "P104 expected for Protocol-0 REDUCE of os.system"

    def test_p0_jit_load_emits_p102(self) -> None:
        """Protocol-0 torch.jit.load must emit P102 (Suspicious), not P101 (Dangerous)."""
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=self._P0_JIT_LOAD,
            artifact_sha256="0" * 64,
        )
        rule_ids = [f.rule_id for f in findings]
        assert "P102" in rule_ids, (
            "P102 must be emitted for Protocol-0 torch.jit.load; "
            f"got rule_ids={rule_ids}"
        )

    def test_p0_member_detected_in_pytorch_zip(self) -> None:
        """PyTorchAnalyzer must detect Protocol-0 Pickle members inside ZIP archives.

        Uses a short payload so the full GLOBAL opcode structure fits in the
        32-byte heuristic head window that PyTorchAnalyzer reads for non-candidate
        members (archive/weights has no .pkl extension, triggering the fallback path).
        """
        # ctorch\nFloatStorage\n. = 20 bytes; fits within the 32-byte head slice.
        p0_short = b"ctorch\nFloatStorage\n."
        zip_bytes = _make_zip({"archive/weights": p0_short})
        art = _make_artifact(zip_bytes, sha256_char="1")
        analyzer = PyTorchAnalyzer()
        findings = analyzer.analyze(art)
        rule_ids = [f.rule_id for f in findings]
        assert "P209" in rule_ids, (
            "PyTorchAnalyzer must detect Protocol-0 Pickle member via heuristic; "
            f"got rule_ids={rule_ids}"
        )

    def test_p0_dangerous_in_zip_member_emits_p101(self) -> None:
        """Dangerous Protocol-0 payload inside a ZIP member must emit P101."""
        zip_bytes = _make_zip({"archive/data.pkl": self._P0_DANGEROUS})
        art = _make_artifact(zip_bytes, sha256_char="2")
        analyzer = PyTorchAnalyzer()
        findings = analyzer.analyze(art)
        rule_ids = [f.rule_id for f in findings]
        assert "P101" in rule_ids, (
            "P101 must propagate from dangerous Protocol-0 ZIP member; "
            f"got rule_ids={rule_ids}"
        )

    def test_p0_consistent_detection_across_multiple_members(self) -> None:
        """Both a safe and a dangerous Protocol-0 member produce consistent findings."""
        zip_bytes = _make_zip(
            {
                "archive/data.pkl": self._P0_SAFE_PYTORCH,
                "archive/extra.pkl": self._P0_DANGEROUS,
            }
        )
        art = _make_artifact(zip_bytes, sha256_char="3")
        analyzer = PyTorchAnalyzer()
        findings = analyzer.analyze(art)
        rule_ids = [f.rule_id for f in findings]

        p209_count = sum(1 for f in findings if f.rule_id == "P209")
        assert p209_count >= 1, "At least one P209 expected for embedded .pkl members"

        assert "P101" in rule_ids, (
            "P101 must come from the dangerous Protocol-0 ZIP member"
        )

        p101_members = {
            f.evidence.archive_member
            for f in findings
            if f.rule_id == "P101"
        }
        assert "archive/extra.pkl" in p101_members
