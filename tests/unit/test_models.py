"""
tests/unit/test_models.py
=========================

Unit tests for model_scanner.core.models and model_scanner.core.exceptions.

Phase 2 — Data Models & Error Hierarchy.

Coverage targets:
  - Valid construction of every dataclass and enum
  - Required-field and type enforcement
  - Enum behaviour (value, membership, iteration)
  - Frozen/immutable behaviour
  - Validation rules per implementation_design.md Sections 4-7 and 21
  - Risk score bounds and band alignment
  - Evidence offset validation
  - ScanWarning / ScanError DTO validation
  - Full exception hierarchy construction and inheritance
  - Exception string representations

Does NOT test:
  - Scanner logic / analyzers (Phase 4-6)
  - Rule engine (Phase 7)
  - Severity engine (Phase 8)
  - CLI / API (Phase 10)
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from model_scanner.core.exceptions import (
    AcquisitionError,
    AnalysisError,
    ConfigurationError,
    FormatDetectionError,
    GGUFAnalysisError,
    InputError,
    PickleAnalysisError,
    ReportingError,
    ResourceLimitError,
    SafeTensorsAnalysisError,
    ScannerException,
    SecurityBoundaryViolationError,
)
from model_scanner.core.models import (
    Artifact,
    ArtifactScanResult,
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    FormatDetectionResult,
    InputSourceType,
    ModelFormat,
    RiskBand,
    RiskScore,
    ScanConfig,
    ScanError,
    ScanOutcome,
    ScanRequest,
    ScanResult,
    ScanSummary,
    ScanWarning,
    Severity,
    risk_band_for_score,
)

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

_ZERO_SHA256 = "a" * 64
_VALID_UUID = "12345678-1234-5678-1234-567812345678"


def _minimal_evidence(
    *,
    coordinate_space: CoordinateSpace = CoordinateSpace.ARTIFACT,
    archive_member: str | None = None,
    byte_offset: int | None = 0,
    matched_value: str = "os.system",
) -> Evidence:
    return Evidence(
        coordinate_space=coordinate_space,
        byte_offset=byte_offset,
        byte_length=None,
        archive_member=archive_member,
        matched_value=matched_value,
    )


def _minimal_risk_score(final_score: int = 0) -> RiskScore:
    return RiskScore(
        raw_score=float(final_score),
        final_score=final_score,
        risk_band=risk_band_for_score(final_score),
        format_multiplier=1.0,
        obfuscation_multiplier=1.0,
        findings_weighted_sum=0.0,
    )


def _minimal_finding(rule_id: str = "P101") -> Finding:
    return Finding(
        finding_id=_VALID_UUID,
        rule_id=rule_id,
        rule_name="Dangerous Global Import",
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        category="code_execution",
        format=ModelFormat.RAW_PICKLE,
        artifact_sha256=_ZERO_SHA256,
        evidence=_minimal_evidence(),
        description="GLOBAL opcode imports os.system.",
        remediation="Do not deserialize this artifact.",
    )


def _minimal_artifact_scan_result(score: int = 0) -> ArtifactScanResult:
    return ArtifactScanResult(
        artifact_id=_VALID_UUID,
        name="model.pkl",
        source_origin="/data/model.pkl",
        input_source=InputSourceType.LOCAL_FILE,
        detected_format=ModelFormat.RAW_PICKLE,
        size_bytes=1024,
        sha256=_ZERO_SHA256,
        scan_outcome=ScanOutcome.COMPLETE,
        risk_score=_minimal_risk_score(score),
        findings=(),
        warnings=(),
        errors=(),
        duration_ms=42,
    )


def _minimal_scan_summary(
    exit_code: int = 0,
    overall: ScanOutcome = ScanOutcome.COMPLETE,
) -> ScanSummary:
    return ScanSummary(
        total_artifacts=1,
        scanned_successfully=1,
        failed_artifacts=0,
        unsupported_artifacts=0,
        findings_by_severity={
            "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0
        },
        artifacts_by_risk_band={"CLEAN": 1},
        max_risk_score=0,
        overall_outcome=overall,
        exit_code=exit_code,
    )


# ===========================================================================
# SECTION 1: Enumerations
# ===========================================================================


class TestInputSourceType:
    def test_values(self) -> None:
        assert InputSourceType.LOCAL_FILE.value == "LOCAL_FILE"
        assert InputSourceType.LOCAL_DIRECTORY.value == "LOCAL_DIRECTORY"
        assert InputSourceType.PUBLIC_URL.value == "PUBLIC_URL"
        assert InputSourceType.IN_MEMORY.value == "IN_MEMORY"

    def test_membership(self) -> None:
        assert InputSourceType("LOCAL_FILE") == InputSourceType.LOCAL_FILE

    def test_count(self) -> None:
        assert len(list(InputSourceType)) == 4


class TestModelFormat:
    def test_all_values(self) -> None:
        expected = {
            # Analyzed formats
            "RAW_PICKLE", "LEGACY_PYTORCH", "PYTORCH_ZIP", "SAFETENSORS", "GGUF",
            "GGML", "UNSUPPORTED",
            # Identified-but-unanalyzed formats (A5)
            "ONNX", "HDF5", "NUMPY_NPY", "NUMPY_NPZ", "TFLITE", "FLAX_MSGPACK",
            # Virtual format for repository/supply-chain findings (B1/B4)
            "REPOSITORY",
            # P2 new formats
            "TF_SAVEDMODEL", "KERAS",
            # WP-B new formats
            "MINDSPORE_CKPT",
            # TF.js weight shards — raw float32 buffer, no analyzer
            "TFJS_WEIGHT_SHARD",
        }
        assert {f.value for f in ModelFormat} == expected

    def test_unsupported(self) -> None:
        assert ModelFormat.UNSUPPORTED.value == "UNSUPPORTED"


class TestScanOutcome:
    def test_values(self) -> None:
        assert ScanOutcome.COMPLETE.value == "COMPLETE"
        assert ScanOutcome.ERROR.value == "ERROR"
        assert ScanOutcome.WARNING.value == "WARNING"
        assert ScanOutcome.UNSUPPORTED.value == "UNSUPPORTED"


class TestSeverity:
    def test_five_levels(self) -> None:
        expected = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}
        assert {s.value for s in Severity} == expected

    def test_ordering_by_value(self) -> None:
        # Verify all five members are distinct
        levels = list(Severity)
        assert len(levels) == 5
        assert len({s.value for s in levels}) == 5


class TestConfidence:
    def test_three_levels(self) -> None:
        assert {c.value for c in Confidence} == {"HIGH", "MEDIUM", "LOW"}


class TestRiskBand:
    def test_bands(self) -> None:
        expected = {
            "CLEAN", "UNANALYZED",
            "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK", "CRITICAL_RISK",
        }
        assert {b.value for b in RiskBand} == expected

    def test_unanalyzed_is_distinct_from_clean(self) -> None:
        """UNANALYZED must never collapse into CLEAN.

        CLEAN asserts analysis ran and found nothing. UNANALYZED asserts only
        that nothing was inspected, so the absence of findings proves nothing.
        """
        assert RiskBand.UNANALYZED != RiskBand.CLEAN


class TestCoordinateSpace:
    def test_three_spaces(self) -> None:
        assert CoordinateSpace.ARTIFACT.value == "artifact"
        assert CoordinateSpace.ARCHIVE_MEMBER.value == "archive_member"
        assert CoordinateSpace.HEADER.value == "header"


# ===========================================================================
# SECTION 2: risk_band_for_score helper
# ===========================================================================


class TestRiskBandForScore:
    def test_score_zero_is_clean(self) -> None:
        assert risk_band_for_score(0) == RiskBand.CLEAN

    def test_low_risk_boundaries(self) -> None:
        assert risk_band_for_score(1) == RiskBand.LOW_RISK
        assert risk_band_for_score(15) == RiskBand.LOW_RISK
        assert risk_band_for_score(30) == RiskBand.LOW_RISK

    def test_medium_risk_boundaries(self) -> None:
        assert risk_band_for_score(31) == RiskBand.MEDIUM_RISK
        assert risk_band_for_score(45) == RiskBand.MEDIUM_RISK
        assert risk_band_for_score(60) == RiskBand.MEDIUM_RISK

    def test_high_risk_boundaries(self) -> None:
        assert risk_band_for_score(61) == RiskBand.HIGH_RISK
        assert risk_band_for_score(75) == RiskBand.HIGH_RISK
        assert risk_band_for_score(89) == RiskBand.HIGH_RISK

    def test_critical_risk_boundaries(self) -> None:
        assert risk_band_for_score(90) == RiskBand.CRITICAL_RISK
        assert risk_band_for_score(95) == RiskBand.CRITICAL_RISK
        assert risk_band_for_score(100) == RiskBand.CRITICAL_RISK


# ===========================================================================
# SECTION 3: ScanConfig
# ===========================================================================


class TestScanConfig:
    def test_default_construction(self) -> None:
        cfg = ScanConfig()
        assert cfg.strict_mode is False
        assert cfg.max_safetensors_header_mb == 16
        assert cfg.max_json_tokens == 100_000
        assert cfg.max_json_depth == 128
        assert cfg.max_gguf_kv_count == 10_000
        assert cfg.max_gguf_value_bytes == 1_048_576
        assert cfg.max_download_repo_bytes == 2 * 1024 * 1024 * 1024
        assert cfg.max_download_file_bytes == 1 * 1024 * 1024 * 1024
        assert cfg.download_timeout_sec == 60
        assert cfg.allowlist_modules == ()
        assert cfg.blocklist_modules == ()
        assert cfg.suppressed_rule_ids == ()
        assert cfg.include_pattern is None
        assert cfg.exclude_pattern is None

    def test_custom_values(self) -> None:
        cfg = ScanConfig(
            strict_mode=True,
            max_safetensors_header_mb=32,
            max_json_tokens=50_000,
            allowlist_modules=("torch", "torch.nn"),
        )
        assert cfg.strict_mode is True
        assert cfg.max_safetensors_header_mb == 32
        assert cfg.allowlist_modules == ("torch", "torch.nn")

    def test_frozen_immutable(self) -> None:
        cfg = ScanConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.strict_mode = True  # type: ignore[misc]

    def test_max_safetensors_header_mb_zero_invalid(self) -> None:
        with pytest.raises(ValueError, match="max_safetensors_header_mb must be positive"):
            ScanConfig(max_safetensors_header_mb=0)

    def test_max_safetensors_header_mb_exceeds_128(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed 128 MB"):
            ScanConfig(max_safetensors_header_mb=129)

    def test_max_json_tokens_zero_invalid(self) -> None:
        with pytest.raises(ValueError, match="max_json_tokens must be positive"):
            ScanConfig(max_json_tokens=0)

    def test_max_json_tokens_exceeds_million(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed 1,000,000"):
            ScanConfig(max_json_tokens=1_000_001)

    def test_negative_gguf_kv_count_invalid(self) -> None:
        with pytest.raises(ValueError, match="max_gguf_kv_count must be positive"):
            ScanConfig(max_gguf_kv_count=-1)

    def test_negative_download_timeout_invalid(self) -> None:
        with pytest.raises(ValueError, match="download_timeout_sec must be positive"):
            ScanConfig(download_timeout_sec=0)

    def test_max_safetensors_128_valid(self) -> None:
        cfg = ScanConfig(max_safetensors_header_mb=128)
        assert cfg.max_safetensors_header_mb == 128

    def test_max_json_tokens_million_valid(self) -> None:
        cfg = ScanConfig(max_json_tokens=1_000_000)
        assert cfg.max_json_tokens == 1_000_000


# ===========================================================================
# SECTION 4: ScanRequest
# ===========================================================================


class TestScanRequest:
    def test_local_file_path_string(self) -> None:
        req = ScanRequest(
            target="/data/model.pkl",
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        assert req.target == "/data/model.pkl"
        assert req.strict is False
        assert req.config is None
        assert req.filename_hint is None

    def test_local_file_path_object(self) -> None:
        p = Path("/data/model.pkl")
        req = ScanRequest(target=p, input_source_type=InputSourceType.LOCAL_FILE)
        assert req.target == p

    def test_in_memory_bytes(self) -> None:
        raw = b"\x80\x04\x95"
        req = ScanRequest(
            target=raw,
            input_source_type=InputSourceType.IN_MEMORY,
            filename_hint="model.pkl",
        )
        assert req.target == raw
        assert req.filename_hint == "model.pkl"

    def test_in_memory_bytesio(self) -> None:
        buf = io.BytesIO(b"\x80\x04\x95")
        req = ScanRequest(target=buf, input_source_type=InputSourceType.IN_MEMORY)
        assert req.target is buf

    def test_public_url_valid(self) -> None:
        req = ScanRequest(
            target="https://huggingface.co/org/model",
            input_source_type=InputSourceType.PUBLIC_URL,
        )
        assert req.target == "https://huggingface.co/org/model"

    def test_public_url_http_valid(self) -> None:
        req = ScanRequest(
            target="http://example.com/model.safetensors",
            input_source_type=InputSourceType.PUBLIC_URL,
        )
        assert req.target.startswith("http://")  # type: ignore[union-attr]

    def test_empty_string_target_invalid(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            ScanRequest(target="", input_source_type=InputSourceType.LOCAL_FILE)

    def test_empty_bytes_target_invalid(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            ScanRequest(target=b"", input_source_type=InputSourceType.IN_MEMORY)

    def test_none_target_invalid(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            ScanRequest(
                target=None,  # type: ignore[arg-type]
                input_source_type=InputSourceType.LOCAL_FILE,
            )

    def test_public_url_requires_string(self) -> None:
        with pytest.raises(ValueError, match="must be a str URL"):
            ScanRequest(
                target=Path("/data/model.pkl"),
                input_source_type=InputSourceType.PUBLIC_URL,
            )

    def test_public_url_bad_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="http:// or https://"):
            ScanRequest(
                target="ftp://example.com/model.pkl",
                input_source_type=InputSourceType.PUBLIC_URL,
            )

    def test_frozen_immutable(self) -> None:
        req = ScanRequest(
            target="/data/model.pkl",
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        with pytest.raises((AttributeError, TypeError)):
            req.strict = True  # type: ignore[misc]


# ===========================================================================
# SECTION 5: Artifact
# ===========================================================================


class TestArtifact:
    def test_valid_local_path(self) -> None:
        art = Artifact(
            artifact_id=_VALID_UUID,
            source_path="/data/model.pkl",
            input_source=InputSourceType.LOCAL_FILE,
            size_bytes=2048,
            sha256=_ZERO_SHA256,
            local_path=Path("/data/model.pkl"),
        )
        assert art.size_bytes == 2048
        assert art.sha256 == _ZERO_SHA256
        assert art.stream is None

    def test_valid_stream(self) -> None:
        buf = io.BytesIO(b"\x00" * 64)
        art = Artifact(
            artifact_id=_VALID_UUID,
            source_path="<memory>",
            input_source=InputSourceType.IN_MEMORY,
            size_bytes=64,
            sha256=_ZERO_SHA256,
            stream=buf,
        )
        assert art.local_path is None

    def test_sha256_must_be_64_chars(self) -> None:
        with pytest.raises(ValueError, match="64 lowercase hex"):
            Artifact(
                artifact_id=_VALID_UUID,
                source_path="/data/model.pkl",
                input_source=InputSourceType.LOCAL_FILE,
                size_bytes=0,
                sha256="abc123",
                local_path=Path("/data/model.pkl"),
            )

    def test_sha256_uppercase_rejected(self) -> None:
        with pytest.raises(ValueError, match="64 lowercase hex"):
            Artifact(
                artifact_id=_VALID_UUID,
                source_path="/data/model.pkl",
                input_source=InputSourceType.LOCAL_FILE,
                size_bytes=0,
                sha256="A" * 64,
                local_path=Path("/data/model.pkl"),
            )

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="size_bytes must be >= 0"):
            Artifact(
                artifact_id=_VALID_UUID,
                source_path="/data/model.pkl",
                input_source=InputSourceType.LOCAL_FILE,
                size_bytes=-1,
                sha256=_ZERO_SHA256,
                local_path=Path("/data/model.pkl"),
            )

    def test_both_path_and_stream_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            Artifact(
                artifact_id=_VALID_UUID,
                source_path="/data/model.pkl",
                input_source=InputSourceType.LOCAL_FILE,
                size_bytes=0,
                sha256=_ZERO_SHA256,
                local_path=Path("/data/model.pkl"),
                stream=io.BytesIO(b""),
            )

    def test_neither_path_nor_stream_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            Artifact(
                artifact_id=_VALID_UUID,
                source_path="/data/model.pkl",
                input_source=InputSourceType.LOCAL_FILE,
                size_bytes=0,
                sha256=_ZERO_SHA256,
            )

    def test_zero_size_valid(self) -> None:
        art = Artifact(
            artifact_id=_VALID_UUID,
            source_path="/data/empty.pkl",
            input_source=InputSourceType.LOCAL_FILE,
            size_bytes=0,
            sha256=_ZERO_SHA256,
            local_path=Path("/data/empty.pkl"),
        )
        assert art.size_bytes == 0

    def test_frozen_immutable(self) -> None:
        art = Artifact(
            artifact_id=_VALID_UUID,
            source_path="/data/model.pkl",
            input_source=InputSourceType.LOCAL_FILE,
            size_bytes=0,
            sha256=_ZERO_SHA256,
            local_path=Path("/data/model.pkl"),
        )
        with pytest.raises((AttributeError, TypeError)):
            art.size_bytes = 999  # type: ignore[misc]


# ===========================================================================
# SECTION 6: FormatDetectionResult
# ===========================================================================


class TestFormatDetectionResult:
    def test_valid_construction(self) -> None:
        fdr = FormatDetectionResult(
            detected_format=ModelFormat.RAW_PICKLE,
            magic_matched=True,
            extension_matched=True,
            format_mismatch=False,
            details="Pickle protocol header 0x80 0x05 detected",
        )
        assert fdr.detected_format == ModelFormat.RAW_PICKLE
        assert fdr.format_mismatch is False

    def test_mismatch_flag(self) -> None:
        fdr = FormatDetectionResult(
            detected_format=ModelFormat.PYTORCH_ZIP,
            magic_matched=True,
            extension_matched=False,
            format_mismatch=True,
            details="ZIP magic but .safetensors extension",
        )
        assert fdr.format_mismatch is True

    def test_frozen_immutable(self) -> None:
        fdr = FormatDetectionResult(
            detected_format=ModelFormat.UNSUPPORTED,
            magic_matched=False,
            extension_matched=False,
            format_mismatch=False,
            details="No match",
        )
        with pytest.raises((AttributeError, TypeError)):
            fdr.detected_format = ModelFormat.GGUF  # type: ignore[misc]


# ===========================================================================
# SECTION 7: Evidence
# ===========================================================================


class TestEvidence:
    def test_artifact_space_valid(self) -> None:
        ev = Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=128,
            byte_length=16,
            archive_member=None,
            matched_value="os.system",
        )
        assert ev.byte_offset == 128
        assert ev.archive_member is None

    def test_archive_member_space_requires_member_name(self) -> None:
        with pytest.raises(ValueError, match="archive_member must be set"):
            Evidence(
                coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                byte_offset=0,
                byte_length=None,
                archive_member=None,
                matched_value="os.system",
            )

    def test_archive_member_valid(self) -> None:
        ev = Evidence(
            coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
            byte_offset=64,
            byte_length=8,
            archive_member="archive/data.pkl",
            matched_value="posix.system",
            opcode="GLOBAL",
        )
        assert ev.archive_member == "archive/data.pkl"
        assert ev.opcode == "GLOBAL"

    def test_header_space_no_archive_member_required(self) -> None:
        ev = Evidence(
            coordinate_space=CoordinateSpace.HEADER,
            byte_offset=8,
            byte_length=None,
            archive_member=None,
            matched_value="eval(",
        )
        assert ev.coordinate_space == CoordinateSpace.HEADER

    def test_negative_byte_offset_rejected(self) -> None:
        with pytest.raises(ValueError, match="byte_offset must be >= 0"):
            Evidence(
                coordinate_space=CoordinateSpace.ARTIFACT,
                byte_offset=-1,
                byte_length=None,
                archive_member=None,
                matched_value="os.system",
            )

    def test_negative_byte_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="byte_length must be >= 0"):
            Evidence(
                coordinate_space=CoordinateSpace.ARTIFACT,
                byte_offset=0,
                byte_length=-1,
                archive_member=None,
                matched_value="os.system",
            )

    def test_none_offsets_valid(self) -> None:
        ev = Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=None,
            byte_length=None,
            archive_member=None,
            matched_value="eval(",
        )
        assert ev.byte_offset is None
        assert ev.byte_length is None

    def test_raw_hex_and_context_optional(self) -> None:
        ev = Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=0,
            byte_length=4,
            archive_member=None,
            matched_value="os.system",
            raw_hex="80 05 95",
            context_snippet="…GLOBAL os system…",
        )
        assert ev.raw_hex == "80 05 95"
        assert ev.context_snippet == "…GLOBAL os system…"

    def test_frozen_immutable(self) -> None:
        ev = _minimal_evidence()
        with pytest.raises((AttributeError, TypeError)):
            ev.byte_offset = 999  # type: ignore[misc]


# ===========================================================================
# SECTION 8: RiskScore
# ===========================================================================


class TestRiskScore:
    def test_clean_score(self) -> None:
        rs = RiskScore.clean()
        assert rs.final_score == 0
        assert rs.risk_band == RiskBand.CLEAN
        assert rs.raw_score == 0.0

    def test_valid_scores_across_bands(self) -> None:
        for score, expected_band in [
            (0, RiskBand.CLEAN),
            (1, RiskBand.LOW_RISK),
            (30, RiskBand.LOW_RISK),
            (31, RiskBand.MEDIUM_RISK),
            (60, RiskBand.MEDIUM_RISK),
            (61, RiskBand.HIGH_RISK),
            (89, RiskBand.HIGH_RISK),
            (90, RiskBand.CRITICAL_RISK),
            (100, RiskBand.CRITICAL_RISK),
        ]:
            rs = RiskScore(
                raw_score=float(score),
                final_score=score,
                risk_band=expected_band,
                format_multiplier=1.0,
                obfuscation_multiplier=1.0,
                findings_weighted_sum=0.0,
            )
            assert rs.final_score == score
            assert rs.risk_band == expected_band

    def test_score_below_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be in \\[0, 100\\]"):
            RiskScore(
                raw_score=-1.0,
                final_score=-1,
                risk_band=RiskBand.CLEAN,
                format_multiplier=1.0,
                obfuscation_multiplier=1.0,
                findings_weighted_sum=0.0,
            )

    def test_score_above_100_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be in \\[0, 100\\]"):
            RiskScore(
                raw_score=101.0,
                final_score=101,
                risk_band=RiskBand.CRITICAL_RISK,
                format_multiplier=1.0,
                obfuscation_multiplier=1.0,
                findings_weighted_sum=0.0,
            )

    def test_mismatched_band_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match the expected band"):
            RiskScore(
                raw_score=50.0,
                final_score=50,
                risk_band=RiskBand.CLEAN,  # Wrong — 50 should be MEDIUM_RISK
                format_multiplier=1.0,
                obfuscation_multiplier=1.0,
                findings_weighted_sum=0.0,
            )

    def test_negative_raw_score_rejected(self) -> None:
        with pytest.raises(ValueError, match="raw_score must be >= 0"):
            RiskScore(
                raw_score=-0.5,
                final_score=0,
                risk_band=RiskBand.CLEAN,
                format_multiplier=1.0,
                obfuscation_multiplier=1.0,
                findings_weighted_sum=0.0,
            )

    def test_negative_multiplier_rejected(self) -> None:
        with pytest.raises(ValueError, match="format_multiplier must be >= 0"):
            RiskScore(
                raw_score=0.0,
                final_score=0,
                risk_band=RiskBand.CLEAN,
                format_multiplier=-0.1,
                obfuscation_multiplier=1.0,
                findings_weighted_sum=0.0,
            )

    def test_frozen_immutable(self) -> None:
        rs = RiskScore.clean()
        with pytest.raises((AttributeError, TypeError)):
            rs.final_score = 50  # type: ignore[misc]


# ===========================================================================
# SECTION 9: Finding
# ===========================================================================


class TestFinding:
    def test_valid_finding(self) -> None:
        f = _minimal_finding()
        assert f.rule_id == "P101"
        assert f.severity == Severity.CRITICAL
        assert f.confidence == Confidence.HIGH
        assert f.cve_ids == ()

    def test_valid_rule_ids(self) -> None:
        valid_ids = [
            "P101", "P102", "P201", "P202",
            "ST101", "ST201",
            "GU101", "GU201",
            "G001",
        ]
        for rule_id in valid_ids:
            f = _minimal_finding(rule_id=rule_id)
            assert f.rule_id == rule_id

    def test_invalid_rule_id_rejected(self) -> None:
        invalid_ids = [
            "X101",    # Invalid namespace prefix
            "P1",      # Too short — missing 2-digit suffix
            "P1011",   # Too long
            "p101",    # Lowercase
            "P 101",   # Space
            "",        # Empty
            "G0001",   # Three-digit suffix
        ]
        for bad_id in invalid_ids:
            with pytest.raises(ValueError, match="rule_id"):
                _minimal_finding(rule_id=bad_id)

    def test_empty_remediation_rejected(self) -> None:
        ev = _minimal_evidence()
        with pytest.raises(ValueError, match="remediation must not be empty"):
            Finding(
                finding_id=_VALID_UUID,
                rule_id="P101",
                rule_name="Test",
                severity=Severity.HIGH,
                confidence=Confidence.HIGH,
                category="code_execution",
                format=ModelFormat.RAW_PICKLE,
                artifact_sha256=_ZERO_SHA256,
                evidence=ev,
                description="Test finding",
                remediation="",
            )

    def test_invalid_sha256_rejected(self) -> None:
        ev = _minimal_evidence()
        with pytest.raises(ValueError, match="artifact_sha256"):
            Finding(
                finding_id=_VALID_UUID,
                rule_id="P101",
                rule_name="Test",
                severity=Severity.HIGH,
                confidence=Confidence.HIGH,
                category="code_execution",
                format=ModelFormat.RAW_PICKLE,
                artifact_sha256="DEADBEEF",
                evidence=ev,
                description="Test",
                remediation="Fix it.",
            )

    def test_cve_ids_tuple(self) -> None:
        ev = _minimal_evidence()
        f = Finding(
            finding_id=_VALID_UUID,
            rule_id="P101",
            rule_name="Test",
            severity=Severity.CRITICAL,
            confidence=Confidence.HIGH,
            category="code_execution",
            format=ModelFormat.RAW_PICKLE,
            artifact_sha256=_ZERO_SHA256,
            evidence=ev,
            description="Test",
            remediation="Fix it.",
            cve_ids=("TEST-ID-1", "TEST-ID-2"),
        )
        assert f.cve_ids == ("TEST-ID-1", "TEST-ID-2")

    def test_frozen_immutable(self) -> None:
        f = _minimal_finding()
        with pytest.raises((AttributeError, TypeError)):
            f.severity = Severity.LOW  # type: ignore[misc]


# ===========================================================================
# SECTION 10: ScanWarning and ScanError DTOs
# ===========================================================================


class TestScanWarning:
    def test_valid_construction(self) -> None:
        w = ScanWarning(warning_code="FMT_WARN", message="Format ambiguity detected")
        assert w.warning_code == "FMT_WARN"
        assert w.context is None

    def test_with_context(self) -> None:
        w = ScanWarning(
            warning_code="UNSUPPORTED_MEMBER",
            message="Skipped archive member",
            context="model.pt::extra/data.pt",
        )
        assert w.context == "model.pt::extra/data.pt"

    def test_empty_code_rejected(self) -> None:
        with pytest.raises(ValueError, match="warning_code must not be empty"):
            ScanWarning(warning_code="", message="Something happened")

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(ValueError, match="message must not be empty"):
            ScanWarning(warning_code="W001", message="")

    def test_frozen_immutable(self) -> None:
        w = ScanWarning(warning_code="W001", message="Test")
        with pytest.raises((AttributeError, TypeError)):
            w.message = "Mutated"  # type: ignore[misc]


class TestScanError:
    def test_valid_construction(self) -> None:
        e = ScanError(
            error_code="ACQ_TIMEOUT",
            message="HTTP request timed out",
            exception_type="AcquisitionError",
        )
        assert e.error_code == "ACQ_TIMEOUT"
        assert e.traceback_summary is None

    def test_with_traceback_summary(self) -> None:
        e = ScanError(
            error_code="PICKLE_ERROR",
            message="Malformed opcode",
            exception_type="PickleAnalysisError",
            traceback_summary="File 'analyzer.py', line 42",
        )
        assert e.traceback_summary == "File 'analyzer.py', line 42"

    def test_empty_code_rejected(self) -> None:
        with pytest.raises(ValueError, match="error_code must not be empty"):
            ScanError(error_code="", message="Oops", exception_type="ScannerError")

    def test_empty_exception_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="exception_type must not be empty"):
            ScanError(error_code="E001", message="Oops", exception_type="")

    def test_frozen_immutable(self) -> None:
        e = ScanError(
            error_code="E001", message="Test", exception_type="AnalysisError"
        )
        with pytest.raises((AttributeError, TypeError)):
            e.message = "Mutated"  # type: ignore[misc]


# ===========================================================================
# SECTION 11: ArtifactScanResult
# ===========================================================================


class TestArtifactScanResult:
    def test_valid_construction(self) -> None:
        asr = _minimal_artifact_scan_result()
        assert asr.scan_outcome == ScanOutcome.COMPLETE
        assert asr.findings == ()
        assert asr.duration_ms == 42

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="size_bytes must be >= 0"):
            ArtifactScanResult(
                artifact_id=_VALID_UUID,
                name="model.pkl",
                source_origin="/data/model.pkl",
                input_source=InputSourceType.LOCAL_FILE,
                detected_format=ModelFormat.RAW_PICKLE,
                size_bytes=-1,
                sha256=_ZERO_SHA256,
                scan_outcome=ScanOutcome.COMPLETE,
                risk_score=RiskScore.clean(),
                findings=(),
                warnings=(),
                errors=(),
                duration_ms=0,
            )

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="duration_ms must be >= 0"):
            ArtifactScanResult(
                artifact_id=_VALID_UUID,
                name="model.pkl",
                source_origin="/data/model.pkl",
                input_source=InputSourceType.LOCAL_FILE,
                detected_format=ModelFormat.RAW_PICKLE,
                size_bytes=0,
                sha256=_ZERO_SHA256,
                scan_outcome=ScanOutcome.COMPLETE,
                risk_score=RiskScore.clean(),
                findings=(),
                warnings=(),
                errors=(),
                duration_ms=-1,
            )

    def test_with_findings(self) -> None:
        f = _minimal_finding()
        asr = ArtifactScanResult(
            artifact_id=_VALID_UUID,
            name="model.pkl",
            source_origin="/data/model.pkl",
            input_source=InputSourceType.LOCAL_FILE,
            detected_format=ModelFormat.RAW_PICKLE,
            size_bytes=1024,
            sha256=_ZERO_SHA256,
            scan_outcome=ScanOutcome.COMPLETE,
            risk_score=_minimal_risk_score(95),
            findings=(f,),
            warnings=(),
            errors=(),
            duration_ms=100,
        )
        assert len(asr.findings) == 1
        assert asr.findings[0].rule_id == "P101"

    def test_frozen_immutable(self) -> None:
        asr = _minimal_artifact_scan_result()
        with pytest.raises((AttributeError, TypeError)):
            asr.duration_ms = 999  # type: ignore[misc]


# ===========================================================================
# SECTION 12: ScanSummary
# ===========================================================================


class TestScanSummary:
    def test_valid_construction(self) -> None:
        ss = _minimal_scan_summary()
        assert ss.exit_code == 0
        assert ss.max_risk_score == 0

    def test_invalid_exit_code_rejected(self) -> None:
        with pytest.raises(ValueError, match="exit_code must be 0, 1, or 2"):
            ScanSummary(
                total_artifacts=1,
                scanned_successfully=1,
                failed_artifacts=0,
                unsupported_artifacts=0,
                findings_by_severity={},
                artifacts_by_risk_band={},
                max_risk_score=0,
                overall_outcome=ScanOutcome.COMPLETE,
                exit_code=3,
            )

    def test_max_risk_score_out_of_bounds(self) -> None:
        with pytest.raises(ValueError, match="max_risk_score must be in \\[0, 100\\]"):
            ScanSummary(
                total_artifacts=1,
                scanned_successfully=1,
                failed_artifacts=0,
                unsupported_artifacts=0,
                findings_by_severity={},
                artifacts_by_risk_band={},
                max_risk_score=101,
                overall_outcome=ScanOutcome.COMPLETE,
                exit_code=0,
            )

    def test_all_exit_codes_valid(self) -> None:
        for code in (0, 1, 2):
            ss = ScanSummary(
                total_artifacts=1,
                scanned_successfully=1,
                failed_artifacts=0,
                unsupported_artifacts=0,
                findings_by_severity={},
                artifacts_by_risk_band={},
                max_risk_score=0,
                overall_outcome=ScanOutcome.COMPLETE,
                exit_code=code,
            )
            assert ss.exit_code == code


# ===========================================================================
# SECTION 13: ScanResult
# ===========================================================================


class TestScanResult:
    def test_valid_construction(self) -> None:
        asr = _minimal_artifact_scan_result()
        ss = _minimal_scan_summary()
        sr = ScanResult(
            scan_id=_VALID_UUID,
            scanner_version="2.0.0",
            timestamp_utc="2026-08-16T17:00:00Z",
            target_request="/data/model.pkl",
            strict_mode=False,
            artifacts=(asr,),
            summary=ss,
            total_duration_ms=50,
        )
        assert sr.scanner_version == "2.0.0"
        assert len(sr.artifacts) == 1

    def test_negative_total_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_duration_ms must be >= 0"):
            ScanResult(
                scan_id=_VALID_UUID,
                scanner_version="2.0.0",
                timestamp_utc="2026-08-16T17:00:00Z",
                target_request="/data/model.pkl",
                strict_mode=False,
                artifacts=(),
                summary=_minimal_scan_summary(),
                total_duration_ms=-1,
            )

    def test_frozen_immutable(self) -> None:
        sr = ScanResult(
            scan_id=_VALID_UUID,
            scanner_version="2.0.0",
            timestamp_utc="2026-08-16T17:00:00Z",
            target_request="/data/model.pkl",
            strict_mode=False,
            artifacts=(_minimal_artifact_scan_result(),),
            summary=_minimal_scan_summary(),
            total_duration_ms=0,
        )
        with pytest.raises((AttributeError, TypeError)):
            sr.strict_mode = True  # type: ignore[misc]


# ===========================================================================
# SECTION 14: Exception hierarchy
# ===========================================================================


class TestScannerException:
    def test_base_construction(self) -> None:
        exc = ScannerException("Something went wrong")
        assert exc.message == "Something went wrong"
        assert exc.error_code == "SCANNER_ERROR"
        assert exc.details is None

    def test_custom_code_and_details(self) -> None:
        exc = ScannerException(
            "Parse failure",
            error_code="PARSE_ERR",
            details="at byte 128",
        )
        assert exc.error_code == "PARSE_ERR"
        assert exc.details == "at byte 128"

    def test_str_representation_without_details(self) -> None:
        exc = ScannerException("Failure", error_code="ERR")
        assert "[ERR] Failure" in str(exc)

    def test_str_representation_with_details(self) -> None:
        exc = ScannerException("Failure", error_code="ERR", details="ctx")
        result = str(exc)
        assert "ERR" in result
        assert "Failure" in result
        assert "ctx" in result

    def test_repr(self) -> None:
        exc = ScannerException("Msg", error_code="CODE")
        assert "ScannerException" in repr(exc)
        assert "CODE" in repr(exc)

    def test_is_exception(self) -> None:
        exc = ScannerException("Test")
        assert isinstance(exc, Exception)


class TestExceptionHierarchy:
    """Verify all exceptions inherit correctly from ScannerException."""

    def test_acquisition_error_hierarchy(self) -> None:
        exc = AcquisitionError("Timeout", error_code="ACQ_TIMEOUT")
        assert isinstance(exc, ScannerException)
        assert isinstance(exc, AcquisitionError)

    def test_input_error_hierarchy(self) -> None:
        exc = InputError("Bad path", error_code="INPUT_ERROR")
        assert isinstance(exc, ScannerException)
        assert isinstance(exc, AcquisitionError)  # InputError IS-A AcquisitionError
        assert isinstance(exc, InputError)

    def test_security_boundary_violation_hierarchy(self) -> None:
        exc = SecurityBoundaryViolationError("eval() attempted")
        assert isinstance(exc, ScannerException)

    def test_format_detection_error_hierarchy(self) -> None:
        exc = FormatDetectionError("Unknown magic bytes")
        assert isinstance(exc, ScannerException)

    def test_analysis_error_hierarchy(self) -> None:
        exc = AnalysisError("Generic analysis failure")
        assert isinstance(exc, ScannerException)

    def test_pickle_analysis_error_hierarchy(self) -> None:
        exc = PickleAnalysisError("Malformed opcode")
        assert isinstance(exc, ScannerException)
        assert isinstance(exc, AnalysisError)
        assert isinstance(exc, PickleAnalysisError)

    def test_safetensors_analysis_error_hierarchy(self) -> None:
        exc = SafeTensorsAnalysisError("Header too large")
        assert isinstance(exc, ScannerException)
        assert isinstance(exc, AnalysisError)

    def test_gguf_analysis_error_hierarchy(self) -> None:
        exc = GGUFAnalysisError("Magic mismatch")
        assert isinstance(exc, ScannerException)
        assert isinstance(exc, AnalysisError)

    def test_resource_limit_error_hierarchy(self) -> None:
        exc = ResourceLimitError(
            "Token budget exceeded",
            limit_name="max_json_tokens",
            limit_value=100_000,
            actual_value=200_000,
        )
        assert isinstance(exc, ScannerException)
        assert exc.limit_name == "max_json_tokens"
        assert exc.limit_value == 100_000
        assert exc.actual_value == 200_000

    def test_configuration_error_hierarchy(self) -> None:
        exc = ConfigurationError("Invalid threshold")
        assert isinstance(exc, ScannerException)

    def test_reporting_error_hierarchy(self) -> None:
        exc = ReportingError("PDF rendering failed")
        assert isinstance(exc, ScannerException)

    def test_analysis_errors_not_acquisition_errors(self) -> None:
        """Format-specific errors must NOT be confused with AcquisitionError."""
        exc = PickleAnalysisError("Bad stream")
        assert not isinstance(exc, AcquisitionError)

    def test_all_can_be_raised_and_caught(self) -> None:
        """All exception types must be raisable and catchable as ScannerException."""
        exception_types: list[type[ScannerException]] = [
            AcquisitionError,
            InputError,
            SecurityBoundaryViolationError,
            FormatDetectionError,
            AnalysisError,
            PickleAnalysisError,
            SafeTensorsAnalysisError,
            GGUFAnalysisError,
            ResourceLimitError,
            ConfigurationError,
            ReportingError,
        ]
        for exc_type in exception_types:
            with pytest.raises(ScannerException):
                raise exc_type("Test raise")

    def test_default_error_codes(self) -> None:
        """Each exception type has a specific default error_code."""
        assert AcquisitionError("x").error_code == "ACQ_ERROR"
        assert InputError("x").error_code == "INPUT_ERROR"
        assert SecurityBoundaryViolationError("x").error_code == "SECURITY_BOUNDARY_VIOLATION"
        assert FormatDetectionError("x").error_code == "FORMAT_DETECTION_ERROR"
        assert AnalysisError("x").error_code == "ANALYSIS_ERROR"
        assert PickleAnalysisError("x").error_code == "PICKLE_ANALYSIS_ERROR"
        assert SafeTensorsAnalysisError("x").error_code == "SAFETENSORS_ANALYSIS_ERROR"
        assert GGUFAnalysisError("x").error_code == "GGUF_ANALYSIS_ERROR"
        assert ResourceLimitError("x").error_code == "RESOURCE_LIMIT_ERROR"
        assert ConfigurationError("x").error_code == "CONFIG_ERROR"
        assert ReportingError("x").error_code == "REPORTING_ERROR"


# ===========================================================================
# SECTION 15: Safety — no forbidden imports
# ===========================================================================


class TestPhase2NoForbiddenImports:
    """Verify Phase 2 modules do not import forbidden libraries."""

    def test_models_no_torch(self) -> None:
        import sys

        import model_scanner.core.models  # noqa: F401
        assert "torch" not in sys.modules

    def test_exceptions_no_torch(self) -> None:
        import sys

        import model_scanner.core.exceptions  # noqa: F401
        assert "torch" not in sys.modules

    def test_models_no_safetensors(self) -> None:
        import sys

        import model_scanner.core.models  # noqa: F401
        assert "safetensors" not in sys.modules

    def test_no_third_party_in_core(self) -> None:
        # Check the delta — not the absolute state — so the test is robust when
        # optional deps (e.g. h5py) are pre-installed by earlier tests in the session.
        import sys
        forbidden = [
            "torch", "safetensors", "transformers",
            "tensorflow", "onnx", "h5py", "numpy", "pandas",
        ]
        pre_import = set(sys.modules.keys())
        import model_scanner.core.exceptions
        import model_scanner.core.models  # noqa: F401
        added = set(sys.modules.keys()) - pre_import
        for mod in forbidden:
            assert mod not in added, (
                f"Forbidden module {mod!r} was imported as a side-effect of "
                f"importing model_scanner core modules"
            )
