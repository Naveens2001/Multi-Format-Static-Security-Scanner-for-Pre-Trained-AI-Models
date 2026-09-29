"""
tests.robustness.test_fuzzing
=============================

Deterministic mutation, fuzzing, and truncation robustness tests ensuring
that malformed, corrupted, or hostile inputs are handled gracefully without
crashing, hanging, or executing code.

Phase 12 Robustness Hardening.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

from model_scanner.api import scan
from model_scanner.core.engine import ScanOrchestrator
from model_scanner.core.models import (
    Artifact,
    InputSourceType,
    ModelFormat,
    ScanConfig,
    ScanOutcome,
)


def _scan_bytes(
    data: bytes,
    filename: str = "test_fuzz.bin",
) -> tuple[ScanOutcome, str, int]:
    """Scan raw bytes via ScanOrchestrator.scan_artifact and return outcome, format, and score."""
    orchestrator = ScanOrchestrator(ScanConfig())
    artifact = Artifact(
        artifact_id="fuzz-test",
        source_path=filename,
        input_source=InputSourceType.LOCAL_FILE,
        size_bytes=len(data),
        sha256="0" * 64,
        stream=io.BytesIO(data),
    )
    art_result = orchestrator.scan_artifact(artifact)
    fmt = art_result.detected_format.value
    return art_result.scan_outcome, fmt, art_result.risk_score.final_score


class TestPickleRobustness:
    """Test malformed and fuzzed Pickle byte sequences."""

    def test_empty_pickle(self) -> None:
        outcome, _fmt, _ = _scan_bytes(b"", "empty.pkl")
        assert outcome in (
            ScanOutcome.UNSUPPORTED,
            ScanOutcome.COMPLETE,
            ScanOutcome.ERROR,
            ScanOutcome.INCONCLUSIVE,
        )

    def test_truncated_pickle_opcodes(self) -> None:
        # Valid PROTO 2 dict truncated before STOP
        valid = b"\x80\x02}q\x00(X\x04\x00\x00\x00testq\x01X\x03\x00\x00\x00valq\x02u."
        for cut in range(1, len(valid)):
            truncated = valid[:cut]
            outcome, _fmt, _ = _scan_bytes(truncated, "truncated.pkl")
            # Must not crash; should produce structured outcome
            assert outcome in (
                ScanOutcome.ERROR,
                ScanOutcome.WARNING,
                ScanOutcome.COMPLETE,
                ScanOutcome.INCONCLUSIVE,
            )

    def test_random_bitflips_in_pickle(self) -> None:
        base = b"\x80\x02}q\x00(X\x06\x00\x00\x00secretq\x01X\x05\x00\x00\x00valueq\x02u."
        for idx in range(len(base)):
            mutated = bytearray(base)
            mutated[idx] ^= 0xFF  # Flip all bits
            outcome, _, _ = _scan_bytes(bytes(mutated), f"flip_{idx}.pkl")
            assert isinstance(outcome, ScanOutcome)

    def test_massive_frame_length_header(self) -> None:
        # FRAME opcode with excessive declared size (DoS guard)
        framed = b"\x80\x04\x95\xff\xff\xff\xff\xff\xff\xff\x7f}."
        outcome, _, _ = _scan_bytes(framed, "large_frame.pkl")
        assert outcome in (
            ScanOutcome.WARNING,
            ScanOutcome.ERROR,
            ScanOutcome.COMPLETE,
            ScanOutcome.INCONCLUSIVE,
        )


class TestPyTorchZipRobustness:
    """Test corrupted, truncated, and malformed PyTorch ZIP archives."""

    def test_truncated_zip_magic(self) -> None:
        outcome, _, _ = _scan_bytes(b"PK\x03\x04", "corrupted.pt")
        assert outcome in (ScanOutcome.ERROR, ScanOutcome.INCONCLUSIVE)

    def test_corrupted_zip_central_directory(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("archive/data.pkl", b"\x80\x02}q\x00.")
        valid_zip = buf.getvalue()

        # Truncate end of central directory
        corrupted = valid_zip[: len(valid_zip) - 15]
        outcome, _, _ = _scan_bytes(corrupted, "corrupted_cd.pt")
        assert outcome in (ScanOutcome.ERROR, ScanOutcome.INCONCLUSIVE)

    def test_zip_with_path_traversal_entry(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../../../evil.pkl", b"cos\nsystem\n(S'id'\ntR.")
        zip_bytes = buf.getvalue()
        outcome, _, _score = _scan_bytes(zip_bytes, "traversal.pt")
        assert isinstance(outcome, ScanOutcome)

    def test_zip_missing_data_pkl(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("archive/version", b"3\n")
        zip_bytes = buf.getvalue()
        outcome, _, _ = _scan_bytes(zip_bytes, "no_data.pt")
        assert outcome in (
            ScanOutcome.COMPLETE,
            ScanOutcome.WARNING,
            ScanOutcome.ERROR,
            ScanOutcome.INCONCLUSIVE,
        )


class TestSafeTensorsRobustness:
    """Test truncated and malformed SafeTensors headers and tensors."""

    def test_truncated_header_length_prefix(self) -> None:
        outcome, _, _ = _scan_bytes(
            b"\x01\x00\x00",
            "truncated_prefix.safetensors",
        )
        assert outcome in (ScanOutcome.ERROR, ScanOutcome.INCONCLUSIVE)

    def test_invalid_json_header_bytes(self) -> None:
        header_len = 32
        prefix = struct.pack("<Q", header_len)
        garbage_json = b"{ NOT VALID JSON SYNTAX !!! }" + b" " * (header_len - 29)
        data = prefix + garbage_json
        outcome, _, _ = _scan_bytes(data, "bad_json.safetensors")
        assert outcome in (ScanOutcome.ERROR, ScanOutcome.INCONCLUSIVE)

    def test_extreme_out_of_bounds_offsets(self) -> None:
        header = json.dumps({
            "t": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 10_000_000_000],
            }
        }).encode("utf-8")
        prefix = struct.pack("<Q", len(header))
        data = prefix + header + bytes(8)
        outcome, _, _ = _scan_bytes(data, "large_offset.safetensors")
        assert outcome in (
            ScanOutcome.WARNING,
            ScanOutcome.ERROR,
            ScanOutcome.COMPLETE,
            ScanOutcome.INCONCLUSIVE,
        )

    def test_negative_tensor_data_offsets(self) -> None:
        header = json.dumps({
            "t": {"dtype": "F32", "shape": [1], "data_offsets": [-10, -5]}
        }).encode("utf-8")
        prefix = struct.pack("<Q", len(header))
        data = prefix + header + bytes(8)
        outcome, _, _ = _scan_bytes(data, "neg_offset.safetensors")
        assert outcome in (
            ScanOutcome.WARNING,
            ScanOutcome.ERROR,
            ScanOutcome.COMPLETE,
            ScanOutcome.INCONCLUSIVE,
        )


class TestGGUFRobustness:
    """Test truncated and malformed GGUF headers and descriptors."""

    def test_truncated_magic(self) -> None:
        outcome, _, score = _scan_bytes(b"GGU", "short_magic.gguf")
        assert outcome in (
            ScanOutcome.ERROR,
            ScanOutcome.UNSUPPORTED,
            ScanOutcome.COMPLETE,
            ScanOutcome.INCONCLUSIVE,
        )
        # INCONCLUSIVE artifacts use unanalyzed() with score=0; otherwise expect a non-zero score.
        if outcome not in (ScanOutcome.INCONCLUSIVE, ScanOutcome.UNSUPPORTED):
            assert score > 0

    def test_unsupported_gguf_version(self) -> None:
        # Magic GGUF + version 9999
        data = b"GGUF" + struct.pack("<I", 9999) + struct.pack("<QQ", 0, 0)
        outcome, _, _ = _scan_bytes(data, "future_ver.gguf")
        assert outcome in (ScanOutcome.ERROR, ScanOutcome.WARNING, ScanOutcome.COMPLETE, ScanOutcome.INCONCLUSIVE)

    def test_enormous_kv_count(self) -> None:
        # Magic + v3 + 0 tensors + 2^32 KV pairs
        data = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 0xFFFFFFFF)
        outcome, _, _ = _scan_bytes(data, "massive_kv.gguf")
        assert outcome in (ScanOutcome.ERROR, ScanOutcome.WARNING, ScanOutcome.COMPLETE, ScanOutcome.INCONCLUSIVE)


class TestUnsupportedFormatRobustness:
    """Test arbitrary binary noise and unrecognized extensions."""

    def test_arbitrary_random_bytes(self, tmp_path: Path) -> None:
        f = tmp_path / "noise.unknown"
        f.write_bytes(b"\xde\xad\xbe\xef\x00\x01\x02\x03\x04\x05" * 100)
        result = scan(f)
        assert result.summary.overall_outcome == ScanOutcome.UNSUPPORTED
        assert result.artifacts[0].detected_format == ModelFormat.UNSUPPORTED

    def test_legacy_non_zip_pytorch_bin(self, tmp_path: Path) -> None:
        """Legacy .bin is analyzed via its leading Pickle stream, not skipped."""
        f = tmp_path / "legacy_weights.bin"
        f.write_bytes(b"\x80\x02}q\x00.RAW_FLOAT_DATA")
        result = scan(f)
        assert result.summary.overall_outcome == ScanOutcome.COMPLETE
        assert result.artifacts[0].detected_format == ModelFormat.LEGACY_PYTORCH
