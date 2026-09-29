"""
tests.unit.test_format_detection
================================

Unit tests for Phase 3 format detection subsystem:
  - 16-byte magic sniffing and header structure detection (detection/magic.py)
  - Two-layer FormatDispatcher with heuristic detection and mismatch logic (detection/detector.py)
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

from model_scanner.core.models import (
    Artifact,
    FormatDetectionResult,
    InputSourceType,
    ModelFormat,
)
from model_scanner.detection.detector import FormatDispatcher
from model_scanner.detection.magic import (
    check_pickle_protocol_0_heuristic,
    is_ggml_magic,
    is_gguf_magic,
    is_pickle_magic,
    is_pytorch_zip_magic,
    is_safetensors_header,
    sniff_magic,
)

# ---------------------------------------------------------------------------
# 1. Magic Sniffing Functions Tests
# ---------------------------------------------------------------------------


class TestMagicSniffing:
    """Test low-level magic sniffing functions."""

    def test_pytorch_zip_magic(self) -> None:
        zip_header = b"PK\x03\x04\x14\x00\x00\x00\x08\x00" + b"\x00" * 20
        assert is_pytorch_zip_magic(zip_header) is True
        assert is_pytorch_zip_magic(b"NOT_A_ZIP_HEADER") is False

        fmt, matched, details = sniff_magic(zip_header)
        assert fmt == ModelFormat.PYTORCH_ZIP
        assert matched is True
        assert "ZIP" in details

    def test_gguf_magic(self) -> None:
        gguf_header = b"GGUF" + struct.pack("<I", 3) + b"\x00" * 16
        assert is_gguf_magic(gguf_header) is True
        assert is_gguf_magic(b"GGUX") is False

        fmt, matched, details = sniff_magic(gguf_header)
        assert fmt == ModelFormat.GGUF
        assert matched is True
        assert "GGUF" in details

    def test_ggml_magic(self) -> None:
        for magic in [b"ggml", b"ggmf", b"ggjt", b"ggla", b"lmgg"]:
            header = magic + b"\x00" * 12
            assert is_ggml_magic(header) is True
            fmt, matched, _ = sniff_magic(header)
            assert fmt == ModelFormat.GGML
            assert matched is True

    def test_safetensors_header_valid(self) -> None:
        json_prefix = b'{"weight": {"dtype": "F32", "shape": [10, 10]}}'
        header_len = len(json_prefix)
        header_bytes = struct.pack("<Q", header_len) + json_prefix + b"TENSOR_DATA"

        assert is_safetensors_header(header_bytes, total_size=len(header_bytes)) is True
        fmt, matched, details = sniff_magic(header_bytes, total_size=len(header_bytes))
        assert fmt == ModelFormat.SAFETENSORS
        assert matched is True
        assert "SafeTensors" in details

    def test_safetensors_header_invalid_non_json(self) -> None:
        # 8-byte LE int followed by non-JSON byte (not '{')
        bad_header = struct.pack("<Q", 100) + b"ABCDEFGH"
        assert is_safetensors_header(bad_header) is False

    def test_safetensors_header_too_large(self) -> None:
        # Header length exceeds max allowed
        oversized = struct.pack("<Q", 500 * 1024 * 1024) + b"{"
        assert is_safetensors_header(oversized, max_header_bytes=16 * 1024 * 1024) is False

    def test_safetensors_header_truncated_under_9_bytes(self) -> None:
        assert is_safetensors_header(b"12345678") is False

    def test_pickle_binary_protocols_deterministic_magic(self) -> None:
        # Binary Proto 2, 3, 4, 5
        for proto in range(2, 6):
            header = bytes([0x80, proto]) + b"\x00" * 10
            assert is_pickle_magic(header) is True
            fmt, matched, _ = sniff_magic(header)
            assert fmt == ModelFormat.RAW_PICKLE
            assert matched is True

    def test_pickle_protocol_0_not_binary_magic(self) -> None:
        """Protocol 0 ASCII opcodes must NOT be treated as binary magic."""
        proto0_global = b"cos\nsystem\n(S'ls'\ntR."
        assert is_pickle_magic(proto0_global) is False

        # Magic sniffing returns UNSUPPORTED because it has no binary magic header
        fmt, matched, _ = sniff_magic(proto0_global)
        assert matched is False
        assert fmt == ModelFormat.UNSUPPORTED

    def test_unsupported_header(self) -> None:
        random_bytes = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"
        fmt, matched, _ = sniff_magic(random_bytes)
        assert fmt == ModelFormat.UNSUPPORTED
        assert matched is False

    def test_empty_header(self) -> None:
        fmt, matched, _ = sniff_magic(b"")
        assert fmt == ModelFormat.UNSUPPORTED
        assert matched is False


# ---------------------------------------------------------------------------
# 2. Protocol 0 Heuristic & False Positive ASCII Tests
# ---------------------------------------------------------------------------


class TestPickleProtocol0Heuristics:
    """Test protocol 0 structural heuristics and false-positive prevention."""

    def test_ascii_false_positive_files_not_classified_as_pickle(self) -> None:
        """Ordinary ASCII/text starting with common characters must NOT be RAW_PICKLE."""
        dispatcher = FormatDispatcher()

        false_positive_samples = [
            b"c",                                    # single 'c'
            b"class MyModel:\n    pass\n",           # Python source
            b"(function() { return 1; })()",         # JavaScript
            b"Image processing configuration text",  # Starts with 'I'
            b"SELECT * FROM models WHERE id = 1;",   # SQL starting with 'S'
            b"}\n// closing brace block",            # C/C++ closing brace
            b"[section_header]\nkey = value\n",      # INI/TOML starting with '['
            b"{\n  \"key\": \"value\"\n}\n",         # JSON starting with '{'
            b"#include <stdio.h>\nint main() {}",    # C include
            b"include \"header.h\"\n",               # include directive
            b"if __name__ == '__main__':\n    pass", # Python if statement
        ]

        for sample in false_positive_samples:
            # When file has no extension
            res = dispatcher.detect_from_bytes(sample, filename_hint="unknown_file")
            assert res.detected_format == ModelFormat.UNSUPPORTED, (
                f"Sample {sample[:20]!r} falsely classified as {res.detected_format}"
            )
            assert res.magic_matched is False

            # When file has .txt extension
            res_txt = dispatcher.detect_from_bytes(sample, filename_hint="notes.txt")
            assert res_txt.detected_format == ModelFormat.UNSUPPORTED
            assert res_txt.magic_matched is False

    def test_genuine_protocol_0_strong_heuristic(self) -> None:
        """Multi-byte structural protocol 0 evidence is correctly classified."""
        dispatcher = FormatDispatcher()

        # Multi-byte GLOBAL opcode structure: c<module>\n<name>\n
        sample_global = b"cos\nsystem\n(S'whoami'\ntR."
        is_h, details = check_pickle_protocol_0_heuristic(sample_global)
        assert is_h is True
        assert "GLOBAL" in details

        res = dispatcher.detect_from_bytes(sample_global, filename_hint="model_no_ext")
        assert res.detected_format == ModelFormat.RAW_PICKLE
        assert res.magic_matched is False  # Correct: heuristic, not deterministic magic
        assert "Strong heuristic" in res.details

        # Multi-byte container structure: (dp0\n...
        sample_dict = b"(dp0\nS'weights'\np1\nS'data'\np2\ns."
        is_h2, _ = check_pickle_protocol_0_heuristic(sample_dict)
        assert is_h2 is True

        res2 = dispatcher.detect_from_bytes(sample_dict, filename_hint="weights_no_ext")
        assert res2.detected_format == ModelFormat.RAW_PICKLE
        assert res2.magic_matched is False

    def test_protocol_0_in_safetensors_triggers_mismatch(self) -> None:
        """Protocol 0 disguised as .safetensors triggers format_mismatch."""
        dispatcher = FormatDispatcher()
        sample_global = b"cos\nsystem\n(S'id'\ntR."

        res = dispatcher.detect_from_bytes(sample_global, filename_hint="model.safetensors")
        assert res.detected_format == ModelFormat.RAW_PICKLE
        assert res.magic_matched is False
        assert res.format_mismatch is True
        assert "Format mismatch detected" in res.details


# ---------------------------------------------------------------------------
# 3. FormatDispatcher Tests
# ---------------------------------------------------------------------------


class TestFormatDispatcher:
    """Test FormatDispatcher two-layer detection and mismatch detection."""

    def test_detect_pytorch_zip_from_bytes(self) -> None:
        dispatcher = FormatDispatcher()
        zip_bytes = b"PK\x03\x04" + b"\x00" * 30
        res = dispatcher.detect_from_bytes(zip_bytes, filename_hint="model.pt")

        assert isinstance(res, FormatDetectionResult)
        assert res.detected_format == ModelFormat.PYTORCH_ZIP
        assert res.magic_matched is True
        assert res.extension_matched is True
        assert res.format_mismatch is False

    def test_detect_safetensors_from_artifact(self, tmp_path: Path) -> None:
        json_meta = b'{"__metadata__": {"format": "pt"}}'
        data = struct.pack("<Q", len(json_meta)) + json_meta + b"tensor_bytes"
        model_file = tmp_path / "model.safetensors"
        model_file.write_bytes(data)

        art = Artifact(
            artifact_id="art-st",
            source_path=str(model_file),
            input_source=InputSourceType.LOCAL_FILE,
            size_bytes=len(data),
            sha256="0" * 64,
            local_path=model_file,
        )

        dispatcher = FormatDispatcher()
        res = dispatcher.detect_format(art)

        assert res.detected_format == ModelFormat.SAFETENSORS
        assert res.magic_matched is True
        assert res.format_mismatch is False

    def test_detect_from_stream_artifact(self) -> None:
        gguf_bytes = b"GGUF" + struct.pack("<I", 3) + b"\x00" * 20
        stream = io.BytesIO(gguf_bytes)

        art = Artifact(
            artifact_id="art-stream",
            source_path="memory://model.gguf",
            input_source=InputSourceType.IN_MEMORY,
            size_bytes=len(gguf_bytes),
            sha256="1" * 64,
            stream=stream,
        )

        dispatcher = FormatDispatcher()
        res = dispatcher.detect_format(art)

        assert res.detected_format == ModelFormat.GGUF
        assert res.magic_matched is True
        assert res.format_mismatch is False

    def test_format_mismatch_extension_spoofing(self) -> None:
        """Test spoofing: file named model.safetensors actually contains Pickle or ZIP."""
        dispatcher = FormatDispatcher()

        # PyTorch ZIP disguised as SafeTensors
        spoofed_zip = b"PK\x03\x04" + b"\x00" * 30
        res = dispatcher.detect_from_bytes(spoofed_zip, filename_hint="model.safetensors")

        assert res.detected_format == ModelFormat.PYTORCH_ZIP
        assert res.magic_matched is True
        assert res.extension_matched is True
        assert res.format_mismatch is True
        assert "Format mismatch detected" in res.details

        # Binary Pickle (Proto 4) disguised as SafeTensors
        spoofed_pkl = b"\x80\x04" + b"\x00" * 30
        res2 = dispatcher.detect_from_bytes(spoofed_pkl, filename_hint="weights.safetensors")

        assert res2.detected_format == ModelFormat.RAW_PICKLE
        assert res2.magic_matched is True
        assert res2.format_mismatch is True

        # Binary Pickle (Proto 4) disguised as GGUF
        res3 = dispatcher.detect_from_bytes(spoofed_pkl, filename_hint="model.gguf")
        assert res3.detected_format == ModelFormat.RAW_PICKLE
        assert res3.format_mismatch is True

    def test_legacy_pytorch_non_zip_detected_as_legacy(self) -> None:
        """Legacy non-ZIP PyTorch (.pt/.pth/.bin/.ckpt) resolves to LEGACY_PYTORCH.

        It is not UNSUPPORTED: the leading Pickle stream is a real, analyzable
        opcode stream. Returning UNSUPPORTED here previously short-circuited the
        Pickle analyzer, so renaming a malicious .pkl to .bin bypassed detection
        entirely and the artifact was reported in the CLEAN band.
        """
        dispatcher = FormatDispatcher()
        pkl_bytes = b"\x80\x02" + b"\x00" * 30

        for hint in ("model.pt", "model.pth", "pytorch_model.bin", "model.ckpt"):
            res = dispatcher.detect_from_bytes(pkl_bytes, filename_hint=hint)
            assert res.detected_format == ModelFormat.LEGACY_PYTORCH, hint
            assert "Legacy non-ZIP PyTorch serialization" in res.details

    def test_legacy_extension_does_not_hide_a_pickle_payload(self) -> None:
        """The same Pickle bytes must be analyzable under every legacy extension."""
        dispatcher = FormatDispatcher()
        payload = b"\x80\x04cos\nsystem\n(S'id'\ntR."

        assert (
            dispatcher.detect_from_bytes(payload, filename_hint="p.pkl").detected_format
            == ModelFormat.RAW_PICKLE
        )
        for hint in ("p.bin", "p.ckpt", "p.pt", "p.pth"):
            res = dispatcher.detect_from_bytes(payload, filename_hint=hint)
            assert res.detected_format == ModelFormat.LEGACY_PYTORCH, hint
            assert res.detected_format != ModelFormat.UNSUPPORTED, hint

    def test_modern_pytorch_zip_detection(self) -> None:
        """Modern PyTorch file (.pt/.bin) containing ZIP archive is PYTORCH_ZIP."""
        dispatcher = FormatDispatcher()

        # .pt containing PyTorch ZIP
        zip_bytes = b"PK\x03\x04" + b"\x00" * 30
        res = dispatcher.detect_from_bytes(zip_bytes, filename_hint="model.pt")
        assert res.detected_format == ModelFormat.PYTORCH_ZIP
        assert res.format_mismatch is False

        # .bin containing PyTorch ZIP
        res2 = dispatcher.detect_from_bytes(zip_bytes, filename_hint="pytorch_model.bin")
        assert res2.detected_format == ModelFormat.PYTORCH_ZIP
        assert res2.format_mismatch is False

    def test_extension_fallback_when_magic_inconclusive(self) -> None:
        dispatcher = FormatDispatcher()
        # Inconclusive bytes with known extension
        ambiguous_bytes = b"\x11\x22\x33\x44\x55\x66\x77\x88"
        res = dispatcher.detect_from_bytes(ambiguous_bytes, filename_hint="model.safetensors")

        assert res.detected_format == ModelFormat.SAFETENSORS
        assert res.magic_matched is False
        assert res.extension_matched is True
        assert res.format_mismatch is False
        assert "fell back to file extension" in res.details

    def test_unsupported_format(self) -> None:
        dispatcher = FormatDispatcher()
        random_bytes = b"UNKNOWN_CONTENT"
        res = dispatcher.detect_from_bytes(random_bytes, filename_hint="data.csv")

        assert res.detected_format == ModelFormat.UNSUPPORTED
        assert res.magic_matched is False
        assert res.extension_matched is False
        assert res.format_mismatch is False


# ---------------------------------------------------------------------------
# Additional coverage: detect_from_path, _extension_for_name, stream artifact
# ---------------------------------------------------------------------------


class TestDetectFromPath:
    def test_detect_safetensors_from_path(self, tmp_path: Path) -> None:
        import struct
        hdr = b'{"dummy": 1}'
        header_len = struct.pack("<Q", len(hdr))
        f = tmp_path / "model.safetensors"
        f.write_bytes(header_len + hdr)
        dispatcher = FormatDispatcher()
        result = dispatcher.detect_from_path(f)
        assert result.detected_format == ModelFormat.SAFETENSORS

    def test_detect_from_missing_path(self, tmp_path: Path) -> None:
        dispatcher = FormatDispatcher()
        result = dispatcher.detect_from_path(tmp_path / "nonexistent.safetensors")
        # Should not raise; may return SAFETENSORS by extension or UNSUPPORTED
        assert result.detected_format in (ModelFormat.SAFETENSORS, ModelFormat.UNSUPPORTED)


class TestReadArtifactHeaderStream:
    def test_seekable_stream(self) -> None:
        import io

        from model_scanner.detection.detector import _read_artifact_header

        data = b"SAFETENSORS_MAGIC"
        stream = io.BytesIO(data)
        artifact = Artifact(
            artifact_id="test",
            source_path="test.safetensors",
            local_path=None,
            input_source=InputSourceType.LOCAL_FILE,
            size_bytes=len(data),
            sha256="a" * 64,
            stream=stream,
        )
        header = _read_artifact_header(artifact)
        assert header == data
        # Stream position should be restored
        assert stream.tell() == 0

    def test_non_seekable_stream(self) -> None:
        import io

        from model_scanner.detection.detector import _read_artifact_header

        class NonSeekableStream(io.RawIOBase):
            def __init__(self, data: bytes) -> None:
                self._data = data

            def readable(self) -> bool:
                return True

            def readinto(self, b: bytearray) -> int:
                n = min(len(b), len(self._data))
                b[:n] = self._data[:n]
                self._data = self._data[n:]
                return n

        stream = io.BufferedReader(NonSeekableStream(b"HEADER_DATA"))
        artifact = Artifact(
            artifact_id="test",
            source_path="test.safetensors",
            local_path=None,
            input_source=InputSourceType.LOCAL_FILE,
            size_bytes=11,
            sha256="a" * 64,
            stream=stream,  # type: ignore[arg-type]
        )
        header = _read_artifact_header(artifact, num_bytes=6)
        assert header == b"HEADER"


class TestExtensionForName:
    def test_empty_string(self) -> None:
        from model_scanner.detection.detector import _extension_for_name
        assert _extension_for_name("") == ""

    def test_url_with_query(self) -> None:
        from model_scanner.detection.detector import _extension_for_name
        ext = _extension_for_name("model.safetensors?token=abc")
        assert ext == ".safetensors"

    def test_url_with_fragment(self) -> None:
        from model_scanner.detection.detector import _extension_for_name
        ext = _extension_for_name("model.gguf#section")
        assert ext == ".gguf"
