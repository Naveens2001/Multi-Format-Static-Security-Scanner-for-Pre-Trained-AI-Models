"""
tests.unit.test_pytorch_zip
===========================

Unit tests for PyTorch ZIP container inspector and embedded Pickle analyzer.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from model_scanner.analyzers.pytorch.analyzer import PyTorchAnalyzer
from model_scanner.analyzers.pytorch.zip_reader import PyTorchZipReader
from model_scanner.core.exceptions import AnalysisError
from model_scanner.core.models import (
    Artifact,
    CoordinateSpace,
    InputSourceType,
    Severity,
)


def _create_synthetic_pytorch_zip(
    pickle_filename: str = "archive/data.pkl",
    pickle_bytes: bytes = b"\x80\x02}q\x00.",
    extra_files: dict[str, bytes] | None = None,
) -> bytes:
    """Helper to build an in-memory synthetic PyTorch ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(pickle_filename, pickle_bytes)
        if extra_files:
            for fn, data in extra_files.items():
                zf.writestr(fn, data)
    return buf.getvalue()


class TestPyTorchZipReader:
    """Test PyTorchZipReader bounded member discovery and extraction."""

    def test_list_members_identifies_candidate_pickles(self) -> None:
        zip_bytes = _create_synthetic_pytorch_zip(
            pickle_filename="archive/data.pkl",
            extra_files={
                "archive/version": b"3",
                "archive/data/0": b"\x00" * 100,  # tensor storage
                "archive/data/1": b"\xff" * 100,  # tensor storage
            },
        )
        reader = PyTorchZipReader()
        members = reader.list_members(zip_bytes)

        assert len(members) == 4
        cand_names = [m.filename for m in members if m.is_candidate_pickle]
        assert cand_names == ["archive/data.pkl"]

        tensor_names = [m.filename for m in members if getattr(m, 'is_tensor_data', False)]
        assert "archive/version" in tensor_names
        assert "archive/data/0" in tensor_names
        assert "archive/data/1" in tensor_names

    def test_extract_member_success(self) -> None:
        expected = b"\x80\x02}q\x00."
        zip_bytes = _create_synthetic_pytorch_zip(pickle_bytes=expected)
        reader = PyTorchZipReader()
        extracted = reader.extract_member(zip_bytes, "archive/data.pkl")
        assert extracted == expected

    def test_extract_missing_member_raises_analysis_error(self) -> None:
        zip_bytes = _create_synthetic_pytorch_zip()
        reader = PyTorchZipReader()
        with pytest.raises(AnalysisError) as exc_info:
            reader.extract_member(zip_bytes, "nonexistent.pkl")
        assert exc_info.value.error_code == "MEMBER_NOT_FOUND"

    def test_corrupted_zip_raises_analysis_error(self) -> None:
        reader = PyTorchZipReader()
        with pytest.raises(AnalysisError) as exc_info:
            reader.list_members(b"PK\x03\x04CORRUPTED_ZIP_STREAM")
        assert exc_info.value.error_code == "INVALID_ZIP_ARCHIVE"

    def test_list_members_too_many_raises_resource_limit(self) -> None:
        # Covers zip_reader.py line 108: member count exceeds max_members ceiling.
        from model_scanner.core.exceptions import ResourceLimitError

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("archive/data.pkl", b"\x80\x02N.")
            zf.writestr("archive/extra.pkl", b"\x80\x02N.")
        zip_bytes = buf.getvalue()

        reader = PyTorchZipReader(max_members=1)
        with pytest.raises(ResourceLimitError) as exc_info:
            reader.list_members(zip_bytes)
        assert exc_info.value.error_code == "ZIP_TOO_MANY_MEMBERS"

    def test_list_members_skips_directory_entries(self) -> None:
        # Covers zip_reader.py line 120: directory entries are excluded from results.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            # Add a directory entry (filename ends with '/')
            dir_info = zipfile.ZipInfo("somedir/")
            zf.writestr(dir_info, "")
            zf.writestr("archive/data.pkl", b"\x80\x02N.")
        zip_bytes = buf.getvalue()

        reader = PyTorchZipReader()
        members = reader.list_members(zip_bytes)
        filenames = [m.filename for m in members]
        # Directory entry is skipped; only the pickle file appears
        assert "somedir/" not in filenames
        assert "archive/data.pkl" in filenames

    def test_extract_member_too_large_raises_resource_limit(self) -> None:
        # Covers zip_reader.py lines 179-187: declared file_size exceeds max_member_bytes.
        from model_scanner.core.exceptions import ResourceLimitError

        content = b"A" * 20
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("archive/data.pkl", content)
        zip_bytes = buf.getvalue()

        reader = PyTorchZipReader(max_member_bytes=10)
        with pytest.raises(ResourceLimitError) as exc_info:
            reader.extract_member(zip_bytes, "archive/data.pkl")
        assert exc_info.value.error_code == "MEMBER_TOO_LARGE"


class TestPyTorchAnalyzer:
    """Test PyTorchAnalyzer static inspection and embedded Pickle findings."""

    def test_safe_pytorch_zip_findings(self) -> None:
        safe_pickle = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00."
        zip_bytes = _create_synthetic_pytorch_zip(
            pickle_filename="archive/data.pkl",
            pickle_bytes=safe_pickle,
        )

        art = Artifact(
            artifact_id="art-pt-safe",
            source_path="model.pt",
            input_source=InputSourceType.IN_MEMORY,
            size_bytes=len(zip_bytes),
            sha256="e" * 64,
            stream=io.BytesIO(zip_bytes),
        )

        analyzer = PyTorchAnalyzer()
        findings = analyzer.analyze(art)

        # Has informational finding P209 for embedded pickle container
        assert any(f.rule_id == "P209" for f in findings)
        p209 = next(f for f in findings if f.rule_id == "P209")
        assert p209.severity == Severity.INFO
        assert p209.evidence.coordinate_space == CoordinateSpace.ARCHIVE_MEMBER
        assert p209.evidence.archive_member == "archive/data.pkl"

        # No dangerous/critical code execution findings
        assert not any(f.rule_id in ("P101", "P102", "P104") for f in findings)

    def test_tensor_storage_ignored_by_analyzer(self) -> None:
        """Test that raw tensors are not parsed as Pickle, avoiding P201 false positives."""
        safe_pickle = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00."
        zip_bytes = _create_synthetic_pytorch_zip(
            pickle_filename="archive/data.pkl",
            pickle_bytes=safe_pickle,
            extra_files={
                "archive/version": b"3",
                "archive/data/0": b"\x80\x02\x00\x00\x00" * 20,  # malicious looking magic bytes
                "archive/data/1": b"`" * 100,  # garbage bytes
            }
        )

        art = Artifact(
            artifact_id="art-pt-tensors",
            source_path="model.pt",
            input_source=InputSourceType.IN_MEMORY,
            size_bytes=len(zip_bytes),
            sha256="a" * 64,
            stream=io.BytesIO(zip_bytes),
        )

        analyzer = PyTorchAnalyzer()
        findings = analyzer.analyze(art)

        # Should only have P209 for data.pkl
        assert len(findings) == 1
        assert findings[0].rule_id == "P209"
        assert findings[0].evidence.archive_member == "archive/data.pkl"

        # No P201 errors from trying to parse data/0 or data/1
        assert not any(f.rule_id == "P201" for f in findings)

    def test_malicious_pytorch_zip_findings(self) -> None:
        malicious_pickle = b"cos\nsystem\n(S'rm -rf /'\ntR."
        zip_bytes = _create_synthetic_pytorch_zip(
            pickle_filename="archive/data.pkl",
            pickle_bytes=malicious_pickle,
        )

        art = Artifact(
            artifact_id="art-pt-malicious",
            source_path="malicious_model.pt",
            input_source=InputSourceType.IN_MEMORY,
            size_bytes=len(zip_bytes),
            sha256="f" * 64,
            stream=io.BytesIO(zip_bytes),
        )

        analyzer = PyTorchAnalyzer()
        findings = analyzer.analyze(art)

        rule_ids = [f.rule_id for f in findings]
        assert "P209" in rule_ids  # Container finding
        assert "P101" in rule_ids  # Dangerous Global
        assert "P104" in rule_ids  # Dangerous REDUCE

        p101 = next(f for f in findings if f.rule_id == "P101")
        assert p101.severity == Severity.CRITICAL
        assert p101.evidence.coordinate_space == CoordinateSpace.ARCHIVE_MEMBER
        assert p101.evidence.archive_member == "archive/data.pkl"
        assert p101.evidence.matched_value == "os.system"

    def test_corrupted_zip_generates_p201_finding(self) -> None:
        bad_bytes = b"PK\x03\x04_corrupted_archive_data"
        art = Artifact(
            artifact_id="art-pt-corrupt",
            source_path="corrupt.pt",
            input_source=InputSourceType.IN_MEMORY,
            size_bytes=len(bad_bytes),
            sha256="9" * 64,
            stream=io.BytesIO(bad_bytes),
        )

        analyzer = PyTorchAnalyzer()
        findings = analyzer.analyze(art)

        assert any(f.rule_id == "P201" for f in findings)

    def test_nested_zip_emits_p210(self) -> None:
        """A ZIP member that is itself a ZIP triggers P210 (nested archive detection)."""
        # Build an inner ZIP archive
        inner_buf = io.BytesIO()
        with zipfile.ZipFile(inner_buf, "w", compression=zipfile.ZIP_STORED) as inner_zf:
            inner_zf.writestr("payload.txt", b"inner content")
        inner_zip_bytes = inner_buf.getvalue()

        # Embed it as a non-pickle, non-tensor member in the outer PyTorch ZIP
        zip_bytes = _create_synthetic_pytorch_zip(
            pickle_filename="archive/data.pkl",
            pickle_bytes=b"\x80\x02}q\x00.",
            extra_files={"archive/extra.zip": inner_zip_bytes},
        )

        art = Artifact(
            artifact_id="art-pt-nested",
            source_path="model.pt",
            input_source=InputSourceType.IN_MEMORY,
            size_bytes=len(zip_bytes),
            sha256="b" * 64,
            stream=io.BytesIO(zip_bytes),
        )

        analyzer = PyTorchAnalyzer()
        findings = analyzer.analyze(art)

        rule_ids = [f.rule_id for f in findings]
        assert "P210" in rule_ids, f"Expected P210 in {rule_ids}"
        p210 = next(f for f in findings if f.rule_id == "P210")
        assert p210.severity == Severity.MEDIUM
        assert p210.evidence.archive_member == "archive/extra.zip"

    def test_extreme_compression_ratio_emits_p211(self) -> None:
        """A member with compress_size / file_size < 0.001 triggers P211 (zip-bomb indicator)."""
        # 1,000,000 zero bytes DEFLATE-compresses to ~985 bytes:
        # ratio = 985 / 1_000_000 = 0.000985 < 0.001 → triggers P211.
        outer_buf = io.BytesIO()
        with zipfile.ZipFile(outer_buf, "w") as zf:
            # Write a normal pickle member
            zf.writestr("archive/data.pkl", b"\x80\x02}q\x00.")
            # Write a highly-compressible member (all-zeros, 1 MB)
            zf.writestr(
                zipfile.ZipInfo("archive/weights.bin"),
                b"\x00" * 1_000_000,
                compress_type=zipfile.ZIP_DEFLATED,
            )
        zip_bytes = outer_buf.getvalue()

        art = Artifact(
            artifact_id="art-pt-zipbomb",
            source_path="model.pt",
            input_source=InputSourceType.IN_MEMORY,
            size_bytes=len(zip_bytes),
            sha256="c" * 64,
            stream=io.BytesIO(zip_bytes),
        )

        analyzer = PyTorchAnalyzer()
        findings = analyzer.analyze(art)

        rule_ids = [f.rule_id for f in findings]
        # P211 should fire for the highly-compressible member
        assert "P211" in rule_ids, (
            f"Expected P211 for high compression ratio member; got {rule_ids}"
        )
        p211 = next(f for f in findings if f.rule_id == "P211")
        assert p211.severity == Severity.HIGH
