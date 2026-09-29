"""
tests.unit.test_pickle_analyzer
===============================

End-to-end unit tests for PickleAnalyzer generating strongly-typed Finding records.
"""

from __future__ import annotations

from pathlib import Path

from model_scanner.analyzers.pickle.analyzer import PickleAnalyzer
from model_scanner.core.models import (
    Artifact,
    Confidence,
    CoordinateSpace,
    InputSourceType,
    Severity,
)


class TestPickleAnalyzer:
    """Test PickleAnalyzer finding generation and forensic evidence."""

    def test_dangerous_os_system_reduce_findings(self) -> None:
        p0_payload = b"cos\nsystem\n(S'whoami'\ntR."
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=p0_payload,
            artifact_sha256="a" * 64,
        )

        assert len(findings) >= 2
        rule_ids = [f.rule_id for f in findings]
        assert "P101" in rule_ids  # Dangerous Global
        assert "P104" in rule_ids  # Dangerous REDUCE

        p101 = next(f for f in findings if f.rule_id == "P101")
        assert p101.severity == Severity.CRITICAL
        assert p101.confidence == Confidence.HIGH
        assert p101.evidence.matched_value == "os.system"
        assert p101.evidence.byte_offset == 0
        assert p101.evidence.coordinate_space == CoordinateSpace.ARTIFACT

        p104 = next(f for f in findings if f.rule_id == "P104")
        assert p104.severity == Severity.CRITICAL
        assert p104.evidence.opcode == "REDUCE"

    def test_suspicious_ctypes_global_finding(self) -> None:
        p2_payload = b"\x80\x02cctypes\nCDLL\nq\x00."
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=p2_payload,
            artifact_sha256="b" * 64,
        )

        assert any(f.rule_id == "P102" for f in findings)
        p102 = next(f for f in findings if f.rule_id == "P102")
        assert p102.severity == Severity.HIGH
        assert p102.evidence.matched_value == "ctypes.CDLL"

        p4_payload = (
            b"\x80\x04\x95\x16\x00\x00\x00\x00\x00\x00\x00"
            b"\x8c\x02os\x94\x8c\x06system\x94\x93\x94\x8c\x02id\x94\x85\x94R\x94."
        )
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=p4_payload,
            artifact_sha256="c" * 64,
        )

        rule_ids = [f.rule_id for f in findings]
        assert "P101" in rule_ids
        assert "P103" in rule_ids  # STACK_GLOBAL
        assert "P104" in rule_ids  # REDUCE

    def test_safe_pytorch_serialization_no_danger_findings(self) -> None:
        # Standard safe PyTorch tensor rebuilder
        safe_payload = (
            b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00"
            b"(ctorch\nFloatStorage\nq\x01K\x00K\x01\x85q\x02K\x01\x85q\x03\x89}q\x04tq\x05Rq\x06."
        )
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=safe_payload,
            artifact_sha256="0" * 64,
        )

        # No dangerous/critical code execution findings
        assert not any(f.rule_id in ("P101", "P102", "P104", "P105") for f in findings)

    def test_analyze_artifact_from_file(self, tmp_path: Path) -> None:
        sample_file = tmp_path / "model.pkl"
        payload = b"cposix\nsystem\n(S'cat /etc/passwd'\ntR."
        sample_file.write_bytes(payload)

        art = Artifact(
            artifact_id="art-test",
            source_path=str(sample_file),
            input_source=InputSourceType.LOCAL_FILE,
            size_bytes=len(payload),
            sha256="d" * 64,
            local_path=sample_file,
        )

        analyzer = PickleAnalyzer()
        findings = analyzer.analyze(art)

        assert len(findings) >= 2
        assert any(f.rule_id == "P101" for f in findings)

    def test_runpy_execution_chain_escalated_to_critical(self) -> None:
        """Verify that runpy._run_code invoked via REDUCE escalates to CRITICAL P104."""
        # GLOBAL runpy._run_code
        # REDUCE (code, init_globals)
        payload = b"crunpy\n_run_code\n(S'print(hello)'\n}tR."
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=payload,
            artifact_sha256="e" * 64,
        )

        rule_ids = [f.rule_id for f in findings]
        assert "P102" in rule_ids  # The global itself is SUSPICIOUS
        assert "P101" not in rule_ids  # It should NOT be DANGEROUS natively
        assert "P104" in rule_ids  # But the REDUCE invocation escalates to CRITICAL

        p104 = next(f for f in findings if f.rule_id == "P104")
        assert p104.severity == Severity.CRITICAL
        assert p104.evidence.opcode == "REDUCE"
        assert "runpy._run_code" in p104.evidence.matched_value

    def test_benign_runpy_reference_is_suspicious_only(self) -> None:
        """Verify that simply importing runpy without REDUCE remains SUSPICIOUS, not CRITICAL."""
        # Only GLOBAL runpy._run_code, no REDUCE
        payload = b"crunpy\n_run_code\nq\x00."
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(
            data=payload,
            artifact_sha256="f" * 64,
        )

        rule_ids = [f.rule_id for f in findings]
        assert "P102" in rule_ids  # HIGH risk, suspicious
        assert "P101" not in rule_ids  # NOT critical
        assert "P104" not in rule_ids  # NO execution
