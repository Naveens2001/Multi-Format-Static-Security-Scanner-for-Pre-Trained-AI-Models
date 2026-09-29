"""
tests.unit.test_pickle_structural
=================================

Unit tests for StructuralAnomalyEngine (PROTO, FRAME, concatenated streams, trailing data).
"""

from __future__ import annotations

from model_scanner.analyzers.pickle.disassembler import PickleDisassembler
from model_scanner.analyzers.pickle.structural import StructuralAnomalyEngine


class TestPickleStructural:
    """Test structural anomaly detection."""

    def test_missing_stop_opcode(self) -> None:
        incomplete = b"\x80\x02}q\x00"  # no '.' at the end
        dis = PickleDisassembler()
        out = dis.disassemble(incomplete)
        engine = StructuralAnomalyEngine()
        anomalies = engine.evaluate_anomalies(incomplete, out)

        assert any(a.anomaly_type == "MISSING_STOP" for a in anomalies)

    def test_concatenated_pickle_streams(self) -> None:
        # Two complete valid streams: {} and {}
        concat = b"\x80\x02}q\x00.\x80\x02}q\x00."
        dis = PickleDisassembler()
        out = dis.disassemble(concat)
        engine = StructuralAnomalyEngine()
        anomalies = engine.evaluate_anomalies(concat, out)

        assert out.stream_count == 2
        assert any(a.anomaly_type == "CONCATENATED_STREAMS" for a in anomalies)
        assert any(a.rule_id == "P206" for a in anomalies)

    def test_trailing_unparsed_data(self) -> None:
        payload = b"\x80\x02}q\x00.EXTRA_UNPARSED_BYTES_AFTER_STOP"
        dis = PickleDisassembler()
        out = dis.disassemble(payload)
        engine = StructuralAnomalyEngine()
        anomalies = engine.evaluate_anomalies(payload, out)

        assert any(a.anomaly_type == "TRAILING_DATA" for a in anomalies)
        assert any(a.rule_id == "P207" for a in anomalies)

    def test_clean_trailing_whitespace_ignored(self) -> None:
        payload = b"\x80\x02}q\x00.\r\n  \n"
        dis = PickleDisassembler()
        out = dis.disassemble(payload)
        engine = StructuralAnomalyEngine()
        anomalies = engine.evaluate_anomalies(payload, out)

        assert not any(a.anomaly_type == "TRAILING_DATA" for a in anomalies)

        framed_bad = (
            b"\x80\x04\x95\x02\x00\x00\x00\x00\x00\x00\x00"
            b"\x8c\x02os\x94\x8c\x06system\x94\x93."
        )
        dis = PickleDisassembler()
        out = dis.disassemble(framed_bad)
        engine = StructuralAnomalyEngine()
        anomalies = engine.evaluate_anomalies(framed_bad, out)

        assert any(a.anomaly_type == "FRAME_BOUNDARY_DESYNC" for a in anomalies)
        assert any(a.rule_id == "P204" for a in anomalies)

    def test_duplicate_proto_detected(self) -> None:
        dup_proto = b"\x80\x02\x80\x02}."
        dis = PickleDisassembler()
        out = dis.disassemble(dup_proto)
        engine = StructuralAnomalyEngine()
        anomalies = engine.evaluate_anomalies(dup_proto, out)

        assert any(a.anomaly_type == "DUPLICATE_PROTO" for a in anomalies)
        assert any(a.rule_id == "P202" for a in anomalies)
