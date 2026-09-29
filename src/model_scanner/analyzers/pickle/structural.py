"""
model_scanner.analyzers.pickle.structural
=========================================

Structural and protocol anomaly validation for Pickle byte streams.

Checks:
  - PROTO Header anomalies (duplicate, misplaced, unsupported)
  - FRAME Boundary anomalies (opcode crosses frame, frame reopened)
  - Concatenated Multi-Stream detection
  - Trailing unparsed byte detection
  - Missing STOP opcode / truncated stream detection

SECURITY GUARANTEE
------------------
Operates strictly on raw stream boundaries and disassembled opcode metadata.
Zero dynamic execution or deserialization.

Design Reference
----------------
Section 14.3, FR-13, FR-23 of docs/implementation_design.md
"""

from __future__ import annotations

from collections.abc import Sequence

from model_scanner.analyzers.pickle.disassembler import DisassemblyOutput
from model_scanner.analyzers.pickle.opcodes import StructuralAnomaly


class StructuralAnomalyEngine:
    """Evaluates structural and protocol-level anomalies in a disassembled Pickle stream."""

    def evaluate_anomalies(
        self,
        stream_bytes: bytes,
        disassembly: DisassemblyOutput,
    ) -> Sequence[StructuralAnomaly]:
        """Perform comprehensive structural validation on stream bytes and disassembly output.

        Parameters
        ----------
        stream_bytes:
            Raw bytes of the Pickle stream.
        disassembly:
            Output from PickleDisassembler containing disassembled opcodes.

        Returns
        -------
        Sequence[StructuralAnomaly]
            List of detected structural anomalies.
        """
        anomalies: list[StructuralAnomaly] = list(disassembly.anomalies)

        # 1. Check for missing STOP opcode
        if not disassembly.stop_positions:
            anomalies.append(
                StructuralAnomaly(
                    anomaly_type="MISSING_STOP",
                    pos=len(stream_bytes),
                    description=(
                        "Stream terminated without reaching a STOP opcode (truncated stream)"
                    ),
                    rule_id="P201",
                )
            )

        # 2. Check for concatenated Pickle streams
        if disassembly.stream_count > 1:
            anomalies.append(
                StructuralAnomaly(
                    anomaly_type="CONCATENATED_STREAMS",
                    pos=disassembly.stop_positions[0],
                    description=(
                        f"Detected {disassembly.stream_count} concatenated Pickle "
                        f"streams delimited by STOP opcodes"
                    ),
                    rule_id="P206",
                )
            )

        # 3. Check for trailing unparsed bytes after final STOP opcode
        if disassembly.stop_positions:
            last_stop_pos = disassembly.stop_positions[-1]
            # In pickle opcode format, STOP opcode is 1 byte (b'.')
            consumed_bytes = last_stop_pos + 1
            trailing_len = len(stream_bytes) - consumed_bytes
            if trailing_len > 0:
                trailing_data = stream_bytes[consumed_bytes:]
                # Ignore clean whitespace / trailing newlines
                if not trailing_data.strip(b"\r\n\t "):
                    pass
                else:
                    anomalies.append(
                        StructuralAnomaly(
                            anomaly_type="TRAILING_DATA",
                            pos=consumed_bytes,
                            description=(
                                f"Detected {trailing_len} trailing unparsed bytes "
                                f"after final STOP opcode"
                            ),
                            rule_id="P207",
                        )
                    )

        # 4. Check for FRAME boundary validation (Protocol 4/5)
        self._validate_frame_boundaries(stream_bytes, disassembly, anomalies)

        # 5. Check for misplaced PROTO opcodes (PROTO not at offset 0 or within first 16 bytes)
        for op in disassembly.opcodes:
            if op.name == "PROTO" and op.pos > 16:
                anomalies.append(
                    StructuralAnomaly(
                        anomaly_type="MISPLACED_PROTO",
                        pos=op.pos,
                        description=f"PROTO opcode appeared unexpectedly at byte offset {op.pos}",
                        rule_id="P202",
                    )
                )

        return anomalies

    def _validate_frame_boundaries(
        self,
        stream_bytes: bytes,
        disassembly: DisassemblyOutput,
        anomalies: list[StructuralAnomaly],
    ) -> None:
        """Validate that all opcodes within a FRAME stay within the declared frame size."""
        current_frame_end: int | None = None

        for op in disassembly.opcodes:
            if op.name == "FRAME":
                if isinstance(op.arg, int):
                    # FRAME opcode payload is 8 bytes LE integer
                    # Opcode position + 1 byte (b'\x95') + 8 bytes length = frame payload start
                    frame_payload_start = op.pos + 9
                    current_frame_end = frame_payload_start + op.arg

                    if current_frame_end > len(stream_bytes):
                        anomalies.append(
                            StructuralAnomaly(
                                anomaly_type="FRAME_OVERFLOW",
                                pos=op.pos,
                                description=(
                                    f"Declared FRAME size ({op.arg} bytes) "
                                    f"exceeds total stream length"
                                ),
                                rule_id="P204",
                            )
                        )
            elif current_frame_end is not None:
                if op.pos > current_frame_end:
                    anomalies.append(
                        StructuralAnomaly(
                            anomaly_type="FRAME_BOUNDARY_DESYNC",
                            pos=op.pos,
                            description=(
                                f"Opcode {op.name} at offset {op.pos} crosses beyond "
                                f"declared frame boundary {current_frame_end}"
                            ),
                            rule_id="P204",
                        )
                    )
                    # Reset frame tracking to avoid duplicate reports for subsequent opcodes
                    current_frame_end = None


__all__ = ["StructuralAnomalyEngine"]
