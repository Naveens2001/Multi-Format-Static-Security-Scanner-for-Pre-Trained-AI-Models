"""
model_scanner.analyzers.pickle.disassembler
==========================================

Linear static Pickle opcode disassembler and stack/memo tracer using pickletools.genops().

SECURITY GUARANTEE
------------------
Uses pickletools.genops() ONLY for static tokenization.
Never invokes pickle.loads(), never instantiates classes, never calls callables.

Design Reference
----------------
Sections 14.2, FR-16, FR-18 of docs/implementation_design.md
"""

from __future__ import annotations

import io
import pickletools
import struct
from dataclasses import dataclass, field

from model_scanner.analyzers.pickle.memo import MemoTracker
from model_scanner.analyzers.pickle.opcodes import (
    BUILDRecord,
    GLOBALRecord,
    NEWOBJRecord,
    OpcodeRecord,
    REDUCERecord,
    StructuralAnomaly,
)

DEFAULT_MAX_OPCODES: int = 500_000
DEFAULT_MAX_STACK_DEPTH: int = 1_000


@dataclass
class DisassemblyOutput:
    """Aggregated output from a static Pickle disassembly run."""

    opcodes: list[OpcodeRecord] = field(default_factory=list)
    globals: list[GLOBALRecord] = field(default_factory=list)
    reduces: list[REDUCERecord] = field(default_factory=list)
    builds: list[BUILDRecord] = field(default_factory=list)
    newobjs: list[NEWOBJRecord] = field(default_factory=list)
    anomalies: list[StructuralAnomaly] = field(default_factory=list)
    stream_count: int = 0
    protocol: int | None = None
    stop_positions: list[int] = field(default_factory=list)
    total_bytes_parsed: int = 0
    resource_limit_exceeded: bool = False


class PickleDisassembler:
    """Linear static Pickle disassembler with bounded stack and memo tracking."""

    def __init__(
        self,
        max_opcodes: int = DEFAULT_MAX_OPCODES,
        max_memo_entries: int = 10_000,
        max_stack_depth: int = DEFAULT_MAX_STACK_DEPTH,
    ) -> None:
        self.max_opcodes = max_opcodes
        self.max_memo_entries = max_memo_entries
        self.max_stack_depth = max_stack_depth

    def disassemble(self, stream_bytes: bytes) -> DisassemblyOutput:
        """Disassemble a raw Pickle byte buffer into structured events.

        Parameters
        ----------
        stream_bytes:
            Raw bytes of the Pickle stream.

        Returns
        -------
        DisassemblyOutput
            Extracted opcode records, global calls, REDUCE calls, and anomalies.
        """
        output = DisassemblyOutput()
        if not stream_bytes:
            output.anomalies.append(
                StructuralAnomaly(
                    anomaly_type="EMPTY_STREAM",
                    pos=0,
                    description="Pickle stream is empty (0 bytes)",
                    rule_id="P201",
                )
            )
            return output

        memo = MemoTracker(max_entries=self.max_memo_entries)
        stack: list[str] = []
        stream = io.BytesIO(stream_bytes)
        opcode_count = 0
        last_pos = 0

        while stream.tell() < len(stream_bytes):
            start_tell = stream.tell()
            remaining = stream_bytes[start_tell:]
            if not remaining.strip(b"\r\n\t "):
                break

            try:
                for opcode, arg, pos in pickletools.genops(stream):
                    pos_int = pos if isinstance(pos, int) else 0
                    last_pos = pos_int
                    opcode_count += 1

                    if opcode_count > self.max_opcodes:
                        output.resource_limit_exceeded = True
                        output.anomalies.append(
                            StructuralAnomaly(
                                anomaly_type="EXCESSIVE_OPCODES",
                                pos=pos_int,
                                description=(
                                    f"Opcode count exceeded maximum limit of {self.max_opcodes}"
                                ),
                                rule_id="P208",
                            )
                        )
                        break

                    arg_repr = repr(arg) if arg is not None else ""
                    op_rec = OpcodeRecord(
                        index=opcode_count,
                        name=opcode.name,
                        code=opcode.code,
                        arg=arg,
                        arg_repr=arg_repr,
                        pos=pos_int,
                        proto=opcode.proto,
                    )
                    output.opcodes.append(op_rec)

                    # Track protocol declaration
                    if opcode.name == "PROTO":
                        if isinstance(arg, int):
                            if output.protocol is not None:
                                output.anomalies.append(
                                    StructuralAnomaly(
                                        anomaly_type="DUPLICATE_PROTO",
                                        pos=pos_int,
                                        description=f"Duplicate PROTO opcode found: {arg}",
                                        rule_id="P202",
                                    )
                                )
                            else:
                                output.protocol = arg
                                if arg > 5:
                                    output.anomalies.append(
                                        StructuralAnomaly(
                                            anomaly_type="UNSUPPORTED_PROTOCOL",
                                            pos=pos_int,
                                            description=(
                                                f"Unsupported Pickle protocol version: {arg}"
                                            ),
                                            rule_id="P203",
                                        )
                                    )

                    # --- 1. GLOBAL & STACK_GLOBAL Detection ---
                    if opcode.name in ("GLOBAL", "INST"):
                        mod_name = ""
                        attr_name = ""
                        if isinstance(arg, str):
                            parts = arg.split(None, 1) if " " in arg else arg.split("\n", 1)
                            if len(parts) == 2:
                                mod_name, attr_name = parts[0].strip(), parts[1].strip()
                            elif len(parts) == 1:
                                mod_name = parts[0].strip()

                        g_rec = GLOBALRecord(
                            opcode_record=op_rec,
                            module=mod_name,
                            name=attr_name,
                            is_stack_global=False,
                            resolution_status="RESOLVED",
                        )
                        output.globals.append(g_rec)
                        self._push_stack(stack, g_rec.full_name)

                    elif opcode.name == "STACK_GLOBAL":
                        if len(stack) >= 2:
                            attr_name = stack.pop()
                            mod_name = stack.pop()
                            g_rec = GLOBALRecord(
                                opcode_record=op_rec,
                                module=mod_name,
                                name=attr_name,
                                is_stack_global=True,
                                resolution_status="RESOLVED",
                            )
                        else:
                            g_rec = GLOBALRecord(
                                opcode_record=op_rec,
                                module="<unknown>",
                                name="<unknown>",
                                is_stack_global=True,
                                resolution_status="UNKNOWN",
                            )
                        output.globals.append(g_rec)
                        self._push_stack(stack, g_rec.full_name)

                    # --- 2. REDUCE Detection ---
                    elif opcode.name == "REDUCE":
                        args_val = stack.pop() if stack else "<unknown_args>"
                        callable_val = stack.pop() if stack else "<unknown_callable>"
                        is_res = callable_val != "<unknown_callable>"

                        r_rec = REDUCERecord(
                            opcode_record=op_rec,
                            callable_target=callable_val,
                            args_repr=args_val,
                            resolution_status="RESOLVED" if is_res else "UNKNOWN",
                        )
                        output.reduces.append(r_rec)
                        self._push_stack(stack, f"REDUCE({callable_val})")

                    # --- 3. BUILD Detection ---
                    elif opcode.name == "BUILD":
                        state_val = stack.pop() if stack else "<unknown_state>"
                        target_val = stack.pop() if stack else "<unknown_target>"

                        b_rec = BUILDRecord(
                            opcode_record=op_rec,
                            target_repr=target_val,
                            state_repr=state_val,
                        )
                        output.builds.append(b_rec)
                        self._push_stack(stack, f"BUILD({target_val})")

                    # --- 4. NEWOBJ / NEWOBJ_EX Detection ---
                    elif opcode.name in ("NEWOBJ", "NEWOBJ_EX"):
                        kwargs_val = stack.pop() if opcode.name == "NEWOBJ_EX" and stack else ""
                        args_val = stack.pop() if stack else "<unknown_args>"
                        cls_val = stack.pop() if stack else "<unknown_cls>"
                        combined_args = f"{args_val}, {kwargs_val}".rstrip(", ")

                        n_rec = NEWOBJRecord(
                            opcode_record=op_rec,
                            cls_repr=cls_val,
                            args_repr=combined_args,
                            is_newobj_ex=(opcode.name == "NEWOBJ_EX"),
                        )
                        output.newobjs.append(n_rec)
                        self._push_stack(stack, f"NEWOBJ({cls_val})")

                    # --- 5. Memo Assignment & Lookup Tracking ---
                    elif opcode.name in ("PUT", "BINPUT", "LONG_BINPUT"):
                        if isinstance(arg, int):
                            top_val = stack[-1] if stack else ""
                            ok = memo.put(arg, top_val)
                            if not ok:
                                output.anomalies.append(
                                    StructuralAnomaly(
                                        anomaly_type="MEMO_OVERFLOW",
                                        pos=pos_int,
                                        description="Memo table capacity exceeded configured limit",
                                        rule_id="P208",
                                    )
                                )

                    elif opcode.name == "MEMOIZE":
                        top_val = stack[-1] if stack else ""
                        ok = memo.memoize(top_val)
                        if not ok:
                            output.anomalies.append(
                                StructuralAnomaly(
                                    anomaly_type="MEMO_OVERFLOW",
                                    pos=pos_int,
                                    description="Memo table capacity exceeded configured limit",
                                    rule_id="P208",
                                )
                            )

                    elif opcode.name in ("GET", "BINGET", "LONG_BINGET"):
                        if isinstance(arg, int):
                            resolved_memo = memo.get(arg)
                            if resolved_memo is not None:
                                self._push_stack(stack, resolved_memo)
                            else:
                                output.anomalies.append(
                                    StructuralAnomaly(
                                        anomaly_type="INVALID_MEMO_GET",
                                        pos=pos_int,
                                        description=f"GET lookup on unassigned memo slot: {arg}",
                                        rule_id="P205",
                                    )
                                )
                                self._push_stack(stack, f"<memo_{arg}>")

                    # --- 6. Container & Marker opcodes ---
                    elif opcode.name == "MARK":
                        self._push_stack(stack, "<MARK>")

                    elif opcode.name == "TUPLE":
                        items: list[str] = []
                        while stack and stack[-1] != "<MARK>":
                            items.append(stack.pop())
                        if stack and stack[-1] == "<MARK>":
                            stack.pop()  # pop <MARK>
                        items.reverse()
                        self._push_stack(stack, f"({', '.join(items)})")

                    elif opcode.name == "TUPLE1":
                        top = stack.pop() if stack else ""
                        self._push_stack(stack, f"({top},)")

                    elif opcode.name == "TUPLE2":
                        b_val = stack.pop() if stack else ""
                        a_val = stack.pop() if stack else ""
                        self._push_stack(stack, f"({a_val}, {b_val})")

                    elif opcode.name == "TUPLE3":
                        c_val = stack.pop() if stack else ""
                        b_val = stack.pop() if stack else ""
                        a_val = stack.pop() if stack else ""
                        self._push_stack(stack, f"({a_val}, {b_val}, {c_val})")

                    elif opcode.name == "EMPTY_TUPLE":
                        self._push_stack(stack, "()")

                    elif opcode.name == "LIST":
                        items = []
                        while stack and stack[-1] != "<MARK>":
                            items.append(stack.pop())
                        if stack and stack[-1] == "<MARK>":
                            stack.pop()
                        items.reverse()
                        self._push_stack(stack, f"[{', '.join(items)}]")

                    elif opcode.name == "EMPTY_LIST":
                        self._push_stack(stack, "[]")

                    elif opcode.name == "DICT":
                        while stack and stack[-1] != "<MARK>":
                            stack.pop()
                        if stack and stack[-1] == "<MARK>":
                            stack.pop()
                        self._push_stack(stack, "{...}")

                    elif opcode.name == "EMPTY_DICT":
                        self._push_stack(stack, "{}")

                    elif opcode.name == "POP":
                        if stack:
                            stack.pop()

                    elif opcode.name == "POP_MARK":
                        while stack and stack[-1] != "<MARK>":
                            stack.pop()
                        if stack and stack[-1] == "<MARK>":
                            stack.pop()

                    elif opcode.name == "DUP":
                        if stack:
                            self._push_stack(stack, stack[-1])

                    # --- 7. Literal Push Opcodes ---
                    elif opcode.name in (
                        "UNICODE",
                        "SHORT_BINUNICODE",
                        "BINUNICODE",
                        "BINUNICODE8",
                        "STRING",
                        "BINSTRING",
                        "SHORT_BINSTRING",
                        "INT",
                        "BININT",
                        "BININT1",
                        "BININT2",
                        "LONG",
                        "LONG1",
                        "LONG4",
                        "FLOAT",
                        "BINFLOAT",
                    ):
                        val_str = str(arg) if arg is not None else ""
                        self._push_stack(stack, val_str)

                    elif opcode.name == "NONE":
                        self._push_stack(stack, "None")

                    # --- 8. STOP boundary ---
                    elif opcode.name == "STOP":
                        output.stream_count += 1
                        output.stop_positions.append(pos_int)

            except (ValueError, IndexError, KeyError, EOFError, struct.error, OSError) as exc:
                output.anomalies.append(
                    StructuralAnomaly(
                        anomaly_type="MALFORMED_STREAM",
                        pos=last_pos,
                        description=f"Malformed opcode stream: {exc}",
                        rule_id="P201",
                    )
                )
                break

            if stream.tell() == start_tell:
                break

        output.total_bytes_parsed = len(stream_bytes)
        return output

    def _push_stack(self, stack: list[str], val: str) -> None:
        """Push a static string descriptor onto the bounded stack."""
        if len(stack) < self.max_stack_depth:
            stack.append(val)
        else:
            # Shift stack to maintain bounded depth
            stack.pop(0)
            stack.append(val)


__all__ = [
    "DEFAULT_MAX_OPCODES",
    "DEFAULT_MAX_STACK_DEPTH",
    "DisassemblyOutput",
    "PickleDisassembler",
]
