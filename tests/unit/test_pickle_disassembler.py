"""
tests.unit.test_pickle_disassembler
===================================

Unit tests for linear static Pickle opcode disassembler and stack walk.
"""

from __future__ import annotations

from model_scanner.analyzers.pickle.disassembler import PickleDisassembler


class TestPickleDisassembler:
    """Test static opcode extraction, STACK_GLOBAL resolution, and resource limits."""

    def test_disassemble_protocol_0_global_and_reduce(self) -> None:
        p0_bytes = b"cos\nsystem\n(S'whoami'\ntR."
        dis = PickleDisassembler()
        out = dis.disassemble(p0_bytes)

        assert len(out.opcodes) > 0
        assert len(out.globals) == 1
        assert out.globals[0].module == "os"
        assert out.globals[0].name == "system"
        assert out.globals[0].full_name == "os.system"
        assert out.globals[0].is_stack_global is False

        assert len(out.reduces) == 1
        assert "os.system" in out.reduces[0].callable_target
        assert out.stream_count == 1

    def test_disassemble_protocol_2_opcode_offsets(self) -> None:
        p2_bytes = b"\x80\x02cos\nsystem\nq\x00(X\x06\x00\x00\x00whoamiq\x01tq\x02Rq\x03."
        dis = PickleDisassembler()
        out = dis.disassemble(p2_bytes)

        assert out.protocol == 2
        assert len(out.globals) == 1
        assert out.globals[0].full_name == "os.system"
        assert out.globals[0].opcode_record.pos == 2  # right after \x80\x02
        assert len(out.reduces) == 1

    def test_disassemble_protocol_4_stack_global(self) -> None:
        p4_bytes = (
            b"\x80\x04\x95\x16\x00\x00\x00\x00\x00\x00\x00"
            b"\x8c\x02os\x94\x8c\x06system\x94\x93\x94\x8c\x02id\x94\x85\x94R\x94."
        )
        dis = PickleDisassembler()
        out = dis.disassemble(p4_bytes)

        assert out.protocol == 4
        assert len(out.globals) == 1
        assert out.globals[0].is_stack_global is True
        assert out.globals[0].module == "os"
        assert out.globals[0].name == "system"
        assert out.globals[0].full_name == "os.system"
        assert out.globals[0].resolution_status == "RESOLVED"

        assert len(out.reduces) == 1
        assert "os.system" in out.reduces[0].callable_target

    def test_disassemble_protocol_5(self) -> None:
        p5_bytes = b"\x80\x05\x95\x02\x00\x00\x00\x00\x00\x00\x00}\x94."
        dis = PickleDisassembler()
        out = dis.disassemble(p5_bytes)

        assert out.protocol == 5
        assert out.stream_count == 1

    def test_empty_stream(self) -> None:
        dis = PickleDisassembler()
        out = dis.disassemble(b"")
        assert len(out.anomalies) == 1
        assert out.anomalies[0].anomaly_type == "EMPTY_STREAM"

    def test_truncated_stream_handled_safely(self) -> None:
        truncated = b"\x80\x04\x8c\x05incomp"
        dis = PickleDisassembler()
        out = dis.disassemble(truncated)

        assert len(out.anomalies) >= 1
        assert out.anomalies[0].rule_id == "P201"

    def test_opcode_limit_guard(self) -> None:
        # Create repetitive opcode stream exceeding small max_opcodes limit
        stream = b"\x80\x02" + (b"N" * 200) + b"."
        dis = PickleDisassembler(max_opcodes=50)
        out = dis.disassemble(stream)

        assert out.resource_limit_exceeded is True
        assert any(a.anomaly_type == "EXCESSIVE_OPCODES" for a in out.anomalies)

    def test_tuple2_opcode(self) -> None:
        # pickle.dumps((1, 2), protocol=2) → uses TUPLE2
        data = bytes.fromhex("80024b014b028671002e")
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0

    def test_tuple3_opcode(self) -> None:
        # pickle.dumps((1, 2, 3), protocol=2) → uses TUPLE3
        data = bytes.fromhex("80024b014b024b038771002e")
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0

    def test_list_opcode(self) -> None:
        # pickle.dumps([1, 2], protocol=2) → uses MARK + LIST
        data = bytes.fromhex("80025d7100284b014b02652e")
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0

    def test_dict_opcode(self) -> None:
        # pickle.dumps({'a': 1}, protocol=2) → uses MARK + DICT
        data = bytes.fromhex("80027d710058010000006171014b01732e")
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0

    def test_stack_global_with_empty_stack(self) -> None:
        # \x93 is STACK_GLOBAL — emit it with no prior pushes
        data = b"\x80\x02\x93."
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        # Should handle gracefully — creates <unknown> record
        assert any(g.module == "<unknown>" for g in out.globals)

    def test_get_unresolved_memo_anomaly(self) -> None:
        # BINGET opcode (h\xNN) without prior PUT → INVALID_MEMO_GET
        data = b"\x80\x02h\x01."  # BINGET slot 1 without PUT
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert any(a.anomaly_type == "INVALID_MEMO_GET" for a in out.anomalies)

    def test_memo_overflow_anomaly(self) -> None:
        # Trigger overflow by using unique slots that exceed max_memo_entries=2
        # PROTO 2, then push 1 + BINPUT 0, push 2 + BINPUT 1, push 3 + BINPUT 2 (overflows at 2)
        data = (
            b"\x80\x02"
            b"K\x01q\x00"  # slot 0
            b"K\x02q\x01"  # slot 1 → fills memo (max=2)
            b"K\x03q\x02"  # slot 2 → new slot, triggers overflow
            b"."
        )
        dis = PickleDisassembler(max_memo_entries=2)
        out = dis.disassemble(data)
        assert any(a.anomaly_type == "MEMO_OVERFLOW" for a in out.anomalies)

    def test_pop_mark_opcode(self) -> None:
        # POP_MARK: \x81 after MARK (\x28)
        # Actually POP_MARK is opcode 1 (old-style)
        # Protocol 0: ( [mark] 1 [pop_mark]
        # Let's use pickle bytes that exercise POP_MARK
        # POP_MARK byte is 0x01
        data = b"(\x01."  # MARK then POP_MARK
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0

    def test_empty_list_empty_dict_opcodes(self) -> None:
        # ) = EMPTY_TUPLE, ] = EMPTY_LIST, } = EMPTY_DICT
        data = b"\x80\x02]q\x00}q\x01\x86q\x02."
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0

    def test_dup_opcode(self) -> None:
        # DUP = 0x32 (decimal 50)
        # Push something then DUP
        data = b"\x80\x02K\x012."  # PROTO2, BININT1 1, DUP
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0

    def test_global_single_part_arg(self) -> None:
        # Craft a GLOBAL opcode where the arg is just one part (no newline separator)
        # c<module>\n<attr>\n in protocol 0
        # We can trigger the single-part path by passing a module name without attribute
        # This is malformed but we test fault tolerance
        # c opcode (GLOBAL) = 0x63; then "only_module\n\n"
        data = b"conly_module\n\n."
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        # Should not crash; module may be parsed as "only_module" with empty attr
        assert isinstance(out.globals, list)

    def test_list_opcode_with_mark_items(self) -> None:
        # LIST opcode l (0x6c) collects items between MARK and top of stack.
        # Protocol 0: MARK, INT 1, INT 2, LIST, STOP
        # b"(I1\nI2\nl." uses ( = MARK, I = INT, l = LIST
        data = b"(I1\nI2\nl."
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0
        names = [op.name for op in out.opcodes]
        assert "LIST" in names

    def test_dict_opcode_with_mark_pairs(self) -> None:
        # DICT opcode d (0x64) collects key-value pairs between MARK.
        # Protocol 0: MARK, INT 1, INT 2, DICT, STOP
        data = b"(I1\nI2\nd."
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0
        names = [op.name for op in out.opcodes]
        assert "DICT" in names

    def test_pop_opcode_with_item_on_stack(self) -> None:
        # POP opcode 0 (0x30) discards the top of stack.
        # Protocol 0: INT 42, POP, STOP  (b"I42\n0.")
        # '0' = 0x30 = POP opcode
        data = b"I42\n0."
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0
        names = [op.name for op in out.opcodes]
        assert "POP" in names

    def test_pop_mark_opcode_with_mark_on_stack(self) -> None:
        # POP_MARK opcode 1 (0x31 = ASCII '1') discards items up to MARK.
        # Protocol 0: MARK, INT 42, POP_MARK, STOP  (b"(I42\n1.")
        # The existing test_pop_mark_opcode uses \x01 (not POP_MARK); this uses b'1'.
        data = b"(I42\n1."
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0
        names = [op.name for op in out.opcodes]
        assert "POP_MARK" in names

    def test_memoize_opcode_overflow(self) -> None:
        # MEMOIZE (\x94) automatically assigns the next memo slot.
        # With max_memo_entries=1, a second MEMOIZE should trigger MEMO_OVERFLOW.
        # Protocol 4: PROTO 4, FRAME, BININT1 1, MEMOIZE, BININT1 2, MEMOIZE, STOP
        data = (
            b"\x80\x04"               # PROTO 4
            b"\x95\x0e\x00\x00\x00\x00\x00\x00\x00"  # FRAME (14-byte payload)
            b"K\x01\x94"              # BININT1 1, MEMOIZE (slot 0)
            b"K\x02\x94"              # BININT1 2, MEMOIZE (slot 1 → overflow at max=1)
            b"."
        )
        dis = PickleDisassembler(max_memo_entries=1)
        out = dis.disassemble(data)
        assert any(
            a.anomaly_type == "MEMO_OVERFLOW" for a in out.anomalies
        ), "Expected MEMO_OVERFLOW from MEMOIZE opcode"

    def test_get_opcode_memo_hit_resolved(self) -> None:
        # GET opcode g (protocol 0) on a populated memo slot pushes the value.
        # Protocol 0: INT 42, PUT slot 0, GET slot 0 (hit)
        # p0\n = PUT decimal 0; g0\n = GET decimal 0
        data = b"I42\np0\ng0\n."
        dis = PickleDisassembler()
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0
        # No INVALID_MEMO_GET anomaly — slot 0 was populated by PUT
        assert not any(
            a.anomaly_type == "INVALID_MEMO_GET" for a in out.anomalies
        ), "GET on populated memo slot should not produce INVALID_MEMO_GET"

    def test_stack_depth_overflow_shifts_bounded_stack(self) -> None:
        # _push_stack shifts the oldest entry when max_stack_depth is exceeded.
        # With max_stack_depth=2, pushing 3 items triggers the pop(0)+append path.
        # NNN = three NONE pushes (b'N' = NONE opcode)
        data = b"NNN."
        dis = PickleDisassembler(max_stack_depth=2)
        out = dis.disassemble(data)
        assert len(out.opcodes) > 0
