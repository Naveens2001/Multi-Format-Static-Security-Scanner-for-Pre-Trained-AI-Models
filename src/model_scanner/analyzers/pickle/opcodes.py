"""
model_scanner.analyzers.pickle.opcodes
======================================

Normalized opcode and analysis data transfer records for static Pickle analysis.

SECURITY GUARANTEE
------------------
These structures store only statically parsed metadata and string representations.
They NEVER store live instantiated Python objects or dynamic callables.

Design Reference
----------------
Sections 14.2, 14.3 of docs/implementation_design.md
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpcodeRecord:
    """Normalized representation of a single disassembled Pickle opcode."""

    index: int
    name: str
    code: str
    arg: object
    arg_repr: str
    pos: int
    proto: int | None = None


@dataclass(frozen=True)
class GLOBALRecord:
    """Static record of a GLOBAL, STACK_GLOBAL, or INST opcode."""

    opcode_record: OpcodeRecord
    module: str
    name: str
    is_stack_global: bool
    resolution_status: str  # "RESOLVED" | "UNKNOWN" | "AMBIGUOUS"

    @property
    def full_name(self) -> str:
        """Return the fully-qualified symbol name e.g. 'os.system'."""
        if self.module and self.name:
            return f"{self.module}.{self.name}"
        return self.module or self.name or "<unknown>"


@dataclass(frozen=True)
class REDUCERecord:
    """Static record of a REDUCE opcode and its targeted callable."""

    opcode_record: OpcodeRecord
    callable_target: str
    args_repr: str
    resolution_status: str  # "RESOLVED" | "UNKNOWN" | "UNTRACKED"


@dataclass(frozen=True)
class BUILDRecord:
    """Static record of a BUILD state mutation opcode."""

    opcode_record: OpcodeRecord
    target_repr: str
    state_repr: str


@dataclass(frozen=True)
class NEWOBJRecord:
    """Static record of a NEWOBJ or NEWOBJ_EX instantiation opcode."""

    opcode_record: OpcodeRecord
    cls_repr: str
    args_repr: str
    is_newobj_ex: bool


@dataclass(frozen=True)
class StructuralAnomaly:
    """Record of a structural or protocol-level anomaly encountered during parsing."""

    anomaly_type: str
    pos: int | None
    description: str
    rule_id: str


__all__ = [
    "BUILDRecord",
    "GLOBALRecord",
    "NEWOBJRecord",
    "OpcodeRecord",
    "REDUCERecord",
    "StructuralAnomaly",
]
