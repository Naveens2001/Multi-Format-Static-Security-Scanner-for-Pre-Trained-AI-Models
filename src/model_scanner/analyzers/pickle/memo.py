"""
model_scanner.analyzers.pickle.memo
===================================

Bounded 1-level memo table tracker for static Pickle opcode analysis.

SECURITY GUARANTEE
------------------
Stores only static strings, numbers, and symbol descriptors.
Never instantiates Python objects or holds live references.

Design Reference
----------------
Sections 14.2, FR-19 of docs/implementation_design.md
"""

from __future__ import annotations

DEFAULT_MAX_MEMO_ENTRIES: int = 10_000


class MemoTracker:
    """Tracks memo table assignments and lookups with bounded storage.

    Supports:
    - PUT, BINPUT, LONG_BINPUT (integer indexed assignment)
    - MEMOIZE (auto-incrementing slot assignment for Protocol 4/5)
    - GET, BINGET, LONG_BINGET (lookup by slot integer)

    Enforces maximum memo entries ceiling against memory exhaustion DoS.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_MEMO_ENTRIES) -> None:
        self.max_entries = max_entries
        self._memo: dict[int, str] = {}
        self._auto_index = 0
        self._overflow = False

    @property
    def is_overflow(self) -> bool:
        """Return True if memo assignments exceeded the configured maximum capacity."""
        return self._overflow

    @property
    def size(self) -> int:
        """Return the number of stored memo entries."""
        return len(self._memo)

    def put(self, slot: int, value: str) -> bool:
        """Store a static value representation at a specific memo slot.

        Returns False if maximum memo entries limit is exceeded.
        """
        if len(self._memo) >= self.max_entries and slot not in self._memo:
            self._overflow = True
            return False
        self._memo[slot] = value
        return True

    def memoize(self, value: str) -> bool:
        """Store a static value representation at the current auto-incrementing slot.

        Used by Protocol 4/5 MEMOIZE opcode.
        """
        slot = self._auto_index
        self._auto_index += 1
        return self.put(slot, value)

    def get(self, slot: int) -> str | None:
        """Look up a static representation from the memo table.

        Returns None if slot was never assigned.
        """
        return self._memo.get(slot)

    def contains(self, slot: int) -> bool:
        """Return True if the slot exists in the memo table."""
        return slot in self._memo

    def clear(self) -> None:
        """Reset the memo table and counters."""
        self._memo.clear()
        self._auto_index = 0
        self._overflow = False


__all__ = [
    "DEFAULT_MAX_MEMO_ENTRIES",
    "MemoTracker",
]
