"""
tests.unit.test_pickle_memo
===========================

Unit tests for bounded MemoTracker.
"""

from __future__ import annotations

from model_scanner.analyzers.pickle.memo import MemoTracker


class TestMemoTracker:
    """Test bounded 1-level memo slot tracking."""

    def test_put_and_get(self) -> None:
        memo = MemoTracker()
        assert memo.put(0, "posix.system") is True
        assert memo.get(0) == "posix.system"
        assert memo.contains(0) is True
        assert memo.size == 1

    def test_unassigned_slot_returns_none(self) -> None:
        memo = MemoTracker()
        assert memo.get(99) is None
        assert memo.contains(99) is False

    def test_memoize_auto_increment(self) -> None:
        memo = MemoTracker()
        assert memo.memoize("first") is True
        assert memo.memoize("second") is True
        assert memo.get(0) == "first"
        assert memo.get(1) == "second"
        assert memo.size == 2

    def test_capacity_limit_and_overflow(self) -> None:
        memo = MemoTracker(max_entries=3)
        assert memo.put(0, "a") is True
        assert memo.put(1, "b") is True
        assert memo.put(2, "c") is True
        assert memo.is_overflow is False

        # Attempt to insert beyond capacity
        assert memo.put(3, "d") is False
        assert memo.is_overflow is True
        assert memo.size == 3
        assert memo.get(3) is None

    def test_clear_resets_state(self) -> None:
        memo = MemoTracker()
        memo.put(0, "test")
        memo.clear()
        assert memo.size == 0
        assert memo.get(0) is None
        assert memo.is_overflow is False
