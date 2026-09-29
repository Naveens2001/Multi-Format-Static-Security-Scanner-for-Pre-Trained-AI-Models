"""
tests.unit.test_pickle_safety
=============================

Security and non-execution verification tests for Phase 4 static analysis engine.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

from model_scanner.analyzers.pickle.analyzer import PickleAnalyzer


class TestPickleSafetyGuarantees:
    """Ensure zero model execution, zero deserialization, and zero dynamic loading."""

    def test_no_torch_imported(self) -> None:
        forbidden = ["torch", "safetensors", "transformers", "fickling", "picklescan"]
        for mod in forbidden:
            assert mod not in sys.modules, f"Forbidden module {mod} is loaded in sys.modules!"

    @patch("os.system")
    def test_dangerous_os_system_not_executed(self, mock_system: object) -> None:
        # Construct pickle targeting os.system
        p0_payload = b"cos\nsystem\n(S'echo EXPLOIT_SUCCESS'\ntR."
        analyzer = PickleAnalyzer()
        findings = analyzer.analyze_bytes(p0_payload, "0" * 64)

        # Mock system must NOT be called
        assert not hasattr(mock_system, "called") or not mock_system.called
        assert any(f.rule_id == "P101" for f in findings)
        assert any(f.rule_id == "P104" for f in findings)

    def test_no_pickle_loads_in_disassembler(self) -> None:
        # Patch pickle.loads to raise if invoked anywhere during analysis
        with patch("pickle.loads", side_effect=RuntimeError("pickle.loads was invoked!")):
            with patch("pickle.load", side_effect=RuntimeError("pickle.load was invoked!")):
                p2_payload = b"\x80\x02}q\x00."
                analyzer = PickleAnalyzer()
                findings = analyzer.analyze_bytes(p2_payload, "1" * 64)
                assert isinstance(findings, list)

    def test_eval_exec_not_called(self) -> None:
        with patch("builtins.eval", side_effect=RuntimeError("eval called")):
            with patch("builtins.exec", side_effect=RuntimeError("exec called")):
                eval_payload = b"cbuiltins\neval\n(S'__import__(\"os\").system(\"id\")'\ntR."
                analyzer = PickleAnalyzer()
                findings = analyzer.analyze_bytes(eval_payload, "2" * 64)
                assert any(f.rule_id == "P101" for f in findings)
