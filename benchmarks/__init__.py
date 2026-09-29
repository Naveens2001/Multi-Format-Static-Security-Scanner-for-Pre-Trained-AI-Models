"""
benchmarks
==========

Static Security Scanner Evaluation Benchmark Suite (Phase 11).

Provides reproducible evaluation metrics, synthetic model corpus,
and comparative baseline checks for static AI model scanning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks.metrics import (
    BenchmarkSuiteResult,
    BinaryClassificationMetrics,
    PerformanceMetrics,
    SampleEvaluationRecord,
    calculate_classification_metrics,
    calculate_performance_metrics,
)

if TYPE_CHECKING:
    from benchmarks.harness import run_benchmark


def __getattr__(name: str) -> object:
    """Resolve public benchmark runner lazily to prevent runtime warnings."""
    if name == "run_benchmark":
        from benchmarks.harness import run_benchmark

        return run_benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BenchmarkSuiteResult",
    "BinaryClassificationMetrics",
    "PerformanceMetrics",
    "SampleEvaluationRecord",
    "calculate_classification_metrics",
    "calculate_performance_metrics",
    "run_benchmark",
]
