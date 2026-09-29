"""
benchmarks.metrics
==================

Deterministic calculation of detection metrics and performance statistics
for the static security scanner evaluation harness (Phase 11).

METRICS COMPUTED
----------------
- Confusion Matrix: TP, TN, FP, FN
- Detection Rates: Precision, Recall (TPR), F1 Score,
  False Positive Rate (FPR), False Negative Rate (FNR)
- Latency Statistics: Mean, Median, Min, Max (in milliseconds)
- Throughput: Total bytes / total time (in MB/s)
- Memory: Peak heap allocation (in MB)
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Standard binary classification performance metrics.

    Positive = Malicious / Security Anomaly
    Negative = Clean / Benign
    """

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return asdict(self)


def calculate_classification_metrics(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> BinaryClassificationMetrics:
    """Compute precision, recall, F1, FPR, FNR, and accuracy with safe division."""
    total_positives = tp + fn
    total_negatives = tn + fp
    total_samples = tp + tn + fp + fn

    # Precision = TP / (TP + FP)
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # Recall (TPR) = TP / (TP + FN)
    recall = (tp / total_positives) if total_positives > 0 else 0.0

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    # False Positive Rate = FP / (FP + TN)
    fpr = (fp / total_negatives) if total_negatives > 0 else 0.0

    # False Negative Rate = FN / (FN + TP)
    fnr = (fn / total_positives) if total_positives > 0 else 0.0

    # Accuracy = (TP + TN) / Total
    accuracy = ((tp + tn) / total_samples) if total_samples > 0 else 0.0

    return BinaryClassificationMetrics(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        false_negative_rate=round(fnr, 4),
        accuracy=round(accuracy, 4),
    )


@dataclass(frozen=True)
class PerformanceMetrics:
    """Latency and throughput performance statistics."""

    sample_count: int
    total_duration_ms: float
    mean_latency_ms: float
    median_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_bytes_processed: int
    throughput_mb_s: float
    peak_memory_mb: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return asdict(self)


def calculate_performance_metrics(
    latencies_ms: list[float],
    file_sizes_bytes: list[int],
    peak_memory_bytes: int = 0,
) -> PerformanceMetrics:
    """Compute latency distributions and throughput in MB/s."""
    sample_count = len(latencies_ms)
    if sample_count == 0:
        return PerformanceMetrics(
            sample_count=0,
            total_duration_ms=0.0,
            mean_latency_ms=0.0,
            median_latency_ms=0.0,
            min_latency_ms=0.0,
            max_latency_ms=0.0,
            total_bytes_processed=0,
            throughput_mb_s=0.0,
            peak_memory_mb=0.0,
        )

    total_duration_ms = sum(latencies_ms)
    mean_lat = statistics.mean(latencies_ms)
    median_lat = statistics.median(latencies_ms)
    min_lat = min(latencies_ms)
    max_lat = max(latencies_ms)

    total_bytes = sum(file_sizes_bytes)
    total_sec = total_duration_ms / 1000.0
    throughput = (
        ((total_bytes / (1024 * 1024)) / total_sec)
        if total_sec > 0
        else 0.0
    )
    peak_mem_mb = peak_memory_bytes / (1024 * 1024)

    return PerformanceMetrics(
        sample_count=sample_count,
        total_duration_ms=round(total_duration_ms, 2),
        mean_latency_ms=round(mean_lat, 2),
        median_latency_ms=round(median_lat, 2),
        min_latency_ms=round(min_lat, 2),
        max_latency_ms=round(max_lat, 2),
        total_bytes_processed=total_bytes,
        throughput_mb_s=round(throughput, 2),
        peak_memory_mb=round(peak_mem_mb, 2),
    )


@dataclass
class SampleEvaluationRecord:
    """Individual benchmark test sample result and validation."""

    case_id: str
    filename: str
    format: str
    expected_label: str  # BENIGN, MALICIOUS, UNSUPPORTED, ERROR
    expected_rules: list[str]
    actual_outcome: str  # COMPLETE, WARNING, ERROR, UNSUPPORTED
    actual_format: str
    final_score: int
    risk_band: str
    findings_count: int
    detected_rules: list[str]
    duration_ms: float
    file_size_bytes: int
    classification_match: bool
    rules_matched: bool
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert record to serializable dictionary."""
        return asdict(self)


@dataclass
class BenchmarkSuiteResult:
    """Complete aggregated benchmark report."""

    timestamp_utc: str
    scanner_version: str
    environment: dict[str, str]
    total_samples: int
    overall_classification: BinaryClassificationMetrics
    performance: PerformanceMetrics
    by_format: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    sample_records: list[SampleEvaluationRecord] = field(default_factory=list)
    comparative_baselines: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert suite result to serializable dictionary."""
        return {
            "timestamp_utc": self.timestamp_utc,
            "scanner_version": self.scanner_version,
            "environment": self.environment,
            "total_samples": self.total_samples,
            "overall_classification": self.overall_classification.to_dict(),
            "performance": self.performance.to_dict(),
            "by_format": self.by_format,
            "by_category": self.by_category,
            "sample_records": [r.to_dict() for r in self.sample_records],
            "comparative_baselines": self.comparative_baselines,
        }


__all__ = [
    "BenchmarkSuiteResult",
    "BinaryClassificationMetrics",
    "PerformanceMetrics",
    "SampleEvaluationRecord",
    "calculate_classification_metrics",
    "calculate_performance_metrics",
]
