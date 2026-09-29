"""
benchmarks.harness
==================

Evaluation harness for executing reproducible benchmark suites against the
Static Security Scanner (Phase 11).

SECURITY GUARANTEE
------------------
- Zero dynamic model execution: uses model_scanner.api.scan() strictly.
- Never imports ML frameworks (no torch, transformers, llama.cpp, etc.).
- Never calls eval(), exec(), pickle.loads(), or torch.load().

Design Reference
----------------
Section 31 of docs/implementation_design.md (Phase 11)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.corpus.generator import write_corpus
from benchmarks.metrics import (
    BenchmarkSuiteResult,
    SampleEvaluationRecord,
    calculate_classification_metrics,
    calculate_performance_metrics,
)
from model_scanner import __version__
from model_scanner.api import scan
from model_scanner.core.models import ScanOutcome


def check_comparative_baselines() -> dict[str, Any]:
    """Check availability of external comparative baseline tools in the environment."""
    baselines: dict[str, Any] = {}

    # Check ModelScan
    modelscan_spec = importlib.util.find_spec("modelscan")
    if modelscan_spec is not None:
        baselines["modelscan"] = {
            "installed": True,
            "status": "available",
            "notes": "ModelScan package is available in the current environment.",
        }
    else:
        baselines["modelscan"] = {
            "installed": False,
            "status": "not available in environment",
            "notes": "ModelScan is not installed in the current environment.",
        }

    # Check Picklescan
    picklescan_spec = importlib.util.find_spec("picklescan")
    if picklescan_spec is not None:
        baselines["picklescan"] = {
            "installed": True,
            "status": "available",
            "notes": "Picklescan package is available in the current environment.",
        }
    else:
        baselines["picklescan"] = {
            "installed": False,
            "status": "not available in environment",
            "notes": "Picklescan is not installed in the current environment.",
        }

    return baselines


def run_benchmark(
    corpus_dir: Path | None = None,
    warmup: bool = True,
) -> BenchmarkSuiteResult:
    """Execute the complete static security scanner benchmark suite.

    Parameters
    ----------
    corpus_dir:
        Directory containing synthetic corpus files and manifest.json.
        If None or non-existent, the corpus will be automatically generated.
    warmup:
        Whether to perform a warmup pass to eliminate one-time JIT/module load overhead.

    Returns
    -------
    BenchmarkSuiteResult
        Aggregated evaluation results including classification metrics and performance stats.
    """
    if corpus_dir is None:
        corpus_dir = Path(__file__).parent / "corpus" / "data"

    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = write_corpus(corpus_dir)

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = manifest_data.get("samples", [])

    # Optional Warmup pass
    if warmup and samples:
        for sample in samples[:3]:
            warmup_file = corpus_dir / sample["filename"]
            if warmup_file.exists():
                try:
                    scan(warmup_file)
                except Exception as exc:
                    # Warmup failures on corrupted samples are non-fatal
                    _ = str(exc)

    # Start memory tracing
    tracemalloc.start()
    tracemalloc.reset_peak()

    latencies_ms: list[float] = []
    file_sizes_bytes: list[int] = []
    sample_records: list[SampleEvaluationRecord] = []

    # Confusion matrix counters for binary classification (Malicious vs Benign)
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for sample in samples:
        filename = sample["filename"]
        file_path = corpus_dir / filename
        expected_label = sample["expected_classification"]
        expected_rules = sample.get("expected_rules", [])

        if not file_path.exists():
            continue

        file_size = file_path.stat().st_size
        file_sizes_bytes.append(file_size)

        # High-resolution execution timing
        t_start = time.perf_counter_ns()
        result = scan(file_path)
        t_end = time.perf_counter_ns()

        duration_ms = (t_end - t_start) / 1_000_000.0
        latencies_ms.append(duration_ms)

        # Extract artifact attributes
        if result.artifacts:
            art = result.artifacts[0]
            actual_outcome = art.scan_outcome.value
            actual_format = art.detected_format.value
            final_score = art.risk_score.final_score
            risk_band = art.risk_score.risk_band.value
            findings_count = len(art.findings)
            detected_rules = [f.rule_id for f in art.findings]
        else:
            actual_outcome = result.summary.overall_outcome.value
            actual_format = "UNKNOWN"
            final_score = result.summary.max_risk_score
            risk_band = "UNKNOWN"
            findings_count = 0
            detected_rules = []

        # Determine classification match
        classification_match = False
        is_malicious_detected = (
            findings_count > 0
            and any(
                r not in ("P209",)  # P209 is informational container metadata
                for r in detected_rules
            )
        ) or final_score > 0

        if expected_label == "BENIGN":
            classification_match = (
                not is_malicious_detected
                and actual_outcome == ScanOutcome.COMPLETE.value
            )
            if not is_malicious_detected:
                tn += 1
            else:
                fp += 1
        elif expected_label == "MALICIOUS":
            classification_match = is_malicious_detected
            if is_malicious_detected:
                tp += 1
            else:
                fn += 1
        elif expected_label == "UNSUPPORTED":
            classification_match = actual_outcome == ScanOutcome.UNSUPPORTED.value
        elif expected_label == "ERROR":
            classification_match = actual_outcome == ScanOutcome.ERROR.value

        # Check rule match
        rules_matched = True
        for er in expected_rules:
            if er not in detected_rules:
                rules_matched = False
                break

        record = SampleEvaluationRecord(
            case_id=sample["case_id"],
            filename=filename,
            format=sample["format"],
            expected_label=expected_label,
            expected_rules=expected_rules,
            actual_outcome=actual_outcome,
            actual_format=actual_format,
            final_score=final_score,
            risk_band=risk_band,
            findings_count=findings_count,
            detected_rules=detected_rules,
            duration_ms=round(duration_ms, 3),
            file_size_bytes=file_size,
            classification_match=classification_match,
            rules_matched=rules_matched,
            details=sample.get("description", ""),
        )
        sample_records.append(record)

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    classification_metrics = calculate_classification_metrics(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )
    performance_metrics = calculate_performance_metrics(
        latencies_ms=latencies_ms,
        file_sizes_bytes=file_sizes_bytes,
        peak_memory_bytes=peak_memory,
    )

    # Format-by-Format Breakdown
    by_format: dict[str, dict[str, Any]] = {}
    for fmt in ("RAW_PICKLE", "PYTORCH_ZIP", "SAFETENSORS", "GGUF", "UNSUPPORTED"):
        fmt_records = [r for r in sample_records if r.format == fmt]
        if fmt_records:
            fmt_latencies = [r.duration_ms for r in fmt_records]
            fmt_sizes = [r.file_size_bytes for r in fmt_records]
            fmt_matched = sum(1 for r in fmt_records if r.classification_match)
            by_format[fmt] = {
                "sample_count": len(fmt_records),
                "accuracy": round(fmt_matched / len(fmt_records), 4),
                "mean_latency_ms": round(sum(fmt_latencies) / len(fmt_latencies), 2),
                "total_bytes": sum(fmt_sizes),
            }

    # Category Breakdown
    by_category: dict[str, dict[str, Any]] = {}
    categories = {s.get("category", "unknown") for s in samples}
    for cat in sorted(categories):
        cat_samples = [s for s in samples if s.get("category") == cat]
        cat_case_ids = {s["case_id"] for s in cat_samples}
        cat_records = [r for r in sample_records if r.case_id in cat_case_ids]
        if cat_records:
            cat_matched = sum(1 for r in cat_records if r.classification_match)
            by_category[cat] = {
                "sample_count": len(cat_records),
                "accuracy": round(cat_matched / len(cat_records), 4),
                "mean_latency_ms": round(
                    sum(r.duration_ms for r in cat_records) / len(cat_records), 2
                ),
            }

    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "Unknown",
        "cpu_count": str(os.cpu_count() or 1),
    }

    return BenchmarkSuiteResult(
        timestamp_utc=datetime.now(UTC).isoformat(),
        scanner_version=__version__,
        environment=environment,
        total_samples=len(sample_records),
        overall_classification=classification_metrics,
        performance=performance_metrics,
        by_format=by_format,
        by_category=by_category,
        sample_records=sample_records,
        comparative_baselines=check_comparative_baselines(),
    )


def generate_results_markdown(result: BenchmarkSuiteResult) -> str:
    """Generate canonical comprehensive RESULTS.md markdown report."""
    cls = result.overall_classification
    perf = result.performance

    lines: list[str] = [
        "# Static Security Scanner — Benchmark Evaluation Results",
        "",
        f"**Date (UTC):** `{result.timestamp_utc}`  ",
        f"**Scanner Version:** `{result.scanner_version}`  ",
        "**Evaluation Framework:** Phase 11 Standard Benchmark Harness  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Core Detection Metrics",
        "",
        "| Metric | Measured Value | Standard Formula / Definition |",
        "| :--- | :--- | :--- |",
        f"| **True Positives (TP)** | `{cls.true_positives}` | "
        "Correctly identified malicious/anomalous models |",
        f"| **True Negatives (TN)** | `{cls.true_negatives}` | "
        "Correctly identified clean/benign models |",
        f"| **False Positives (FP)** | `{cls.false_positives}` | "
        "Benign models incorrectly flagged as malicious |",
        f"| **False Negatives (FN)** | `{cls.false_negatives}` | "
        "Malicious models missed by scanner |",
        f"| **Precision** | **`{cls.precision:.4f}`** | $TP / (TP + FP)$ |",
        f"| **Recall (TPR)** | **`{cls.recall:.4f}`** | $TP / (TP + FN)$ |",
        f"| **F1 Score** | **`{cls.f1_score:.4f}`** | "
        r"$2 \cdot (Precision \cdot Recall) / (Precision + Recall)$ |",
        f"| **False Positive Rate (FPR)** | **`{cls.false_positive_rate:.4f}`** | "
        r"$FP / (FP + TN)$ |",
        f"| **False Negative Rate (FNR)** | **`{cls.false_negative_rate:.4f}`** | "
        r"$FN / (FN + TP)$ |",
        f"| **Overall Accuracy** | **`{cls.accuracy:.4f}`** | $(TP + TN) / Total$ |",
        "",
        "---",
        "",
        "## 2. Performance, Latency & Throughput",
        "",
        "| Performance Characteristic | Measured Value |",
        "| :--- | :--- |",
        f"| **Total Samples Evaluated** | `{perf.sample_count}` |",
        f"| **Total Scan Time** | `{perf.total_duration_ms:.2f} ms` |",
        f"| **Mean Detection Latency** | `{perf.mean_latency_ms:.2f} ms` |",
        f"| **Median Detection Latency** | `{perf.median_latency_ms:.2f} ms` |",
        f"| **Min Latency** | `{perf.min_latency_ms:.2f} ms` |",
        f"| **Max Latency** | `{perf.max_latency_ms:.2f} ms` |",
        f"| **Total Bytes Processed** | `{perf.total_bytes_processed:,} bytes` |",
        f"| **Throughput** | `{perf.throughput_mb_s:.2f} MB/s` |",
        f"| **Peak Traced Heap Memory** | `{perf.peak_memory_mb:.2f} MB` |",
        "",
        "---",
        "",
        "## 3. Format-by-Format Breakdown",
        "",
        "| Model Format | Samples | Accuracy | Mean Latency | Total Data |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for fmt, stats in sorted(result.by_format.items()):
        lines.append(
            f"| `{fmt}` | `{stats['sample_count']}` | "
            f"`{stats['accuracy'] * 100:.1f}%` | "
            f"`{stats['mean_latency_ms']:.2f} ms` | "
            f"`{stats['total_bytes']:,} B` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Category Breakdown",
        "",
        "| Evaluation Category | Samples | Accuracy | Mean Latency |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for cat, stats in sorted(result.by_category.items()):
        lines.append(
            f"| `{cat}` | `{stats['sample_count']}` | "
            f"`{stats['accuracy'] * 100:.1f}%` | "
            f"`{stats['mean_latency_ms']:.2f} ms` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Comparative Tool Baselines",
        "",
        "| Tool | Status in Environment | Notes |",
        "| :--- | :--- | :--- |",
    ])

    for tool_name, info in sorted(result.comparative_baselines.items()):
        lines.append(
            f"| **{tool_name.capitalize()}** | `{info['status']}` | {info['notes']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Sample-by-Sample Validation Matrix",
        "",
        "| Case ID | Format | Expected | Outcome | Score | Band | "
        "Detected Rules | Duration | Match |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for r in result.sample_records:
        rules_str = ", ".join(r.detected_rules) if r.detected_rules else "-"
        status_icon = "PASS" if r.classification_match else "FAIL"
        lines.append(
            f"| `{r.case_id}` | `{r.format}` | `{r.expected_label}` | "
            f"`{r.actual_outcome}` | `{r.final_score}` | `{r.risk_band}` | "
            f"`{rules_str}` | `{r.duration_ms:.2f} ms` | **{status_icon}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 7. Methodology, Environment & Limitations",
        "",
        "### 7.1 Security Guarantee",
        "- **Zero Dynamic Execution:** All evaluation runs statically via "
        "`model_scanner.api.scan()`. No ML execution environments, frameworks "
        "(`torch`, `transformers`, `llama.cpp`), or dynamic `eval()`/`exec()` "
        "calls are invoked.",
        "",
        "### 7.2 Measurement Environment",
        f"- **Python Version:** `{result.environment.get('python_version', 'Unknown')}`",
        f"- **Platform OS:** `{result.environment.get('platform', 'Unknown')}`",
        f"- **Processor:** `{result.environment.get('processor', 'Unknown')}`",
        f"- **Logical CPU Cores:** `{result.environment.get('cpu_count', 'Unknown')}`",
        "",
        "### 7.3 Limitations",
        "- The synthetic corpus consists of deterministic in-memory fixtures "
        "designed to validate every specific detection rule without exposing the "
        "test environment to real weaponized payloads.",
        "- High throughput rates reflect in-memory file I/O on synthetic models; "
        "large multi-gigabyte models will be bounded primarily by physical disk I/O.",
        "- Memory measurement is performed via standard library `tracemalloc` "
        "capturing peak Python heap allocation during the scan cycle.",
        "",
    ])

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running benchmarks."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.harness",
        description="Run static security scanner benchmark evaluation harness.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="Path to corpus directory (default: benchmarks/corpus/data).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to output results.json (default: benchmarks/results.json).",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Path to output RESULTS.md (default: benchmarks/RESULTS.md).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress console output.",
    )

    args = parser.parse_args(argv)

    benchmarks_dir = Path(__file__).parent
    corpus_dir = (
        Path(args.corpus_dir)
        if args.corpus_dir
        else benchmarks_dir / "corpus" / "data"
    )
    output_json = (
        Path(args.output_json)
        if args.output_json
        else benchmarks_dir / "results.json"
    )
    output_md = (
        Path(args.output_md)
        if args.output_md
        else benchmarks_dir / "RESULTS.md"
    )

    if not args.quiet:
        print(
            f"\n============================================================\n"
            f" Model Scanner Benchmark Evaluation Harness (Phase 11)\n"
            f" Version: {__version__}\n"
            f" Corpus: {corpus_dir}\n"
            f"============================================================\n",
            flush=True,
        )

    result = run_benchmark(corpus_dir=corpus_dir)

    # Save results.json
    output_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    # Save RESULTS.md
    md_content = generate_results_markdown(result)
    output_md.write_text(md_content, encoding="utf-8")

    if not args.quiet:
        cls = result.overall_classification
        perf = result.performance
        print(
            f"Evaluation Complete:\n"
            f"  Total Samples : {result.total_samples}\n"
            f"  Precision     : {cls.precision:.4f}\n"
            f"  Recall (TPR)  : {cls.recall:.4f}\n"
            f"  F1 Score      : {cls.f1_score:.4f}\n"
            f"  FPR / FNR     : {cls.false_positive_rate:.4f} / "
            f"{cls.false_negative_rate:.4f}\n"
            f"  Accuracy      : {cls.accuracy * 100:.2f}%\n"
            f"  Mean Latency  : {perf.mean_latency_ms:.2f} ms\n"
            f"  Throughput    : {perf.throughput_mb_s:.2f} MB/s\n"
            f"  Peak Memory   : {perf.peak_memory_mb:.2f} MB\n"
            f"\nArtifacts saved:\n"
            f"  - JSON Report : {output_json.resolve()}\n"
            f"  - Markdown Doc: {output_md.resolve()}\n",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
