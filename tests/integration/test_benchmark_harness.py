"""
tests.integration.test_benchmark_harness
========================================

Integration tests for Phase 11 benchmark evaluation harness and metrics calculation.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.corpus.generator import generate_corpus_data, write_corpus
from benchmarks.harness import (
    check_comparative_baselines,
    generate_results_markdown,
    main,
    run_benchmark,
)
from benchmarks.metrics import (
    BenchmarkSuiteResult,
    calculate_classification_metrics,
    calculate_performance_metrics,
)


class TestBenchmarkMetrics:
    """Test metric calculations and division-by-zero guards."""

    def test_perfect_classification_metrics(self) -> None:
        """Verify metric calculations for 100% precision and recall."""
        m = calculate_classification_metrics(tp=10, tn=10, fp=0, fn=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1_score == 1.0
        assert m.false_positive_rate == 0.0
        assert m.false_negative_rate == 0.0
        assert m.accuracy == 1.0

    def test_imperfect_classification_metrics(self) -> None:
        """Verify metric calculations with false positives and false negatives."""
        m = calculate_classification_metrics(tp=8, tn=8, fp=2, fn=2)
        assert m.precision == 0.8
        assert m.recall == 0.8
        assert m.f1_score == 0.8
        assert m.false_positive_rate == 0.2
        assert m.false_negative_rate == 0.2
        assert m.accuracy == 0.8

    def test_zero_samples_division_by_zero_safety(self) -> None:
        """Verify safe handling of zero samples without ZeroDivisionError."""
        m = calculate_classification_metrics(tp=0, tn=0, fp=0, fn=0)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1_score == 0.0
        assert m.false_positive_rate == 0.0
        assert m.false_negative_rate == 0.0
        assert m.accuracy == 0.0

    def test_performance_metrics_calculation(self) -> None:
        """Verify performance and latency distribution calculations."""
        latencies = [10.0, 20.0, 30.0]
        sizes = [1024 * 1024, 2 * 1024 * 1024, 3 * 1024 * 1024]
        p = calculate_performance_metrics(
            latencies,
            sizes,
            peak_memory_bytes=10 * 1024 * 1024,
        )

        assert p.sample_count == 3
        assert p.mean_latency_ms == 20.0
        assert p.median_latency_ms == 20.0
        assert p.min_latency_ms == 10.0
        assert p.max_latency_ms == 30.0
        assert p.total_bytes_processed == 6 * 1024 * 1024
        assert p.peak_memory_mb == 10.0


class TestCorpusGeneration:
    """Test corpus generation and manifest integrity."""

    def test_generate_corpus_data(self) -> None:
        """Verify synthetic corpus data contains expected samples and manifest fields."""
        files, manifest = generate_corpus_data()
        assert len(files) == 27
        assert len(manifest) == 27

        for entry in manifest:
            assert "case_id" in entry
            assert "filename" in entry
            assert "format" in entry
            assert "expected_classification" in entry
            assert "expected_outcome" in entry
            assert entry["filename"] in files

    def test_write_corpus_to_disk(self, tmp_path: Path) -> None:
        """Verify write_corpus writes all binary files and valid manifest.json."""
        manifest_p = write_corpus(tmp_path)
        assert manifest_p.exists()

        data = json.loads(manifest_p.read_text(encoding="utf-8"))
        assert data["total_samples"] == 27
        assert len(data["samples"]) == 27


class TestBenchmarkHarnessExecution:
    """Test end-to-end benchmark execution and report generation."""

    def test_comparative_baselines_check(self) -> None:
        """Verify comparative baselines checker returns dict with tool statuses."""
        baselines = check_comparative_baselines()
        assert "modelscan" in baselines
        assert "picklescan" in baselines
        assert "installed" in baselines["modelscan"]
        assert "status" in baselines["modelscan"]

    def test_run_benchmark_end_to_end(self, tmp_path: Path) -> None:
        """Verify benchmark runs through all samples without errors."""
        result = run_benchmark(corpus_dir=tmp_path, warmup=False)
        assert isinstance(result, BenchmarkSuiteResult)
        assert result.total_samples == 27
        assert result.overall_classification.precision > 0.0
        assert result.overall_classification.recall > 0.0
        assert len(result.sample_records) == 27

    def test_generate_results_markdown(self, tmp_path: Path) -> None:
        """Verify markdown report generation produces expected sections."""
        result = run_benchmark(corpus_dir=tmp_path, warmup=False)
        md = generate_results_markdown(result)

        assert "# Static Security Scanner — Benchmark Evaluation Results" in md
        assert "Executive Summary & Core Detection Metrics" in md
        assert "Performance, Latency & Throughput" in md
        assert "Format-by-Format Breakdown" in md
        assert "Comparative Tool Baselines" in md
        assert "Sample-by-Sample Validation Matrix" in md

    def test_harness_cli_main(self, tmp_path: Path) -> None:
        """Verify running harness CLI creates results.json and RESULTS.md."""
        corpus_dir = tmp_path / "corpus"
        json_out = tmp_path / "results.json"
        md_out = tmp_path / "RESULTS.md"

        exit_code = main([
            "--corpus-dir", str(corpus_dir),
            "--output-json", str(json_out),
            "--output-md", str(md_out),
            "--quiet",
        ])

        assert exit_code == 0
        assert json_out.exists()
        assert md_out.exists()
