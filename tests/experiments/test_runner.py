"""Tests for the VERITAS Experiment Runner and Analysis suite.

Tests:
- Configuration parsing, default resolution, and train-split safety check.
- Factor isolation validation (confounding variable detection).
- Crash resumption: verifies skipped claims on re-run.
- Excluded claims logging.
- End-to-end smoke test execution across experimental questions.
- Cross-experiment aggregation, summary table generation, and plotting.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from veritas.experiments.analysis import (
    aggregate_experiment_results,
    export_summary_table_csv,
    generate_comparative_svg_chart,
    generate_markdown_summary_table,
)
from veritas.experiments.config import (
    ExperimentConfig,
    ExperimentConfigError,
    validate_factor_isolation,
)
from veritas.experiments.fixtures import FIXTURE_CLAIMS, build_fixture_retriever
from veritas.experiments.runner import ExperimentRunner
from veritas.pipeline.pipeline import PipelineConfig, VeritasPipeline
from veritas.verification.llm_client import MockLLMClient


class TestExperimentConfig:
    def test_valid_config_from_yaml(self) -> None:
        cfg = ExperimentConfig.from_yaml("configs/experiments/rq1_rag.yaml")
        assert cfg.name == "rq1_rag"
        assert cfg.research_question == "RQ1"
        assert cfg.comparison_factor == "retrieval_mode"
        assert cfg.split == "test"
        assert cfg.pipeline.retrieval_mode == "bm25"

    def test_train_split_rejection(self) -> None:
        """Never tune or evaluate on the training split."""
        raw = {
            "experiment": {"name": "invalid_train_run"},
            "dataset": {"split": "train"},
        }
        with pytest.raises(ExperimentConfigError, match="Never tune or evaluate on the training split"):
            ExperimentConfig.from_dict(raw)

    def test_empty_name_rejection(self) -> None:
        raw = {
            "experiment": {"name": "   "},
            "dataset": {"split": "test"},
        }
        with pytest.raises(ExperimentConfigError, match="Experiment name must not be empty"):
            ExperimentConfig.from_dict(raw)


class TestFactorIsolation:
    def test_pure_factor_difference_passes(self) -> None:
        base = ExperimentConfig.from_yaml("configs/experiments/rq2_bm25.yaml")
        variant = ExperimentConfig.from_yaml("configs/experiments/rq2_dense.yaml")

        violations = validate_factor_isolation(
            base,
            variant,
            allowed_difference_keys=["retrieval_mode", "pipeline.retrieval_mode", "retrieval"],
        )
        assert violations == [], f"Expected zero violations, got: {violations}"

    def test_unintended_change_is_detected(self) -> None:
        base = ExperimentConfig.from_yaml("configs/experiments/rq1_rag.yaml")
        variant = ExperimentConfig.from_yaml("configs/experiments/rq1_rag.yaml")

        # Introduce unintended change in temperature
        variant.name = "rq1_rag_modified"
        variant.pipeline.verification["temperature"] = 0.9

        violations = validate_factor_isolation(
            base,
            variant,
            allowed_difference_keys=["retrieval_mode"],
        )
        assert len(violations) > 0
        assert any("temperature" in v for v in violations)


class TestExperimentRunnerExecution:
    def test_smoke_test_run(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig.from_yaml("configs/experiments/rq1_rag.yaml")
        cfg.results_dir = str(tmp_path / "test_run")
        cfg.smoke_test = True
        cfg.max_claims = 3

        retriever = build_fixture_retriever()
        runner = ExperimentRunner(config=cfg, retriever=retriever)
        report = runner.run(resume=False)

        assert report is not None
        assert report.run_metadata["total_evaluated_claims"] == 3

        # Verify output artifacts exist
        assert (tmp_path / "test_run" / "predictions.jsonl").exists()
        assert (tmp_path / "test_run" / "report.json").exists()
        assert (tmp_path / "test_run" / "experiment_metadata.json").exists()
        assert (tmp_path / "test_run" / "coverage_vs_accuracy.svg").exists()

    def test_resumption_skips_completed_claims(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig.from_yaml("configs/experiments/rq1_rag.yaml")
        results_dir = tmp_path / "resume_run"
        cfg.results_dir = str(results_dir)
        cfg.smoke_test = True
        cfg.max_claims = 4

        retriever = build_fixture_retriever()

        # Pre-populate 2 completed claims
        results_dir.mkdir(parents=True, exist_ok=True)
        pred_file = results_dir / "predictions.jsonl"
        pred_file.write_text(
            json.dumps({"claim_id": "claim_001", "label": "True", "confidence": 0.9}) + "\n" +
            json.dumps({"claim_id": "claim_002", "label": "True", "confidence": 0.9}) + "\n",
            encoding="utf-8",
        )

        runner = ExperimentRunner(config=cfg, retriever=retriever)
        completed_ids = runner._load_completed_ids()
        assert "claim_001" in completed_ids
        assert "claim_002" in completed_ids

        report = runner.run(resume=True)
        # Total claims should be 4
        assert report.run_metadata["total_evaluated_claims"] == 4

        # Read lines in predictions file: should still be exactly 4 (no duplicates)
        lines = [l.strip() for l in pred_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 4
        recorded_ids = [json.loads(l)["claim_id"] for l in lines]
        assert len(recorded_ids) == len(set(recorded_ids))

    def test_excluded_claims_logging(self, tmp_path: Path) -> None:
        bad_dataset = tmp_path / "bad_claims.jsonl"
        bad_dataset.write_text(
            json.dumps({"id": "valid_1", "claim": "Valid claim text."}) + "\n" +
            json.dumps({"id": "missing_text"}) + "\n" +
            json.dumps({"claim": "Missing claim ID."}) + "\n" +
            "not a json line\n",
            encoding="utf-8",
        )

        cfg = ExperimentConfig(
            name="test_exclusions",
            dataset_path=str(bad_dataset),
            results_dir=str(tmp_path / "exclusion_results"),
            smoke_test=False,
        )

        runner = ExperimentRunner(config=cfg)
        valid, excluded = runner.load_claims_dataset()

        assert len(valid) == 1
        assert valid[0]["id"] == "valid_1"
        assert len(excluded) == 2

        # Verify excluded log written
        excl_file = Path(cfg.results_dir) / "excluded_claims.jsonl"
        assert excl_file.exists()
        excl_lines = [l.strip() for l in excl_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(excl_lines) >= 3


class TestAnalysisAndSummary:
    def test_summary_table_generation(self) -> None:
        records = [
            {
                "name": "exp_a",
                "rq": "RQ1",
                "factor": "mode",
                "factor_value": "rag",
                "accuracy": 0.85,
                "macro_f1": 0.82,
                "faithfulness": 0.90,
                "abstention_rate": 0.10,
                "recall_at_5": 0.78,
                "latency_p95_s": 0.05,
            },
            {
                "name": "exp_b",
                "rq": "RQ1",
                "factor": "mode",
                "factor_value": "llm_only",
                "accuracy": 0.60,
                "macro_f1": 0.55,
                "faithfulness": 0.50,
                "abstention_rate": 0.00,
                "recall_at_5": None,
                "latency_p95_s": 0.02,
            },
        ]
        md = generate_markdown_summary_table(records)
        assert "| `exp_a` | RQ1 |" in md
        assert "| `exp_b` | RQ1 |" in md
        assert "0.850" in md

    def test_export_summary_csv_and_svg(self, tmp_path: Path) -> None:
        records = [
            {
                "name": "exp_1",
                "rq": "RQ1",
                "factor": "mode",
                "factor_value": "bm25",
                "accuracy": 0.80,
                "macro_f1": 0.75,
                "faithfulness": 0.85,
                "abstention_rate": 0.05,
                "recall_at_5": 0.70,
                "latency_p95_s": 0.04,
            }
        ]
        csv_file = tmp_path / "summary.csv"
        svg_file = tmp_path / "plot.svg"

        export_summary_table_csv(records, csv_file)
        generate_comparative_svg_chart(records, svg_file)

        assert csv_file.exists()
        assert "exp_1" in csv_file.read_text(encoding="utf-8")
        assert svg_file.exists()
        assert "<svg" in svg_file.read_text(encoding="utf-8")
