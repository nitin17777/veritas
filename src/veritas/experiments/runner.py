"""VERITAS Experiment Runner.

Executes controlled, reproducible research experiments across:
- RQ1: LLM-only vs. RAG
- RQ2: BM25 vs. Dense vs. Hybrid retrieval
- RQ3: Credibility scoring vs. No credibility vs. Random credibility
- RQ4: Temporal scoring vs. No temporal scoring
- RQ5: Free prompt vs. Evidence-constrained prompt
- RQ6: Held-out topic or dataset generalization
- RQ7: Explanation faithfulness and abstention behavior
- Overall: Full VERITAS system vs. baselines

Guarantees:
- Resumption support: resumes incomplete batches without repeating claims.
- Index and cache reuse across runs.
- Detailed audit logs: captures raw LLM responses, retrieved IDs, stage timings.
- Records all excluded claims and failed calls.
- Enforces test split integrity (prevents tuning on test split).
- Fully compatible with P9 evaluation reporting and automated plotting.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import time
from typing import Any, Iterator

from veritas.config import ROOT_DIR
from veritas.evaluation.report import (
    EvaluationReport,
    build_evaluation_report,
    save_report_artifacts,
)
from veritas.experiments.config import ExperimentConfig, ExperimentConfigError
from veritas.experiments.fixtures import FIXTURE_CLAIMS, build_fixture_retriever
from veritas.pipeline.pipeline import PipelineConfig, Verdict, VeritasPipeline
from veritas.retrieval.hybrid import HybridRetriever
from veritas.verification.llm_client import BaseLLMClient, MockLLMClient

logger = logging.getLogger("veritas.experiments.runner")


class ExperimentRunner:
    """Executes a single VERITAS experiment from an ExperimentConfig.

    Manages:
    - Loading input claims and matching gold labels.
    - Skipping already-completed claims (crash resumption).
    - Excluded claim tracking.
    - Executing pipeline verification.
    - Compiling and exporting evaluation metrics and charts.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        pipeline: VeritasPipeline | None = None,
        retriever: HybridRetriever | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.config = config

        # Set up output directories
        self.results_dir = Path(config.results_dir)
        if not self.results_dir.is_absolute():
            self.results_dir = ROOT_DIR / self.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_path = self.results_dir / "predictions.jsonl"
        self.excluded_path = self.results_dir / "excluded_claims.jsonl"
        self.failed_path = self.results_dir / "failed_calls.jsonl"
        self.report_json_path = self.results_dir / "report.json"
        self.metrics_csv_path = self.results_dir / "metrics.csv"
        self.metadata_path = self.results_dir / "experiment_metadata.json"

        # Initialize or receive pipeline
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            # Check if smoke-test or missing index requires fixture retriever
            effective_retriever = retriever
            if config.smoke_test and effective_retriever is None:
                effective_retriever = build_fixture_retriever()

            # For mock LLM backend, use mock client if none provided
            effective_client = llm_client
            if effective_client is None and config.pipeline.verification.get("backend") == "mock":
                effective_client = MockLLMClient()

            self.pipeline = VeritasPipeline(
                config=config.pipeline,
                retriever=effective_retriever,
                llm_client=effective_client,
            )

    def _load_completed_ids(self) -> set[str]:
        """Read claim IDs from already persisted predictions for crash-resumption."""
        completed: set[str] = set()
        if not self.predictions_path.exists():
            return completed

        with self.predictions_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                    cid = record.get("claim_id") or record.get("id")
                    if cid:
                        completed.add(str(cid).strip())
                except Exception:
                    continue
        logger.info("Found %d completed claims in %s", len(completed), self.predictions_path)
        return completed

    def _log_excluded_claim(self, claim_record: dict[str, Any], reason: str) -> None:
        """Record an unprocessable or excluded claim to audit file."""
        logger.warning("Excluding claim %s: %s", claim_record.get("id", "unknown"), reason)
        record = {
            "claim_record": claim_record,
            "exclusion_reason": reason,
            "experiment_name": self.config.name,
            "timestamp": time.time(),
        }
        with self.excluded_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_failed_call(self, claim_id: str, error: str, raw_output: str | None = None) -> None:
        """Record a failed verifier or model execution to audit file."""
        record = {
            "claim_id": claim_id,
            "error": error,
            "raw_output": raw_output,
            "experiment_name": self.config.name,
            "timestamp": time.time(),
        }
        with self.failed_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_claims_dataset(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load and partition input claims and gold ground-truth records.

        Returns:
            Tuple of (valid_claims, excluded_claims).
        """
        # If in smoke test mode or dataset path does not exist, use fixture claims
        ds_file = Path(self.config.dataset_path)
        if not ds_file.is_absolute():
            ds_file = ROOT_DIR / ds_file

        raw_records: list[dict[str, Any]] = []
        if self.config.smoke_test or not ds_file.exists():
            if not self.config.smoke_test:
                logger.warning(
                    "Dataset %s not found. Falling back to built-in smoke test fixtures.",
                    ds_file,
                )
            raw_records = list(FIXTURE_CLAIMS)
        else:
            with ds_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        rec = json.loads(line_str)
                        if isinstance(rec, dict):
                            raw_records.append(rec)
                    except json.JSONDecodeError as exc:
                        self._log_excluded_claim({"raw_line": line_str}, f"Malformed JSON: {exc}")

        # Limit claims if max_claims is set
        if self.config.max_claims and self.config.max_claims > 0:
            raw_records = raw_records[: self.config.max_claims]

        valid_claims: list[dict[str, Any]] = []
        excluded_claims: list[dict[str, Any]] = []

        for rec in raw_records:
            cid = str(rec.get("id") or rec.get("claim_id") or "").strip()
            claim_text = str(rec.get("claim") or rec.get("text") or "").strip()

            if not cid:
                self._log_excluded_claim(rec, "Missing 'id' / 'claim_id' field.")
                excluded_claims.append(rec)
                continue

            if not claim_text:
                self._log_excluded_claim(rec, "Missing 'claim' / 'text' statement.")
                excluded_claims.append(rec)
                continue

            valid_claims.append(rec)

        return valid_claims, excluded_claims

    def run(self, resume: bool = True) -> EvaluationReport:
        """Execute the experiment run and return the comprehensive evaluation report."""
        logger.info(
            "Starting Experiment '%s' [RQ: %s, Factor: %s=%s]",
            self.config.name,
            self.config.research_question,
            self.config.comparison_factor,
            self.config.factor_value,
        )

        valid_claims, _ = self.load_claims_dataset()
        completed_ids = self._load_completed_ids() if resume else set()

        # If not resuming, remove old predictions file
        if not resume and self.predictions_path.exists():
            self.predictions_path.unlink()

        # Execute claims verification
        predictions: list[dict[str, Any]] = []

        # If resuming, first load existing predictions into the list
        if resume and self.predictions_path.exists():
            with self.predictions_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        try:
                            predictions.append(json.loads(line_str))
                        except Exception:
                            pass

        processed_this_run = 0
        for rec in valid_claims:
            cid = str(rec.get("id") or rec.get("claim_id")).strip()
            if cid in completed_ids:
                logger.debug("Skipping already completed claim %s", cid)
                continue

            claim_text = str(rec.get("claim") or rec.get("text")).strip()
            claim_date = rec.get("claim_date") or rec.get("date")

            try:
                verdict: Verdict = self.pipeline.verify(
                    claim=claim_text,
                    claim_date=claim_date,
                    claim_id=cid,
                    run_id=self.config.name,
                )

                v_dict = verdict.to_dict()
                predictions.append(v_dict)

                # Persist immediately for crash resiliency
                with self.predictions_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(v_dict, ensure_ascii=False) + "\n")

                if verdict.parse_failed:
                    self._log_failed_call(cid, "Verifier JSON parse failed", verdict.raw_llm_output)

                completed_ids.add(cid)
                processed_this_run += 1

            except Exception as exc:
                logger.error("Error evaluating claim %s: %s", cid, exc, exc_info=True)
                self._log_failed_call(cid, str(exc))

        logger.info(
            "Experiment '%s': Completed %d claims (%d processed in this run).",
            self.config.name,
            len(predictions),
            processed_this_run,
        )

        # Match gold records against predictions
        gold_by_id = {
            str(r.get("id") or r.get("claim_id")).strip(): r for r in valid_claims
        }
        matched_gold: list[dict[str, Any]] = []
        matched_preds: list[dict[str, Any]] = []

        for p in predictions:
            pid = str(p.get("claim_id") or p.get("id")).strip()
            if pid in gold_by_id:
                matched_gold.append(gold_by_id[pid])
                matched_preds.append(p)

        # Build EvaluationReport using P9 framework
        report = build_evaluation_report(
            gold_records=matched_gold,
            prediction_records=matched_preds,
            config=self.config.to_dict(),
            model_version=f"veritas_{self.config.name}",
            dataset_name=self.config.dataset_name,
            dataset_version=self.config.dataset_version,
            seed=self.config.seed,
            run_id=self.config.name,
        )

        # Save machine-readable JSON & CSV results
        self.report_json_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        save_report_artifacts(report, output_dir=self.results_dir)

        # Save metadata record
        meta = {
            "name": self.config.name,
            "research_question": self.config.research_question,
            "comparison_factor": self.config.comparison_factor,
            "factor_value": self.config.factor_value,
            "seed": self.config.seed,
            "dataset_version": self.config.dataset_version,
            "split": self.config.split,
            "total_claims": len(valid_claims),
            "evaluated_claims": len(matched_preds),
            "accuracy": report.classification.get("accuracy"),
            "macro_f1": report.classification.get("macro_f1"),
            "faithfulness": report.system_quality.get("faithfulness_score"),
            "abstention_rate": report.system_quality.get("abstention_rate"),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return report
