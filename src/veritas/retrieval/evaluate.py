"""Evaluation script and harness for VERITAS evidence retrieval.

Evaluates BM25, Dense, or Hybrid retrieval against claims with gold evidence.
Enforces the requirement:
- Computes Recall@k and nDCG@k strictly when gold evidence is present.
- Never reports retrieval metrics when gold evidence is missing.
- Saves evaluation results and run configuration to output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from veritas.config import ROOT_DIR, load_config
from veritas.retrieval.bm25 import BM25Index
from veritas.retrieval.corpus import PassageStore, create_synthetic_corpus
from veritas.retrieval.dense import DenseEncoder, DenseIndex, MockEmbedder
from veritas.retrieval.hybrid import HybridRetriever
from veritas.retrieval.metrics import evaluate_retrieval_predictions
from veritas.retrieval.schema import Passage

logger = logging.getLogger("veritas.retrieval.evaluate")


def run_retrieval_evaluation(
    claims_file: str | Path | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    mode: str = "hybrid",
    bm25_top_k: int = 50,
    dense_top_k: int = 50,
    rrf_k: int = 60,
    top_k: int = 50,
    k_values: list[int] | None = None,
    retriever: HybridRetriever | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Execute retrieval evaluation across claims and persist metrics."""
    if k_values is None:
        k_values = [1, 5, 10, 20, 50]

    # Resolve output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_dir is None:
        target_out = ROOT_DIR / "results" / f"retrieval_eval_{timestamp}"
    else:
        target_out = Path(output_dir)
        if not target_out.is_absolute():
            target_out = ROOT_DIR / target_out

    target_out.mkdir(parents=True, exist_ok=True)

    if smoke_test:
        logger.info("Executing retrieval evaluation in SMOKE-TEST mode with synthetic records...")
        # Create small test corpus and claims with known gold evidence
        p1 = Passage(passage_id="psg:cdc_flu", text="Annual influenza vaccination prevents severe respiratory disease and hospitalization.")
        p2 = Passage(passage_id="psg:bls_jobs", text="Total nonfarm payroll employment increased by 275000 in February across health care and government.")
        p3 = Passage(passage_id="psg:nasa_jwst", text="The James Webb Space Telescope detected atmospheric methane on a habitable-zone exoplanet.")

        store = PassageStore([p1, p2, p3])
        bm25_idx = BM25Index.build(store)
        mock_enc = DenseEncoder(model_name_or_path="mock-embedder", device="cpu", embedder=MockEmbedder(dimension=16))
        dense_idx = DenseIndex.build(store, encoder=mock_enc)

        retriever = HybridRetriever(
            bm25_index=bm25_idx,
            dense_index=dense_idx,
            mode=mode,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
            rrf_k=rrf_k,
            top_k=top_k,
        )

        claims = [
            # Claim with gold evidence
            {
                "claim_id": "claim_test_1",
                "queries": ["influenza vaccination prevents respiratory hospitalization"],
                "gold_ids": ["psg:cdc_flu"],
            },
            # Claim with gold evidence
            {
                "claim_id": "claim_test_2",
                "queries": ["nonfarm payroll employment increased in February"],
                "gold_ids": ["psg:bls_jobs"],
            },
            # Claim WITHOUT gold evidence (tests exclusion rule)
            {
                "claim_id": "claim_test_missing_gold",
                "queries": ["unverified rumor without any evidence in benchmark"],
                "gold_ids": [],
            },
        ]
    else:
        if retriever is None:
            retriever = HybridRetriever.from_config(
                config=config_path or "configs/default.yaml",
                mode=mode,
            )

        if claims_file is None:
            raise ValueError("claims_file must be specified when not in smoke_test mode.")

        claims_path = Path(claims_file)
        if not claims_path.is_absolute():
            claims_path = ROOT_DIR / claims_path

        claims = []
        with claims_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                c_id = str(rec.get("id", "unknown"))
                statement = rec.get("claim", "")
                # Queries: allow decomposed queries from metadata if present, else claim statement
                queries = rec.get("metadata", {}).get("queries") or [statement]
                # Gold evidence IDs:
                gold_ids = rec.get("gold_passage_ids")
                if gold_ids is None and "evidence" in rec and rec["evidence"]:
                    # Default: passage IDs generated for this claim's evidence
                    gold_ids = [f"{c_id}#e{i}" for i in range(len(rec["evidence"]))]
                elif gold_ids is None:
                    gold_ids = []

                claims.append({
                    "claim_id": c_id,
                    "queries": queries,
                    "gold_ids": gold_ids,
                })

    logger.info("Evaluating retrieval over %d claims (mode=%s)...", len(claims), mode)

    eval_instances: list[dict[str, Any]] = []
    for c in claims:
        retrieved_results = retriever.retrieve(c["queries"], top_k=top_k)
        retrieved_ids = [r.passage_id for r in retrieved_results]
        eval_instances.append({
            "claim_id": c["claim_id"],
            "retrieved_ids": retrieved_ids,
            "gold_ids": c["gold_ids"],
        })

    # Compute metrics respecting missing gold evidence constraint
    eval_report = evaluate_retrieval_predictions(eval_instances, k_values=k_values)

    eval_config = {
        "timestamp": timestamp,
        "mode": mode,
        "bm25_top_k": bm25_top_k,
        "dense_top_k": dense_top_k,
        "rrf_k": rrf_k,
        "top_k": top_k,
        "k_values": k_values,
        "claims_file": str(claims_file) if claims_file else "synthetic_fixture",
        "smoke_test": smoke_test,
    }

    # Save results
    metrics_file = target_out / "retrieval_metrics.json"
    config_file = target_out / "evaluation_config.json"

    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    with config_file.open("w", encoding="utf-8") as f:
        json.dump(eval_config, f, indent=2)

    logger.info("Saved evaluation report to: %s", metrics_file)
    logger.info("Saved evaluation configuration to: %s", config_file)
    if eval_report.get("metrics_reported"):
        logger.info("Mean Retrieval Metrics:")
        for m_name, val in eval_report.get("mean_metrics", {}).items():
            logger.info("  %-12s: %.4f", m_name, val)
    else:
        logger.info("Metrics not reported: %s", eval_report.get("reason"))

    return {
        "output_dir": str(target_out),
        "metrics": eval_report,
        "config": eval_config,
    }


def main() -> None:
    """CLI entrypoint for retrieval evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate VERITAS retrieval (BM25, Dense, or Hybrid RRF).")
    parser.add_argument("--claims-file", type=str, default=None, help="Path to processed JSONL claims file.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML configuration.")
    parser.add_argument("--output-dir", type=str, default=None, help="Results output directory.")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "bm25", "dense"], help="Retrieval mode.")
    parser.add_argument("--bm25-top-k", type=int, default=50, help="BM25 candidate count (default 50).")
    parser.add_argument("--dense-top-k", type=int, default=50, help="Dense candidate count (default 50).")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant k (default 60).")
    parser.add_argument("--top-k", type=int, default=50, help="Final retrieved count (default 50).")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast evaluation on synthetic fixture.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")

    run_retrieval_evaluation(
        claims_file=args.claims_file,
        config_path=args.config,
        output_dir=args.output_dir,
        mode=args.mode,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        rrf_k=args.rrf_k,
        top_k=args.top_k,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
