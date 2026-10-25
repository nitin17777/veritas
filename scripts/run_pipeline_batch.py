#!/usr/bin/env python
"""CLI script for running the VERITAS P8 pipeline on a batch of claims.

Usage
-----
# Basic run (BM25 index must already exist):
python scripts/run_pipeline_batch.py \\
    --input  data/claims.jsonl \\
    --output results/verdicts.jsonl \\
    --config configs/pipeline.yaml

# Resume a crashed run (same command – already-done claims are skipped):
python scripts/run_pipeline_batch.py \\
    --input  data/claims.jsonl \\
    --output results/verdicts.jsonl \\
    --config configs/pipeline.yaml

# Quick smoke-test with 3 built-in synthetic claims and a mock LLM:
python scripts/run_pipeline_batch.py --smoke-test

Input JSONL format
------------------
Each line must be a JSON object with at minimum a "claim" field.
Optional fields: "id" (or "claim_id"), "claim_date" (YYYY-MM-DD).

Example:
    {"id": "c1", "claim": "Earth orbits the Sun.", "claim_date": "2024-01-01"}
    {"id": "c2", "claim": "The Moon is made of cheese."}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the src package is importable when the script is run directly.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from veritas.pipeline.batch import BatchRunner, run_batch
from veritas.pipeline.pipeline import PipelineConfig, VeritasPipeline


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _smoke_test() -> None:
    """Run 3 synthetic claims through a fully-in-memory mock pipeline."""
    from veritas.retrieval.bm25 import BM25Index
    from veritas.retrieval.corpus import PassageStore
    from veritas.retrieval.hybrid import HybridRetriever
    from veritas.retrieval.schema import Passage
    from veritas.verification.llm_client import MockLLMClient

    print("\n=== VERITAS Pipeline Smoke-Test ===\n")

    passages = [
        Passage("p1", "The Earth completes one orbit around the Sun every 365.25 days.", domain="nasa.gov"),
        Passage("p2", "The Moon has a rocky surface covered in craters; it is not edible.", domain="nasa.gov"),
        Passage("p3", "Mount Everest is the highest peak above sea level at 8,849 m.", domain="nationalgeographic.com"),
    ]
    store = PassageStore(passages)
    bm25_idx = BM25Index.build(store)
    retriever = HybridRetriever(bm25_index=bm25_idx, mode="bm25", bm25_top_k=3, top_k=3)

    canned = [
        {"label": "True", "confidence": 0.97, "evidence_ids": ["p1"],
         "explanation": "p1 directly confirms Earth's orbital period."},
        {"label": "False", "confidence": 0.94, "evidence_ids": ["p2"],
         "explanation": "p2 states the Moon is rocky, contradicting the claim."},
        {"label": "True", "confidence": 0.91, "evidence_ids": ["p3"],
         "explanation": "p3 confirms the height of Everest."},
    ]
    client = MockLLMClient(canned_responses=canned)

    cfg = PipelineConfig(
        retrieval_mode="bm25",
        retrieval_top_k=3,
        save_verdicts=False,
        verification={"backend": "mock", "save_traces": False},
    )
    pipeline = VeritasPipeline(config=cfg, retriever=retriever, llm_client=client)

    claims = [
        ("Earth orbits the Sun once every year.", None),
        ("The Moon is made of cheese.", None),
        ("Mount Everest is the tallest mountain.", "2024-03-01"),
    ]

    for claim, date in claims:
        v = pipeline.verify(claim=claim, claim_date=date)
        print(f"Claim   : {claim!r}")
        print(f"Label   : {v.label}  (confidence={v.confidence:.2f})")
        print(f"Evidence: {v.evidence_ids}")
        print(f"Timing  : {v.stage_timings_s}")
        print()

    print("=== Smoke-test passed ===\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VERITAS P8 batch pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", help="Path to input JSONL file of claims.")
    parser.add_argument("--output", "-o", help="Path for output JSONL file of verdicts.")
    parser.add_argument("--config", "-c", default="configs/pipeline.yaml",
                        help="Path to YAML configuration (default: configs/pipeline.yaml).")
    parser.add_argument("--run-id", default=None, help="Optional run identifier.")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a 3-claim in-memory smoke test and exit.")
    args = parser.parse_args()

    _setup_logging(args.log_level)

    if args.smoke_test:
        _smoke_test()
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --smoke-test is passed.")

    ckpt = run_batch(
        input_path=args.input,
        output_path=args.output,
        config=args.config if Path(args.config).is_file() else None,
        run_id=args.run_id,
    )
    print(f"\nBatch complete: {ckpt.done} done, {ckpt.failed} failed, {ckpt.skipped} skipped.")
    print(f"Verdicts saved to: {ckpt.output_path}")
    print(f"Checkpoint:        {ckpt.output_path}.ckpt.json")


if __name__ == "__main__":
    main()
