"""Unified BM25 Index Build Command for VERITAS.

Extracts evidence passages from processed datasets, chunks documents,
constructs the PassageStore, fits BM25Okapi, and serializes the index
under data/indexes/. Reuses cached indexes when corpus and configuration
are unchanged.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from veritas.config import ROOT_DIR, load_config
from veritas.retrieval.bm25 import BM25Index
from veritas.retrieval.chunking import ChunkerConfig
from veritas.retrieval.corpus import (
    PassageStore,
    build_corpus_from_processed,
    create_synthetic_corpus,
)

logger = logging.getLogger("veritas.retrieval.build_index")


def setup_logging(level: str = "INFO") -> None:
    """Configure console logging for CLI execution."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_build_index(
    processed_dir: str | Path = "data/processed",
    datasets: list[str] | None = None,
    splits: list[str] | None = None,
    index_dir: str | Path = "data/indexes/bm25_default",
    chunk_size: int = 150,
    chunk_overlap: int = 20,
    k1: float = 1.5,
    b: float = 0.75,
    force_rebuild: bool = False,
    smoke_test: bool = False,
    test_query: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the BM25 corpus preparation and index build pipeline.

    Args:
        processed_dir: Path to directory containing processed dataset folders.
        datasets: List of dataset names to include. If None, inspects all in processed_dir.
        splits: List of dataset splits to include.
        index_dir: Output directory where index and passage store are saved.
        chunk_size: Target chunk size in words (target 100-200).
        chunk_overlap: Word overlap between successive chunks (default 20).
        k1: BM25 term frequency saturation parameter (default 1.5).
        b: BM25 length normalization parameter (default 0.75).
        force_rebuild: If True, forces re-indexing even if cache matches.
        smoke_test: If True, builds a small synthetic index to verify pipeline health.
        test_query: Optional natural language query to test immediately after building.
        config_path: Optional path to YAML configuration file to load defaults from.

    Returns:
        Summary dictionary of build results.
    """
    # Load configuration if provided
    if config_path:
        cfg = load_config(config_path)
        retrieval_cfg = cfg.get("retrieval", {})
        if datasets is None and "datasets" in retrieval_cfg:
            datasets = retrieval_cfg["datasets"]
        if "index_dir" in retrieval_cfg and index_dir == "data/indexes/bm25_default":
            index_dir = retrieval_cfg["index_dir"]
        chunk_size = retrieval_cfg.get("chunk_size", chunk_size)
        chunk_overlap = retrieval_cfg.get("chunk_overlap", chunk_overlap)
        k1 = float(retrieval_cfg.get("k1", k1))
        b = float(retrieval_cfg.get("b", b))

    resolved_index_dir = Path(index_dir)
    if not resolved_index_dir.is_absolute():
        resolved_index_dir = ROOT_DIR / resolved_index_dir

    chunker_config = ChunkerConfig(
        target_words=chunk_size,
        max_words=max(200, chunk_size),
        overlap_words=chunk_overlap,
    )

    logger.info("=" * 65)
    logger.info("VERITAS BM25 Retrieval Index Build")
    logger.info("=" * 65)
    logger.info("Target index dir:  %s", resolved_index_dir)
    logger.info("Chunk config:      target=%d, overlap=%d", chunk_size, chunk_overlap)
    logger.info("BM25 parameters:   k1=%.2f, b=%.2f", k1, b)
    logger.info("Force rebuild:     %s", force_rebuild)

    if smoke_test:
        logger.info("Running SMOKE-TEST with small synthetic corpus...")
        synthetic_records = [
            {
                "id": "claim_test_1",
                "evidence": [
                    {
                        "text": "The World Health Organization confirmed that vaccination reduces severe COVID-19 hospitalizations by over 90 percent according to clinical trials.",
                        "url": "https://www.who.int/news/vaccine-study",
                        "date": "2024-03-15",
                        "domain": "who.int",
                    }
                ],
            },
            {
                "id": "claim_test_2",
                "evidence": [
                    {
                        "text": "The Bureau of Labor Statistics reported that nonfarm payroll employment rose by 275,000 in February, while the unemployment rate edged up to 3.9 percent.",
                        "url": "https://www.bls.gov/news.release/empsit.nr0.htm",
                        "date": "2024-03-08",
                        "domain": "bls.gov",
                    }
                ],
            },
            {
                "id": "claim_test_3",
                "evidence": [
                    {
                        "text": "NASA's James Webb Space Telescope detected carbon dioxide and methane in the atmosphere of exoplanet K2-18 b in the habitable zone.",
                        "url": "https://www.nasa.gov/universe/webb-discovers-methane-carbon-dioxide-in-atmosphere-of-k2-18-b/",
                        "date": "2023-09-11",
                        "domain": "nasa.gov",
                    }
                ],
            },
        ]
        passage_store = create_synthetic_corpus(synthetic_records, chunker_config=chunker_config)
        summary = {
            "mode": "smoke_test",
            "total_passages_created": len(passage_store),
            "datasets": {"synthetic": len(synthetic_records)},
        }
    else:
        resolved_processed = Path(processed_dir)
        if not resolved_processed.is_absolute():
            resolved_processed = ROOT_DIR / resolved_processed

        logger.info("Scanning processed evidence in: %s", resolved_processed)
        passage_store, summary = build_corpus_from_processed(
            processed_dir=resolved_processed,
            datasets=datasets,
            splits=splits,
            chunker_config=chunker_config,
            deduplicate=True,
        )

    # Build or reuse cached index
    index = BM25Index.build_or_load(
        passage_store=passage_store,
        index_dir=resolved_index_dir,
        k1=k1,
        b=b,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_datasets=datasets or list(summary.get("datasets", {}).keys()),
        force_rebuild=force_rebuild,
    )

    result_summary = {
        "status": "success",
        "index_dir": str(resolved_index_dir),
        "total_passages": len(passage_store),
        "k1": k1,
        "b": b,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "corpus_summary": summary,
    }

    if test_query:
        logger.info("Executing test query: %r", test_query)
        search_results = index.search(test_query, top_k=3)
        result_summary["test_query_results"] = [r.to_dict() for r in search_results]
        logger.info("Retrieved %d matches for test query.", len(search_results))
        for r in search_results:
            snippet = r.passage.text[:80] + "..." if len(r.passage.text) > 80 else r.passage.text
            logger.info("  Rank %d [score=%.3f]: %s | %s", r.rank, r.score, r.passage_id, snippet)

    logger.info("=" * 65)
    logger.info("BM25 Index Build Completed Successfully.")
    logger.info("Passages in index: %d", len(passage_store))
    logger.info("=" * 65)

    return result_summary


def main() -> None:
    """CLI entry point for veritas-build-bm25-index."""
    parser = argparse.ArgumentParser(
        description="Build or update VERITAS BM25 retrieval index from evidence corpus."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g. configs/retrieval_bm25.yaml).",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Path to directory containing processed JSONL datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to ingest evidence from (e.g. rawfc averitec custom_recent liar).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to ingest (e.g. train validation test).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/indexes/bm25_default",
        help="Output directory to store index, passages, and metadata.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=150,
        help="Target chunk size in words (default: 150).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=20,
        help="Overlap between adjacent chunks in words (default: 20).",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter (default: 1.5).",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (default: 0.75).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild even if cached index matches corpus and config.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build a fast synthetic index for pipeline verification.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional test query to run against the index after building.",
    )

    args = parser.parse_args()
    setup_logging()

    try:
        run_build_index(
            processed_dir=args.processed_dir,
            datasets=args.datasets,
            splits=args.splits,
            index_dir=args.index_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            k1=args.k1,
            b=args.b,
            force_rebuild=args.force_rebuild,
            smoke_test=args.smoke_test,
            test_query=args.query,
            config_path=args.config,
        )
    except Exception as exc:
        logger.exception("Index build failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
