"""Hybrid evidence retrieval combining BM25 and dense retrieval with Reciprocal Rank Fusion.

Supports:
- Modes: 'hybrid', 'bm25', 'dense'.
- BM25 top 50 and dense top 50 candidate pools by default.
- RRF constant k=60 by default (sum of 1 / (k + rank) over candidate lists).
- Deduplicated candidate passage IDs.
- Single query or multiple query reformulations per claim.
- Rich return objects with passage IDs, ranks, source methods, and fusion scores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from veritas.config import ROOT_DIR, load_config
from veritas.retrieval.bm25 import BM25Index
from veritas.retrieval.dense import DenseIndex
from veritas.retrieval.fusion import FusedSearchResult, reciprocal_rank_fusion
from veritas.retrieval.schema import Passage, SearchResult

logger = logging.getLogger("veritas.retrieval.hybrid")


class HybridRetriever:
    """Configurable hybrid retriever combining BM25 and dense vector search via RRF."""

    def __init__(
        self,
        bm25_index: BM25Index | None = None,
        dense_index: DenseIndex | None = None,
        mode: str = "hybrid",
        bm25_top_k: int = 50,
        dense_top_k: int = 50,
        rrf_k: int = 60,
        top_k: int = 50,
        weights: dict[str, float] | None = None,
    ) -> None:
        valid_modes = {"hybrid", "bm25", "dense"}
        mode_lower = mode.lower().strip()
        if mode_lower not in valid_modes:
            raise ValueError(f"Invalid retrieval mode '{mode}'. Must be one of {sorted(valid_modes)}.")

        self.mode = mode_lower
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.bm25_top_k = int(bm25_top_k)
        self.dense_top_k = int(dense_top_k)
        self.rrf_k = int(rrf_k)
        self.top_k = int(top_k)
        self.weights = dict(weights) if weights else {"bm25": 1.0, "dense": 1.0}

    @property
    def passage_store(self):
        """Access underlying passage store from whichever index is available."""
        if self.bm25_index and self.bm25_index.passage_store:
            return self.bm25_index.passage_store
        if self.dense_index and self.dense_index.passage_store:
            return self.dense_index.passage_store
        return None

    def retrieve(
        self,
        queries: str | list[str],
        mode: str | None = None,
        top_k: int | None = None,
    ) -> list[FusedSearchResult]:
        """Retrieve and rank candidate passages for a claim.

        Args:
            queries: Single query string or list of multiple query variations/decompositions.
            mode: Optional mode override ('hybrid', 'bm25', 'dense'). Defaults to self.mode.
            top_k: Maximum number of fused candidates to return. Defaults to self.top_k.

        Returns:
            Deduplicated list of FusedSearchResult objects ordered by rank.
        """
        active_mode = (mode or self.mode).lower().strip()
        limit = top_k if top_k is not None else self.top_k

        if isinstance(queries, str):
            query_list = [queries.strip()] if queries.strip() else []
        elif isinstance(queries, (list, tuple)):
            query_list = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        else:
            query_list = []

        if not query_list:
            return []

        ranked_lists: list[list[SearchResult]] = []
        source_names: list[str] = []
        source_weights: dict[str, float] = {}

        # 1. Run BM25 candidates
        if active_mode in ("hybrid", "bm25"):
            if self.bm25_index is None:
                raise RuntimeError("BM25 retrieval requested but bm25_index is not initialized.")

            for q_idx, q in enumerate(query_list):
                bm25_res = self.bm25_index.search(q, top_k=self.bm25_top_k)
                src_tag = f"bm25_q{q_idx}" if len(query_list) > 1 else "bm25"
                ranked_lists.append(bm25_res)
                source_names.append(src_tag)
                source_weights[src_tag] = self.weights.get("bm25", 1.0)

        # 2. Run Dense candidates
        if active_mode in ("hybrid", "dense"):
            if self.dense_index is None:
                raise RuntimeError("Dense retrieval requested but dense_index is not initialized.")

            for q_idx, q in enumerate(query_list):
                dense_res = self.dense_index.search(q, top_k=self.dense_top_k)
                src_tag = f"dense_q{q_idx}" if len(query_list) > 1 else "dense"
                ranked_lists.append(dense_res)
                source_names.append(src_tag)
                source_weights[src_tag] = self.weights.get("dense", 1.0)

        # If only a single list was retrieved (e.g. single query in bm25 or dense mode)
        if len(ranked_lists) == 1:
            single_list = ranked_lists[0]
            src = source_names[0]
            # Normalize method name (strip _q0)
            base_src = src.split("_")[0]
            results: list[FusedSearchResult] = []
            for rank_idx, item in enumerate(single_list[:limit], start=1):
                results.append(
                    FusedSearchResult(
                        passage_id=item.passage_id,
                        rank=rank_idx,
                        score=item.score,
                        sources=[base_src],
                        source_ranks={base_src: item.rank},
                        source_scores={base_src: item.score},
                        passage=item.passage,
                    )
                )
            return results

        # 3. Reciprocal Rank Fusion across all candidate lists
        fused = reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            source_names=source_names,
            k=self.rrf_k,
            top_k=limit,
            weights=source_weights,
        )

        # Normalize source names in the output (e.g. map 'bm25_q0' -> 'bm25', 'dense_q1' -> 'dense')
        store = self.passage_store
        for item in fused:
            canonical_sources: list[str] = []
            for s in item.sources:
                base = s.split("_")[0]
                if base not in canonical_sources:
                    canonical_sources.append(base)
            item.sources = canonical_sources

            # Ensure passage object is populated
            if item.passage is None and store is not None:
                item.passage = store.get_by_id(item.passage_id)

        return fused

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | str | Path = "configs/default.yaml",
        bm25_dir: str | Path | None = None,
        dense_dir: str | Path | None = None,
        mode: str | None = None,
    ) -> HybridRetriever:
        """Initialize HybridRetriever by loading BM25 and Dense indexes specified in configuration."""
        if isinstance(config, (str, Path)):
            cfg = load_config(config)
        else:
            cfg = config

        retrieval_cfg = cfg.get("retrieval", {})

        active_mode = mode or retrieval_cfg.get("mode", "hybrid")
        bm25_top_k = retrieval_cfg.get("bm25_top_k", retrieval_cfg.get("bm25", {}).get("bm25_top_k", 50))
        dense_top_k = retrieval_cfg.get("dense_top_k", retrieval_cfg.get("dense", {}).get("dense_top_k", 50))
        rrf_k = retrieval_cfg.get("rrf_constant", retrieval_cfg.get("hybrid", {}).get("rrf_constant", 60))

        # Resolve index directories
        if bm25_dir is None:
            bm25_dir = retrieval_cfg.get("bm25", {}).get("index_dir", "data/indexes/bm25_default")
        if dense_dir is None:
            dense_dir = retrieval_cfg.get("dense", {}).get("index_dir", "data/indexes/dense_default")

        bm25_path = Path(bm25_dir)
        if not bm25_path.is_absolute():
            bm25_path = ROOT_DIR / bm25_path

        dense_path = Path(dense_dir)
        if not dense_path.is_absolute():
            dense_path = ROOT_DIR / dense_path

        bm25_index: BM25Index | None = None
        if active_mode in ("hybrid", "bm25") and bm25_path.is_dir():
            try:
                bm25_index = BM25Index.load(bm25_path)
            except Exception as exc:
                logger.warning("Could not load BM25 index from %s: %s", bm25_path, exc)

        dense_index: DenseIndex | None = None
        if active_mode in ("hybrid", "dense") and dense_path.is_dir():
            try:
                dense_index = DenseIndex.load(dense_path)
            except Exception as exc:
                logger.warning("Could not load Dense index from %s: %s", dense_path, exc)

        return cls(
            bm25_index=bm25_index,
            dense_index=dense_index,
            mode=active_mode,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
            rrf_k=rrf_k,
        )
