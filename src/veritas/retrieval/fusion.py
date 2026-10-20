"""Reciprocal Rank Fusion (RRF) and candidate rank aggregation for VERITAS.

Combines multiple ranked candidate lists (e.g. BM25 and dense retrieval,
or multiple query reformulations per claim) into a unified, deduplicated ranking.
Formula:
    RRF_score(d) = sum_{m in M} ( 1 / (k + rank_m(d)) )
where k is the smoothing constant (default 60) and rank_m(d) is the 1-based rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from veritas.retrieval.schema import Passage, SearchResult


@dataclass
class FusedSearchResult:
    """A deduplicated search result produced by Reciprocal Rank Fusion.

    Attributes:
        passage_id: Unique identifier for the passage.
        rank: Final 1-based fused rank.
        score: Computed RRF fusion score.
        sources: List of retrieval source names that retrieved this passage (e.g. ['bm25', 'dense']).
        source_ranks: Mapping from source name to the 1-based rank in that source's list.
        source_scores: Mapping from source name to original similarity score.
        passage: The Passage record itself, if available.
        metadata: Arbitrary metadata.
    """

    passage_id: str
    rank: int
    score: float
    sources: list[str] = field(default_factory=list)
    source_ranks: dict[str, int] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)
    passage: Passage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert fused search result to dictionary."""
        return {
            "passage_id": self.passage_id,
            "rank": int(self.rank),
            "score": float(self.score),
            "rrf_score": float(self.score),
            "sources": list(self.sources),
            "source_ranks": dict(self.source_ranks),
            "source_scores": dict(self.source_scores),
            "passage": self.passage.to_dict() if self.passage else None,
            "metadata": dict(self.metadata),
        }


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult] | list[dict[str, Any]] | list[str]],
    source_names: list[str] | None = None,
    k: int = 60,
    top_k: int | None = None,
    weights: dict[str, float] | None = None,
) -> list[FusedSearchResult]:
    """Fuse multiple ranked candidate lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked candidate lists. Items can be SearchResult objects,
                      dictionaries with 'passage_id' (and optional 'score', 'rank'),
                      or string passage IDs.
        source_names: Optional labels identifying the method or query for each list.
                      Defaults to ['source_0', 'source_1', ...].
        k: RRF smoothing constant (default: 60).
        top_k: Maximum number of fused candidates to return. If None, returns all.
        weights: Optional source weighting multiplier dict {source_name: weight}.
                 Defaults to 1.0 for all sources.

    Returns:
        Deduplicated list of FusedSearchResult objects ordered by RRF score descending.
    """
    if k <= 0:
        raise ValueError(f"RRF constant k must be positive, got {k}")

    if not ranked_lists:
        return []

    if source_names is None:
        source_names = [f"source_{i}" for i in range(len(ranked_lists))]
    elif len(source_names) != len(ranked_lists):
        raise ValueError(
            f"Length of source_names ({len(source_names)}) must match ranked_lists ({len(ranked_lists)})"
        )

    # Accumulator dict: passage_id -> intermediate data
    accumulators: dict[str, dict[str, Any]] = {}

    for list_idx, (r_list, source) in enumerate(zip(ranked_lists, source_names)):
        weight = float(weights.get(source, 1.0)) if weights and source in weights else 1.0

        for pos, item in enumerate(r_list, start=1):
            if isinstance(item, SearchResult):
                p_id = item.passage_id
                item_score = item.score
                passage_obj = item.passage
                item_rank = item.rank if item.rank is not None else pos
            elif isinstance(item, dict):
                p_id = item.get("passage_id") or item.get("id")
                if not p_id:
                    continue
                item_score = float(item.get("score", 0.0))
                passage_obj = item.get("passage")
                if isinstance(passage_obj, dict):
                    passage_obj = Passage.from_dict(passage_obj)
                item_rank = int(item.get("rank", pos))
            elif isinstance(item, str):
                p_id = item
                item_score = 0.0
                passage_obj = None
                item_rank = pos
            else:
                continue

            p_id = str(p_id).strip()
            if not p_id:
                continue

            # RRF contribution: weight / (k + rank)
            contribution = weight / (k + item_rank)

            if p_id not in accumulators:
                accumulators[p_id] = {
                    "passage_id": p_id,
                    "rrf_score": 0.0,
                    "sources": [],
                    "source_ranks": {},
                    "source_scores": {},
                    "passage": passage_obj,
                }

            acc = accumulators[p_id]
            acc["rrf_score"] += contribution
            if source not in acc["sources"]:
                acc["sources"].append(source)
            acc["source_ranks"][source] = item_rank
            acc["source_scores"][source] = item_score
            if acc["passage"] is None and passage_obj is not None:
                acc["passage"] = passage_obj

    if not accumulators:
        return []

    # Sort descending by rrf_score, tie-break by passage_id ascending for 100% determinism
    sorted_items = sorted(
        accumulators.values(),
        key=lambda x: (-x["rrf_score"], x["passage_id"]),
    )

    if top_k is not None and top_k > 0:
        sorted_items = sorted_items[:top_k]

    fused_results: list[FusedSearchResult] = []
    for rank_idx, item in enumerate(sorted_items, start=1):
        fused_results.append(
            FusedSearchResult(
                passage_id=item["passage_id"],
                rank=rank_idx,
                score=float(item["rrf_score"]),
                sources=item["sources"],
                source_ranks=item["source_ranks"],
                source_scores=item["source_scores"],
                passage=item["passage"],
            )
        )

    return fused_results
