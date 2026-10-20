"""Tests for Reciprocal Rank Fusion (RRF) in VERITAS retrieval."""

import pytest

from veritas.retrieval.fusion import FusedSearchResult, reciprocal_rank_fusion
from veritas.retrieval.schema import Passage, SearchResult


def test_rrf_deterministic_toy_list_expected_ordering():
    """CRITICAL TEST: Verify exact mathematical RRF scores and expected ranking on toy candidate lists."""
    # List 1: doc_A (rank 1), doc_B (rank 2), doc_C (rank 3)
    # List 2: doc_B (rank 1), doc_D (rank 2), doc_A (rank 3)
    list_1 = ["doc_A", "doc_B", "doc_C"]
    list_2 = ["doc_B", "doc_D", "doc_A"]

    k = 60
    # Expected calculations:
    # doc_B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.01612903 + 0.01639344 = 0.03252247
    # doc_A: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.01639344 + 0.01587302 = 0.03226646
    # doc_D: 1/(60+2)           = 1/62           = 0.01612903
    # doc_C: 1/(60+3)           = 1/63           = 0.01587302

    expected_score_b = (1.0 / 62.0) + (1.0 / 61.0)
    expected_score_a = (1.0 / 61.0) + (1.0 / 63.0)
    expected_score_d = 1.0 / 62.0
    expected_score_c = 1.0 / 63.0

    fused = reciprocal_rank_fusion(
        ranked_lists=[list_1, list_2],
        source_names=["bm25", "dense"],
        k=k,
    )

    # 1. Exact ordering verification
    assert len(fused) == 4
    assert [x.passage_id for x in fused] == ["doc_B", "doc_A", "doc_D", "doc_C"]
    assert [x.rank for x in fused] == [1, 2, 3, 4]

    # 2. Exact score verification
    assert abs(fused[0].score - expected_score_b) < 1e-6
    assert abs(fused[1].score - expected_score_a) < 1e-6
    assert abs(fused[2].score - expected_score_d) < 1e-6
    assert abs(fused[3].score - expected_score_c) < 1e-6

    # 3. Source tracking and deduplication verification
    assert fused[0].passage_id == "doc_B"
    assert set(fused[0].sources) == {"bm25", "dense"}
    assert fused[0].source_ranks == {"bm25": 2, "dense": 1}

    assert fused[1].passage_id == "doc_A"
    assert set(fused[1].sources) == {"bm25", "dense"}
    assert fused[1].source_ranks == {"bm25": 1, "dense": 3}

    assert fused[2].passage_id == "doc_D"
    assert fused[2].sources == ["dense"]
    assert fused[2].source_ranks == {"dense": 2}

    assert fused[3].passage_id == "doc_C"
    assert fused[3].sources == ["bm25"]
    assert fused[3].source_ranks == {"bm25": 3}


def test_rrf_with_search_result_objects():
    p1 = Passage(passage_id="p1", text="Text 1")
    p2 = Passage(passage_id="p2", text="Text 2")

    res_bm25 = [
        SearchResult(passage_id="p1", score=12.5, rank=1, passage=p1),
        SearchResult(passage_id="p2", score=8.0, rank=2, passage=p2),
    ]
    res_dense = [
        SearchResult(passage_id="p2", score=0.88, rank=1, passage=p2),
        SearchResult(passage_id="p1", score=0.72, rank=2, passage=p1),
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists=[res_bm25, res_dense],
        source_names=["bm25", "dense"],
        k=60,
    )

    assert len(fused) == 2
    # Both documents have identical rank sum (1 and 2): 1/61 + 1/62
    expected_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert abs(fused[0].score - expected_score) < 1e-6
    assert abs(fused[1].score - expected_score) < 1e-6

    # Both should have passages populated
    assert fused[0].passage is not None
    assert fused[1].passage is not None


def test_rrf_top_k_clipping():
    list_1 = [f"doc_{i}" for i in range(100)]
    fused = reciprocal_rank_fusion([list_1], k=60, top_k=20)
    assert len(fused) == 20
    assert fused[0].rank == 1
    assert fused[-1].rank == 20


def test_rrf_weighted():
    # doc_1 in bm25 only, doc_2 in dense only, both rank 1
    list_bm25 = ["doc_bm25"]
    list_dense = ["doc_dense"]

    # Dense weighted 2.0x vs BM25 1.0x
    fused = reciprocal_rank_fusion(
        ranked_lists=[list_bm25, list_dense],
        source_names=["bm25", "dense"],
        k=60,
        weights={"bm25": 1.0, "dense": 2.0},
    )

    assert fused[0].passage_id == "doc_dense"
    assert fused[1].passage_id == "doc_bm25"
    assert abs(fused[0].score - (2.0 / 61.0)) < 1e-6
    assert abs(fused[1].score - (1.0 / 61.0)) < 1e-6


def test_rrf_invalid_k_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["doc_1"]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["doc_1"]], k=-10)


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
