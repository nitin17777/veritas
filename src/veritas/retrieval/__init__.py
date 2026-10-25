"""VERITAS retrieval module.

Provides passage schema contracts, document chunking, corpus preparation from
fact-checking evidence, and BM25 index construction and search.
"""

from veritas.retrieval.bm25 import BM25Index, tokenize_bm25
from veritas.retrieval.chunking import ChunkerConfig, chunk_text
from veritas.retrieval.corpus import (
    PassageStore,
    build_corpus_from_processed,
    build_corpus_from_records,
    create_synthetic_corpus,
)
from veritas.retrieval.dense import (
    DenseEncoder,
    DenseIndex,
    DenseIndexMetadata,
    MockEmbedder,
    resolve_model_prefixes,
)
from veritas.retrieval.fusion import FusedSearchResult, reciprocal_rank_fusion
from veritas.retrieval.hybrid import HybridRetriever
from veritas.retrieval.metrics import (
    compute_ndcg_at_k,
    compute_recall_at_k,
    evaluate_retrieval_predictions,
)
from veritas.retrieval.reranker import (
    CrossEncoderProtocol,
    CrossEncoderReranker,
    MockCrossEncoder,
    RerankResult,
    RerankerConfig,
    normalize_scores,
    rerank_candidates,
)
from veritas.retrieval.schema import IndexMetadata, Passage, SearchResult

__all__ = [
    "BM25Index",
    "ChunkerConfig",
    "CrossEncoderProtocol",
    "CrossEncoderReranker",
    "DenseEncoder",
    "DenseIndex",
    "DenseIndexMetadata",
    "FusedSearchResult",
    "HybridRetriever",
    "IndexMetadata",
    "MockCrossEncoder",
    "MockEmbedder",
    "Passage",
    "PassageStore",
    "RerankResult",
    "RerankerConfig",
    "SearchResult",
    "build_corpus_from_processed",
    "build_corpus_from_records",
    "chunk_text",
    "compute_ndcg_at_k",
    "compute_recall_at_k",
    "create_synthetic_corpus",
    "evaluate_retrieval_predictions",
    "normalize_scores",
    "reciprocal_rank_fusion",
    "rerank_candidates",
    "resolve_model_prefixes",
    "tokenize_bm25",
]
