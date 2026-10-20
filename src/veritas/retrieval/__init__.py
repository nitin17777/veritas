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
from veritas.retrieval.schema import IndexMetadata, Passage, SearchResult

__all__ = [
    "BM25Index",
    "ChunkerConfig",
    "IndexMetadata",
    "Passage",
    "PassageStore",
    "SearchResult",
    "build_corpus_from_processed",
    "build_corpus_from_records",
    "chunk_text",
    "create_synthetic_corpus",
    "tokenize_bm25",
]
