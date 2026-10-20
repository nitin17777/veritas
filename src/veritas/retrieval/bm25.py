"""BM25 retrieval index implementation for VERITAS.

Implements rank_bm25-backed full-text search with consistent case-insensitive
tokenization, strict preservation of passage ordering, index metadata tracking,
and caching logic to build once and reuse when corpus and config are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import joblib
except ImportError:
    joblib = None

import numpy as np
from rank_bm25 import BM25Okapi

from veritas.retrieval.corpus import PassageStore
from veritas.retrieval.schema import IndexMetadata, Passage, SearchResult

logger = logging.getLogger("veritas.retrieval.bm25")

# Token pattern: match contiguous alphanumeric word characters
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text consistently and case-insensitively for BM25.

    Normalizes Unicode casing to lowercase and extracts alphanumeric word tokens.
    Guarantees that index time and query time tokenization are 100% identical.

    Args:
        text: Raw text string to tokenize.

    Returns:
        List of lowercased string tokens.
    """
    if not text or not isinstance(text, str):
        return []
    return TOKEN_PATTERN.findall(text.lower())


def compute_config_fingerprint(
    k1: float,
    b: float,
    chunk_size: int = 150,
    chunk_overlap: int = 20,
    backend: str = "BM25Okapi",
) -> str:
    """Compute a deterministic hash of the retrieval configuration."""
    cfg_str = f"backend={backend}|k1={k1:.4f}|b={b:.4f}|chunk_size={chunk_size}|chunk_overlap={chunk_overlap}"
    return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()


class BM25Index:
    """BM25 retrieval index wrapping rank_bm25.BM25Okapi.

    Maintains 1:1 positional correspondence between corpus document vectors
    and the underlying PassageStore so that search results always resolve
    to the correct Passage records.
    """

    def __init__(
        self,
        passage_store: PassageStore | None = None,
        bm25: BM25Okapi | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: IndexMetadata | None = None,
        index_dir: Path | None = None,
    ) -> None:
        self.passage_store: PassageStore = passage_store if passage_store is not None else PassageStore([])
        self.bm25: BM25Okapi | None = bm25
        self.k1: float = float(k1)
        self.b: float = float(b)
        self.metadata: IndexMetadata | None = metadata
        self.index_dir: Path | None = Path(index_dir) if index_dir else None

    @classmethod
    def build(
        cls,
        passage_store: PassageStore,
        k1: float = 1.5,
        b: float = 0.75,
        index_name: str = "bm25_default",
        chunk_size: int = 150,
        chunk_overlap: int = 20,
        source_datasets: list[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> BM25Index:
        """Construct and fit a BM25Index from an existing PassageStore.

        Args:
            passage_store: Ordered store of Passage instances.
            k1: BM25 term frequency saturation parameter (default 1.5).
            b: BM25 document length normalization parameter (default 0.75).
            index_name: Descriptive identifier for this index.
            chunk_size: Configured chunk size used during corpus creation.
            chunk_overlap: Configured chunk overlap used during corpus creation.
            source_datasets: List of dataset names that contributed evidence.
            extra_metadata: Optional additional metadata fields.

        Returns:
            Fitted BM25Index instance.
        """
        n_passages = len(passage_store)
        logger.info("Building BM25 index '%s' across %d passages...", index_name, n_passages)

        if n_passages == 0:
            logger.warning("PassageStore is empty. Initializing empty BM25 index.")
            bm25_instance = None
        else:
            # Tokenize all passages in strict order matching passage_store.passages
            tokenized_corpus = [tokenize_bm25(p.text) for p in passage_store.passages]
            bm25_instance = BM25Okapi(tokenized_corpus, k1=k1, b=b)

        corpus_fp = passage_store.compute_fingerprint()
        config_fp = compute_config_fingerprint(
            k1=k1,
            b=b,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        metadata = IndexMetadata(
            index_name=index_name,
            backend="BM25Okapi",
            total_passages=n_passages,
            corpus_fingerprint=corpus_fp,
            config_fingerprint=config_fp,
            created_at=now_iso,
            k1=k1,
            b=b,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_datasets=list(source_datasets or []),
            files={
                "passages": "passages.jsonl",
                "index": "bm25_index.joblib",
                "metadata": "index_metadata.json",
            },
            extra=dict(extra_metadata or {}),
        )

        return cls(
            passage_store=passage_store,
            bm25=bm25_instance,
            k1=k1,
            b=b,
            metadata=metadata,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float | None = 0.0,
    ) -> list[SearchResult]:
        """Query the index and return top-k ranked passages.

        Args:
            query: Natural language query statement or claim sentence.
            top_k: Maximum number of ranked results to return.
            min_score: Minimum BM25 score required for inclusion.
                       If 0.0, drops passages with no matching terms.
                       If None, returns up to top_k without score filtering.

        Returns:
            List of SearchResult objects sorted by score descending.
        """
        if self.bm25 is None or len(self.passage_store) == 0:
            return []

        tokens = tokenize_bm25(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        if len(scores) == 0:
            return []

        # Argsort descending
        ranked_indices = np.argsort(scores)[::-1]

        results: list[SearchResult] = []
        rank = 1

        for idx in ranked_indices:
            score = float(scores[idx])

            if min_score is not None and score < min_score:
                # Since array is sorted descending, any subsequent scores will also be < min_score
                break

            passage = self.passage_store.passages[idx]
            results.append(
                SearchResult(
                    passage_id=passage.passage_id,
                    score=score,
                    rank=rank,
                    passage=passage,
                )
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def save(self, index_dir: Path | str) -> Path:
        """Persist index, passage store, and metadata to directory.

        Saves:
        - index_dir/passages.jsonl
        - index_dir/bm25_index.joblib
        - index_dir/index_metadata.json

        Args:
            index_dir: Target directory path.

        Returns:
            Resolved Path to index_dir.
        """
        target = Path(index_dir)
        target.mkdir(parents=True, exist_ok=True)

        # 1. Save passage store
        passages_file = target / "passages.jsonl"
        self.passage_store.save_jsonl(passages_file)

        # 2. Save fitted BM25 model
        bm25_file = target / "bm25_index.joblib"
        if joblib is not None:
            joblib.dump(self.bm25, bm25_file)
        else:
            with bm25_file.open("wb") as f:
                pickle.dump(self.bm25, f)

        # 3. Save metadata
        if self.metadata is None:
            now_iso = datetime.now(timezone.utc).isoformat()
            self.metadata = IndexMetadata(
                index_name=target.name,
                backend="BM25Okapi",
                total_passages=len(self.passage_store),
                corpus_fingerprint=self.passage_store.compute_fingerprint(),
                config_fingerprint=compute_config_fingerprint(self.k1, self.b),
                created_at=now_iso,
                k1=self.k1,
                b=self.b,
            )

        metadata_file = target / "index_metadata.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        self.index_dir = target
        logger.info("Saved BM25 index and %d passages to %s", len(self.passage_store), target)
        return target

    @classmethod
    def load(cls, index_dir: Path | str) -> BM25Index:
        """Load an existing index, passage store, and metadata from directory.

        Args:
            index_dir: Directory containing passages.jsonl, bm25_index.joblib (or .pkl), and index_metadata.json.

        Returns:
            Loaded BM25Index instance.
        """
        target = Path(index_dir)
        if not target.is_dir():
            raise FileNotFoundError(f"Index directory does not exist: {target}")

        passages_file = target / "passages.jsonl"
        bm25_file = target / "bm25_index.joblib"
        if not bm25_file.is_file():
            bm25_file = target / "bm25_index.pkl"
        metadata_file = target / "index_metadata.json"

        if not passages_file.is_file():
            raise FileNotFoundError(f"Passage store file missing: {passages_file}")
        if not bm25_file.is_file():
            raise FileNotFoundError(f"BM25 model file missing: {bm25_file}")
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Index metadata file missing: {metadata_file}")

        passage_store = PassageStore.load_jsonl(passages_file)
        try:
            if joblib is not None:
                bm25_instance = joblib.load(bm25_file)
            else:
                with bm25_file.open("rb") as f:
                    bm25_instance = pickle.load(f)
        except Exception:
            with bm25_file.open("rb") as f:
                bm25_instance = pickle.load(f)

        with metadata_file.open("r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        metadata = IndexMetadata.from_dict(meta_dict)

        logger.info(
            "Loaded BM25 index '%s' with %d passages from %s",
            metadata.index_name,
            len(passage_store),
            target,
        )

        return cls(
            passage_store=passage_store,
            bm25=bm25_instance,
            k1=metadata.k1,
            b=metadata.b,
            metadata=metadata,
            index_dir=target,
        )

    @classmethod
    def build_or_load(
        cls,
        passage_store: PassageStore,
        index_dir: Path | str,
        k1: float = 1.5,
        b: float = 0.75,
        chunk_size: int = 150,
        chunk_overlap: int = 20,
        source_datasets: list[str] | None = None,
        force_rebuild: bool = False,
    ) -> BM25Index:
        """Build a new index or reuse the cached index if corpus and config are unchanged.

        Verifies corpus and configuration fingerprints against index_metadata.json.
        If both match and all files are intact, loads the cached index directly.

        Args:
            passage_store: Current PassageStore instance.
            index_dir: Target directory path for storing/loading index.
            k1: BM25 k1 parameter.
            b: BM25 b parameter.
            chunk_size: Target chunk word count.
            chunk_overlap: Chunk overlap word count.
            source_datasets: List of dataset names.
            force_rebuild: If True, forces rebuilding even if cache matches.

        Returns:
            Fitted or loaded BM25Index.
        """
        target = Path(index_dir)
        meta_file = target / "index_metadata.json"
        psg_file = target / "passages.jsonl"
        idx_file = target / "bm25_index.joblib"
        if not idx_file.is_file():
            idx_file = target / "bm25_index.pkl"

        current_corpus_fp = passage_store.compute_fingerprint()
        current_config_fp = compute_config_fingerprint(
            k1=k1,
            b=b,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not force_rebuild and meta_file.is_file() and psg_file.is_file() and idx_file.is_file():
            try:
                with meta_file.open("r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
                cached_corpus_fp = meta_dict.get("corpus_fingerprint")
                cached_config_fp = meta_dict.get("config_fingerprint")

                if cached_corpus_fp == current_corpus_fp and cached_config_fp == current_config_fp:
                    logger.info(
                        "Corpus and configuration unchanged. Reusing cached BM25 index from %s",
                        target,
                    )
                    return cls.load(target)
                else:
                    logger.info(
                        "Fingerprint mismatch (corpus_match=%s, config_match=%s). Rebuilding index...",
                        cached_corpus_fp == current_corpus_fp,
                        cached_config_fp == current_config_fp,
                    )
            except Exception as exc:
                logger.warning("Failed to validate cached index metadata (%s). Rebuilding...", exc)

        # Build fresh index
        index = cls.build(
            passage_store=passage_store,
            k1=k1,
            b=b,
            index_name=target.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_datasets=source_datasets,
        )
        index.save(target)
        return index
