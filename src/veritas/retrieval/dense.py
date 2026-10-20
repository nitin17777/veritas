"""Dense retrieval index implementation for VERITAS using sentence-transformers and FAISS.

Provides:
- Configurable embedding models (BGE, E5, MiniLM, etc.) with automatic or explicit query prefixes.
- L2-normalized embeddings with FAISS IndexFlatIP to compute exact cosine similarity.
- Strict 1:1 positional correspondence between FAISS rows, passage_ids.json, and the PassageStore.
- Batched embedding operations with automatic CUDA / CPU fallback.
- Metadata persistence (model name, dimension, normalization, fingerprints).
- Reusable index caching and a lightweight deterministic mock embedder for unit tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from veritas.retrieval.corpus import PassageStore
from veritas.retrieval.schema import Passage, SearchResult

logger = logging.getLogger("veritas.retrieval.dense")


class EmbedderProtocol(Protocol):
    """Protocol matching sentence-transformers SentenceTransformer.encode."""

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> np.ndarray: ...


class MockEmbedder:
    """Deterministic, lightweight embedder for tests that requires no downloads or GPU."""

    def __init__(self, dimension: int = 16, model_name: str = "mock-embedder") -> None:
        self.dimension = int(dimension)
        self.model_name = model_name

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]

        vectors: list[np.ndarray] = []
        for text in sentences:
            # Deterministic projection from text hash
            h = hashlib.sha256(text.encode("utf-8")).digest()
            # Generate deterministic pseudo-random floats in [-1, 1]
            rng = np.random.default_rng(seed=int.from_bytes(h[:8], "little"))
            vec = rng.standard_normal(self.dimension).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)

        return np.stack(vectors, axis=0)


def resolve_model_prefixes(
    model_name_or_path: str,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
) -> tuple[str, str]:
    """Resolve model-specific query and document prefixes based on model architecture conventions.

    - BGE models (BAAI/bge-*): require query prefix for retrieval; document prefix is empty.
    - E5 models (intfloat/e5-*): require 'query: ' and 'passage: '.
    - General models (MiniLM, Contriever): no prefix required.
    """
    lower = model_name_or_path.lower()

    if query_prefix is None:
        if "bge" in lower:
            query_prefix = "Represent this sentence for searching relevant passages: "
        elif "e5" in lower:
            query_prefix = "query: "
        else:
            query_prefix = ""

    if document_prefix is None:
        if "e5" in lower:
            document_prefix = "passage: "
        else:
            document_prefix = ""

    return query_prefix, document_prefix


def compute_dense_config_fingerprint(
    model_name_or_path: str,
    dimension: int,
    normalize_embeddings: bool,
    query_prefix: str,
    document_prefix: str,
    index_type: str = "IndexFlatIP",
) -> str:
    """Compute a deterministic hash of the dense retrieval configuration."""
    cfg_str = (
        f"model={model_name_or_path}|dim={dimension}|norm={normalize_embeddings}|"
        f"q_pre={query_prefix}|d_pre={document_prefix}|type={index_type}"
    )
    return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()


@dataclass
class DenseIndexMetadata:
    """Metadata tracking model architecture, dimensions, normalization, and index state."""

    index_name: str
    model_name_or_path: str
    dimension: int
    normalize_embeddings: bool = True
    index_type: str = "IndexFlatIP"
    corpus_fingerprint: str = ""
    config_fingerprint: str = ""
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 32
    total_passages: int = 0
    created_at: str = ""
    files: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DenseIndexMetadata:
        return cls(
            index_name=str(data.get("index_name", "")),
            model_name_or_path=str(data.get("model_name_or_path", "")),
            dimension=int(data.get("dimension", 0)),
            normalize_embeddings=bool(data.get("normalize_embeddings", True)),
            index_type=str(data.get("index_type", "IndexFlatIP")),
            corpus_fingerprint=str(data.get("corpus_fingerprint", "")),
            config_fingerprint=str(data.get("config_fingerprint", "")),
            query_prefix=str(data.get("query_prefix", "")),
            document_prefix=str(data.get("document_prefix", "")),
            batch_size=int(data.get("batch_size", 32)),
            total_passages=int(data.get("total_passages", 0)),
            created_at=str(data.get("created_at", "")),
            files=dict(data.get("files", {})),
            extra=dict(data.get("extra", {})),
        )


class DenseEncoder:
    """Wraps a SentenceTransformer or embedder with query/document prefix handling and batching."""

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto",
        normalize_embeddings: bool = True,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.normalize_embeddings = normalize_embeddings
        self.query_prefix, self.document_prefix = resolve_model_prefixes(
            model_name_or_path=model_name_or_path,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        )

        # Device resolution
        if device == "auto":
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        if embedder is not None:
            self._embedder = embedder
            # Probe dimension
            dummy_vec = self._embedder.encode(["probe"], normalize_embeddings=self.normalize_embeddings)
            self.dimension = int(dummy_vec.shape[1])
        else:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformers model '%s' on %s...", model_name_or_path, self.device)
            self._embedder = SentenceTransformer(model_name_or_path, device=self.device)
            self.dimension = int(self._embedder.get_sentence_embedding_dimension())

    def encode_queries(
        self,
        queries: list[str] | str,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode query texts applying the required query_prefix and L2 normalization."""
        if isinstance(queries, str):
            queries = [queries]

        prefixed = [f"{self.query_prefix}{q}" if self.query_prefix else q for q in queries]
        embeddings = self._embedder.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_documents(
        self,
        documents: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode passage texts applying document_prefix and L2 normalization in batches."""
        if not documents:
            return np.empty((0, self.dimension), dtype=np.float32)

        prefixed = [f"{self.document_prefix}{d}" if self.document_prefix else d for d in documents]
        embeddings = self._embedder.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(embeddings, dtype=np.float32)


class DenseIndex:
    """Dense retrieval index using FAISS IndexFlatIP for exact cosine similarity."""

    def __init__(
        self,
        passage_store: PassageStore | None = None,
        passage_ids: list[str] | None = None,
        index: faiss.IndexFlatIP | None = None,
        encoder: DenseEncoder | None = None,
        metadata: DenseIndexMetadata | None = None,
        index_dir: Path | None = None,
    ) -> None:
        self.passage_store: PassageStore = passage_store if passage_store is not None else PassageStore([])
        self.passage_ids: list[str] = list(passage_ids) if passage_ids is not None else self.passage_store.passage_ids
        self.index: faiss.IndexFlatIP | None = index
        self.encoder: DenseEncoder | None = encoder
        self.metadata: DenseIndexMetadata | None = metadata
        self.index_dir: Path | None = Path(index_dir) if index_dir else None

    @classmethod
    def build(
        cls,
        passage_store: PassageStore,
        encoder: DenseEncoder,
        index_name: str = "dense_default",
        batch_size: int = 32,
        extra_metadata: dict[str, Any] | None = None,
    ) -> DenseIndex:
        """Build and populate FAISS IndexFlatIP from passage texts in exact sequential order."""
        n_passages = len(passage_store)
        logger.info(
            "Building dense index '%s' across %d passages (model=%s, dim=%d)...",
            index_name,
            n_passages,
            encoder.model_name_or_path,
            encoder.dimension,
        )

        passage_ids = passage_store.passage_ids
        index = faiss.IndexFlatIP(encoder.dimension)

        if n_passages > 0:
            doc_texts = [p.text for p in passage_store.passages]
            doc_embeddings = encoder.encode_documents(doc_texts, batch_size=batch_size)
            if not doc_embeddings.flags["C_CONTIGUOUS"]:
                doc_embeddings = np.ascontiguousarray(doc_embeddings)
            index.add(doc_embeddings)

        corpus_fp = passage_store.compute_fingerprint()
        config_fp = compute_dense_config_fingerprint(
            model_name_or_path=encoder.model_name_or_path,
            dimension=encoder.dimension,
            normalize_embeddings=encoder.normalize_embeddings,
            query_prefix=encoder.query_prefix,
            document_prefix=encoder.document_prefix,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        metadata = DenseIndexMetadata(
            index_name=index_name,
            model_name_or_path=encoder.model_name_or_path,
            dimension=encoder.dimension,
            normalize_embeddings=encoder.normalize_embeddings,
            index_type="IndexFlatIP",
            corpus_fingerprint=corpus_fp,
            config_fingerprint=config_fp,
            query_prefix=encoder.query_prefix,
            document_prefix=encoder.document_prefix,
            batch_size=batch_size,
            total_passages=n_passages,
            created_at=now_iso,
            files={
                "index": "dense_index.faiss",
                "passage_ids": "passage_ids.json",
                "passages": "passages.jsonl",
                "metadata": "index_metadata.json",
            },
            extra=dict(extra_metadata or {}),
        )

        return cls(
            passage_store=passage_store,
            passage_ids=passage_ids,
            index=index,
            encoder=encoder,
            metadata=metadata,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Query the index and return top-k ranked passages with cosine similarity scores."""
        if self.index is None or self.index.ntotal == 0 or self.encoder is None:
            return []

        q_vec = self.encoder.encode_queries([query])
        if not q_vec.flags["C_CONTIGUOUS"]:
            q_vec = np.ascontiguousarray(q_vec)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k)

        results: list[SearchResult] = []
        rank = 1

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            score_val = float(score)
            if min_score is not None and score_val < min_score:
                continue

            passage_id = self.passage_ids[idx]
            passage = self.passage_store.get_by_id(passage_id)
            if passage is None:
                # Fallback to positional lookup
                passage = self.passage_store.passages[idx]

            results.append(
                SearchResult(
                    passage_id=passage_id,
                    score=score_val,
                    rank=rank,
                    passage=passage,
                )
            )
            rank += 1

        return results

    def save(self, index_dir: Path | str) -> Path:
        """Persist FAISS index, ordered passage-ID mapping, passages, and metadata."""
        target = Path(index_dir)
        target.mkdir(parents=True, exist_ok=True)

        # 1. Save FAISS index
        faiss_file = target / "dense_index.faiss"
        if self.index is not None:
            faiss.write_index(self.index, str(faiss_file))

        # 2. Save separate ordered passage-ID mapping
        ids_file = target / "passage_ids.json"
        with ids_file.open("w", encoding="utf-8") as f:
            json.dump(self.passage_ids, f, indent=2)

        # 3. Save passage store JSONL
        self.passage_store.save_jsonl(target / "passages.jsonl")

        # 4. Save metadata
        if self.metadata is not None:
            meta_file = target / "index_metadata.json"
            with meta_file.open("w", encoding="utf-8") as f:
                json.dump(self.metadata.to_dict(), f, indent=2)

        self.index_dir = target
        logger.info("Saved dense index and %d passages to %s", len(self.passage_ids), target)
        return target

    @classmethod
    def load(
        cls,
        index_dir: Path | str,
        encoder: DenseEncoder | None = None,
    ) -> DenseIndex:
        """Load a persisted dense index, passage-ID mapping, and passage store."""
        target = Path(index_dir)
        if not target.is_dir():
            raise FileNotFoundError(f"Dense index directory does not exist: {target}")

        faiss_file = target / "dense_index.faiss"
        ids_file = target / "passage_ids.json"
        passages_file = target / "passages.jsonl"
        meta_file = target / "index_metadata.json"

        if not faiss_file.is_file():
            raise FileNotFoundError(f"FAISS index file missing: {faiss_file}")
        if not ids_file.is_file():
            raise FileNotFoundError(f"Passage ID mapping missing: {ids_file}")
        if not passages_file.is_file():
            raise FileNotFoundError(f"Passage store missing: {passages_file}")
        if not meta_file.is_file():
            raise FileNotFoundError(f"Metadata file missing: {meta_file}")

        with meta_file.open("r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        metadata = DenseIndexMetadata.from_dict(meta_dict)

        with ids_file.open("r", encoding="utf-8") as f:
            passage_ids = json.load(f)

        passage_store = PassageStore.load_jsonl(passages_file)
        index = faiss.read_index(str(faiss_file))

        if encoder is None:
            encoder = DenseEncoder(
                model_name_or_path=metadata.model_name_or_path,
                normalize_embeddings=metadata.normalize_embeddings,
                query_prefix=metadata.query_prefix,
                document_prefix=metadata.document_prefix,
            )

        return cls(
            passage_store=passage_store,
            passage_ids=passage_ids,
            index=index,
            encoder=encoder,
            metadata=metadata,
            index_dir=target,
        )

    @classmethod
    def build_or_load(
        cls,
        passage_store: PassageStore,
        index_dir: Path | str,
        encoder: DenseEncoder,
        batch_size: int = 32,
        force_rebuild: bool = False,
    ) -> DenseIndex:
        """Build or reuse cached dense index based on corpus and config fingerprints."""
        target = Path(index_dir)
        meta_file = target / "index_metadata.json"
        faiss_file = target / "dense_index.faiss"
        ids_file = target / "passage_ids.json"
        psg_file = target / "passages.jsonl"

        current_corpus_fp = passage_store.compute_fingerprint()
        current_config_fp = compute_dense_config_fingerprint(
            model_name_or_path=encoder.model_name_or_path,
            dimension=encoder.dimension,
            normalize_embeddings=encoder.normalize_embeddings,
            query_prefix=encoder.query_prefix,
            document_prefix=encoder.document_prefix,
        )

        if not force_rebuild and meta_file.is_file() and faiss_file.is_file() and ids_file.is_file() and psg_file.is_file():
            try:
                with meta_file.open("r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
                cached_corpus_fp = meta_dict.get("corpus_fingerprint")
                cached_config_fp = meta_dict.get("config_fingerprint")

                if cached_corpus_fp == current_corpus_fp and cached_config_fp == current_config_fp:
                    logger.info("Corpus and config unchanged. Reusing cached dense index from %s", target)
                    return cls.load(target, encoder=encoder)
            except Exception as exc:
                logger.warning("Could not read cached metadata (%s). Rebuilding dense index...", exc)

        dense_index = cls.build(
            passage_store=passage_store,
            encoder=encoder,
            index_name=target.name,
            batch_size=batch_size,
        )
        dense_index.save(target)
        return dense_index


def run_build_dense_index(
    processed_dir: str | Path = "data/processed",
    datasets: list[str] | None = None,
    splits: list[str] | None = None,
    index_dir: str | Path = "data/indexes/dense_default",
    model_name_or_path: str = "BAAI/bge-small-en-v1.5",
    device: str = "auto",
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    force_rebuild: bool = False,
    smoke_test: bool = False,
    test_query: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute dense retrieval corpus preparation and FAISS index build pipeline."""
    from veritas.config import ROOT_DIR, load_config
    from veritas.retrieval.corpus import build_corpus_from_processed, create_synthetic_corpus

    if config_path:
        cfg = load_config(config_path)
        retrieval_cfg = cfg.get("retrieval", {})
        dense_cfg = retrieval_cfg.get("dense", retrieval_cfg)
        if datasets is None and "datasets" in retrieval_cfg:
            datasets = retrieval_cfg["datasets"]
        if "index_dir" in dense_cfg and index_dir == "data/indexes/dense_default":
            index_dir = dense_cfg["index_dir"]
        model_name_or_path = dense_cfg.get("model_name_or_path", model_name_or_path)
        device = dense_cfg.get("device", device)
        batch_size = dense_cfg.get("batch_size", batch_size)
        normalize_embeddings = dense_cfg.get("normalize_embeddings", normalize_embeddings)
        query_prefix = dense_cfg.get("query_prefix", query_prefix)
        document_prefix = dense_cfg.get("document_prefix", document_prefix)

    resolved_index_dir = Path(index_dir)
    if not resolved_index_dir.is_absolute():
        resolved_index_dir = ROOT_DIR / resolved_index_dir

    logger.info("=" * 65)
    logger.info("VERITAS Dense Retrieval (FAISS) Index Build")
    logger.info("=" * 65)
    logger.info("Target index dir:  %s", resolved_index_dir)
    logger.info("Model name:        %s", model_name_or_path)
    logger.info("Normalize L2:      %s", normalize_embeddings)
    logger.info("Device:            %s", device)
    logger.info("Force rebuild:     %s", force_rebuild)

    if smoke_test:
        logger.info("Running SMOKE-TEST with deterministic MockEmbedder...")
        synthetic_records = [
            {
                "id": "claim_test_1",
                "evidence": [
                    {
                        "text": "The World Health Organization confirmed that vaccination reduces severe COVID-19 hospitalizations by over 90 percent.",
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
                        "text": "The Bureau of Labor Statistics reported that nonfarm payroll employment rose by 275,000 in February.",
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
                        "text": "NASA's James Webb Space Telescope detected carbon dioxide and methane in exoplanet atmosphere.",
                        "url": "https://www.nasa.gov/universe/webb-discovers-methane/",
                        "date": "2023-09-11",
                        "domain": "nasa.gov",
                    }
                ],
            },
        ]
        passage_store = create_synthetic_corpus(synthetic_records)
        encoder = DenseEncoder(
            model_name_or_path="mock-embedder",
            device="cpu",
            normalize_embeddings=normalize_embeddings,
            embedder=MockEmbedder(dimension=16),
        )
        summary = {
            "mode": "smoke_test",
            "total_passages_created": len(passage_store),
            "datasets": {"synthetic": len(synthetic_records)},
        }
    else:
        resolved_processed = Path(processed_dir)
        if not resolved_processed.is_absolute():
            resolved_processed = ROOT_DIR / resolved_processed

        passage_store, summary = build_corpus_from_processed(
            processed_dir=resolved_processed,
            datasets=datasets,
            splits=splits,
            deduplicate=True,
        )
        encoder = DenseEncoder(
            model_name_or_path=model_name_or_path,
            device=device,
            normalize_embeddings=normalize_embeddings,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        )

    dense_index = DenseIndex.build_or_load(
        passage_store=passage_store,
        index_dir=resolved_index_dir,
        encoder=encoder,
        batch_size=batch_size,
        force_rebuild=force_rebuild,
    )

    result_summary = {
        "status": "success",
        "index_dir": str(resolved_index_dir),
        "total_passages": len(passage_store),
        "model_name_or_path": encoder.model_name_or_path,
        "dimension": encoder.dimension,
        "normalize_embeddings": encoder.normalize_embeddings,
        "corpus_summary": summary,
    }

    if test_query:
        logger.info("Executing test query: %r", test_query)
        search_results = dense_index.search(test_query, top_k=3)
        result_summary["test_query_results"] = [r.to_dict() for r in search_results]
        logger.info("Retrieved %d matches for test query.", len(search_results))
        for r in search_results:
            snippet = r.passage.text[:80] + "..." if len(r.passage.text) > 80 else r.passage.text
            logger.info("  Rank %d [score=%.3f]: %s | %s", r.rank, r.score, r.passage_id, snippet)

    logger.info("=" * 65)
    logger.info("Dense Index Build Completed Successfully.")
    logger.info("Passages in index: %d", len(passage_store))
    logger.info("=" * 65)

    return result_summary


def main() -> None:
    """CLI entry point for building dense index."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Build or update VERITAS Dense retrieval index (FAISS) from evidence corpus."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g. configs/retrieval_dense.yaml).",
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
        default="data/indexes/dense_default",
        help="Output directory to store index, passage IDs, and metadata.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="SentenceTransformers model name or path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild even if cached index matches corpus and config.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build a fast synthetic index using MockEmbedder without model download.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional test query to run against the index after building.",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        run_build_dense_index(
            processed_dir=args.processed_dir,
            datasets=args.datasets,
            splits=args.splits,
            index_dir=args.index_dir,
            model_name_or_path=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            force_rebuild=args.force_rebuild,
            smoke_test=args.smoke_test,
            test_query=args.query,
            config_path=args.config,
        )
    except Exception as exc:
        logger.exception("Dense index build failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
