"""Relevance reranking component for VERITAS (P4).

Scores claim-passage candidate pairs using a configurable cross-encoder model,
normalizes scores onto a [0.0, 1.0] scale, and returns top-N ranked candidates
with full provenance.

IMPORTANT DISCLAIMER ON SCORE INTERPRETATION:
---------------------------------------------
The normalized relevance scores produced by this module represent monotonic
relative ranking signals on a [0.0, 1.0] scale. They are NOT calibrated
frequentist or Bayesian probabilities and should NEVER be interpreted as
posterior probabilities that a claim or passage is true, nor as calibrated
likelihoods of factual relevance.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from veritas.retrieval.schema import Passage, SearchResult

logger = logging.getLogger("veritas.retrieval.reranker")

# Supported score normalization methods
VALID_NORMALIZATION_METHODS = ("sigmoid", "minmax", "softmax", "none")
VALID_TIE_BREAKERS = ("original", "passage_id")


# ---------------------------------------------------------------------------
# Score Normalization Functions
# ---------------------------------------------------------------------------

def normalize_scores(
    scores: np.ndarray | Sequence[float],
    method: str = "sigmoid",
) -> np.ndarray:
    """Normalize raw cross-encoder scalar logits onto a [0.0, 1.0] interval.

    Normalization Methods:
    ----------------------
    1. 'sigmoid':
       Maps unbounded logits s in (-inf, +inf) to (0.0, 1.0) via:
           sigma(s) = 1.0 / (1.0 + exp(-s))
       Extreme values are clipped to [-50.0, 50.0] to prevent arithmetic overflow.
       Sigmoid is the recommended default for cross-encoders because it evaluates
       each claim-passage pair independently without artificially stretching narrow
       score distributions when all candidates are uniformly relevant or irrelevant.

    2. 'minmax':
       Linearly rescales the candidate batch to span [0.0, 1.0]:
           (s - min(s)) / (max(s) - min(s))
       If all scores in the batch are identical (variance is zero), assigns 1.0
       to all candidates.

    3. 'softmax':
       Exponentiates logits with max-subtraction for numerical stability and divides
       by the sum:
           exp(s_i - max(s)) / sum_j(exp(s_j - max(s)))
       Scores sum to 1.0 across the candidate set.

    4. 'none':
       Retains raw scores, clipping strictly to [0.0, 1.0].

    DISCLAIMER:
    -----------
    Normalized scores are monotonic transformations of model logits.
    They are NOT calibrated probabilities.

    Args:
        scores: 1D array or sequence of raw numerical scores / logits.
        method: Normalization method name ('sigmoid', 'minmax', 'softmax', 'none').

    Returns:
        1D numpy float32 array with all values in [0.0, 1.0].

    Raises:
        ValueError: If method is not one of the supported methods.
    """
    method_lower = method.lower().strip()
    if method_lower not in VALID_NORMALIZATION_METHODS:
        raise ValueError(
            f"Unknown normalization method {method!r}. "
            f"Must be one of: {list(VALID_NORMALIZATION_METHODS)}"
        )

    scores_arr = np.asarray(scores, dtype=np.float32)
    if scores_arr.size == 0:
        return np.empty((0,), dtype=np.float32)

    if method_lower == "sigmoid":
        clipped = np.clip(scores_arr, -50.0, 50.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)

    elif method_lower == "minmax":
        s_min = float(np.min(scores_arr))
        s_max = float(np.max(scores_arr))
        diff = s_max - s_min
        if math.isclose(diff, 0.0, abs_tol=1e-9):
            # All candidates have identical scores; assign 1.0
            return np.ones_like(scores_arr, dtype=np.float32)
        return ((scores_arr - s_min) / diff).astype(np.float32)

    elif method_lower == "softmax":
        shifted = scores_arr - float(np.max(scores_arr))
        clipped = np.clip(shifted, -50.0, 50.0)
        exp_s = np.exp(clipped)
        sum_exp = float(np.sum(exp_s))
        if math.isclose(sum_exp, 0.0, abs_tol=1e-9):
            return (np.ones_like(scores_arr) / scores_arr.size).astype(np.float32)
        return (exp_s / sum_exp).astype(np.float32)

    else:  # "none"
        return np.clip(scores_arr, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Protocols and Mocks
# ---------------------------------------------------------------------------

class CrossEncoderProtocol(Protocol):
    """Protocol matching sentence_transformers.CrossEncoder.predict interface."""

    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | list[float]: ...


class MockCrossEncoder:
    """Deterministic, lightweight Cross-Encoder for unit tests.

    Requires NO GPU, downloads NO weights, and makes NO network calls.
    Allows testing reranker sorting, normalization, empty inputs, ties,
    and metadata persistence in isolation.
    """

    def __init__(
        self,
        canned_scores: list[float] | None = None,
        score_map: dict[tuple[str, str], float] | None = None,
        score_fn: Callable[[str, str], float] | None = None,
        model_name: str = "mock-cross-encoder",
        model_version: str = "0.1.0",
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.canned_scores = list(canned_scores) if canned_scores is not None else None
        self.score_map = dict(score_map) if score_map is not None else None
        self.score_fn = score_fn
        self.history: list[dict[str, Any]] = []

    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        call_record = {
            "num_pairs": len(sentences),
            "batch_size": batch_size,
            "sentences": list(sentences),
            "kwargs": kwargs,
        }
        self.history.append(call_record)

        if not sentences:
            return np.empty((0,), dtype=np.float32)

        results: list[float] = []
        for idx, pair in enumerate(sentences):
            text_a, text_b = pair[0], pair[1]

            if self.canned_scores is not None:
                if idx < len(self.canned_scores):
                    results.append(float(self.canned_scores[idx]))
                else:
                    results.append(0.0)
            elif self.score_map is not None:
                # Check (text_a, text_b) or fallback
                score = self.score_map.get((text_a, text_b))
                if score is None:
                    # Also try partial matching on second item (e.g. passage id or snippet)
                    score = next(
                        (v for k, v in self.score_map.items() if k[1] in text_b or text_b in k[1]),
                        0.0,
                    )
                results.append(float(score))
            elif self.score_fn is not None:
                results.append(float(self.score_fn(text_a, text_b)))
            else:
                # Deterministic pseudo-random score from SHA-256 of pair
                h = hashlib.sha256(f"{text_a}::{text_b}".encode("utf-8")).digest()
                raw_int = int.from_bytes(h[:4], "little", signed=True)
                # Map to roughly [-5.0, 5.0] logit range
                score = float(raw_int) / (2**31) * 5.0
                results.append(score)

        return np.asarray(results, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RerankerConfig:
    """Configuration for the Cross-Encoder relevance reranker.

    Attributes:
        model_name: Identifier for the cross-encoder model (e.g.
            'cross-encoder/ms-marco-MiniLM-L-6-v2').
        model_version: Version tag or checkpoint revision of the model.
        device: Execution device ('auto', 'cpu', 'cuda').
        batch_size: Batch size for cross-encoder inference.
        max_length: Maximum sequence length (tokens) for claim-passage pairs.
        top_n: Optional default maximum number of candidates to retain after reranking.
        normalization_method: Score normalization strategy ('sigmoid', 'minmax', 'softmax', 'none').
        tie_breaker: Strategy for breaking score ties ('original' preserves input rank,
            'passage_id' sorts ascending by ID).
        extra: Additional custom configuration options.
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_version: str = "1.0.0"
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    top_n: int | None = None
    normalization_method: str = "sigmoid"
    tie_breaker: str = "original"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.normalization_method not in VALID_NORMALIZATION_METHODS:
            raise ValueError(
                f"Invalid normalization_method {self.normalization_method!r}. "
                f"Must be one of {list(VALID_NORMALIZATION_METHODS)}"
            )
        if self.tie_breaker not in VALID_TIE_BREAKERS:
            raise ValueError(
                f"Invalid tie_breaker {self.tie_breaker!r}. "
                f"Must be one of {list(VALID_TIE_BREAKERS)}"
            )
        if self.top_n is not None and self.top_n < 0:
            raise ValueError(f"top_n must be non-negative or None, got {self.top_n}")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RerankerConfig:
        """Create RerankerConfig from dictionary."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for RerankerConfig, got {type(data).__name__}")
        valid_keys = {
            "model_name",
            "model_version",
            "device",
            "batch_size",
            "max_length",
            "top_n",
            "normalization_method",
            "tie_breaker",
            "extra",
        }
        known = {k: v for k, v in data.items() if k in valid_keys}
        extra = dict(data.get("extra", {}))
        for k, v in data.items():
            if k not in valid_keys:
                extra[k] = v
        known["extra"] = extra
        return cls(**known)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RerankerConfig:
        """Load RerankerConfig from a YAML file."""
        import yaml

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Reranker config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # Support nesting under 'reranker' key if present
        cfg_dict = raw.get("reranker", raw)
        return cls.from_dict(cfg_dict)


@dataclass
class RerankResult:
    """Result of scoring and ranking a single candidate passage.

    Attributes:
        passage_id: Unique identifier of the passage.
        score: Normalized relevance score in [0.0, 1.0].
        rank: 1-indexed rank after reranking (1 = most relevant).
        raw_score: Original unnormalized logit directly from the model.
        passage: The original Passage object if available.
        original_rank: Rank of candidate prior to reranking (if available).
        metadata: Additional provenance and model metadata.
    """

    passage_id: str
    score: float
    rank: int
    raw_score: float
    passage: Passage | None = None
    original_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "passage_id": self.passage_id,
            "score": round(float(self.score), 6),
            "rank": int(self.rank),
            "raw_score": round(float(self.raw_score), 6),
            "original_rank": self.original_rank,
            "passage": self.passage.to_dict() if self.passage else None,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Candidate Helper
# ---------------------------------------------------------------------------

def _extract_candidate_info(
    candidate: Any,
    default_index: int = 0,
) -> tuple[str, str, Passage | None, int | None]:
    """Extract (passage_id, text, passage_obj, original_rank) from a candidate item.

    Supports:
        - Passage
        - SearchResult
        - FusedSearchResult
        - dict
        - Any object with passage_id and text attributes
    """
    if isinstance(candidate, Passage):
        return candidate.passage_id, candidate.text, candidate, default_index + 1

    if isinstance(candidate, SearchResult):
        return (
            candidate.passage_id,
            candidate.passage.text if candidate.passage else "",
            candidate.passage,
            candidate.rank,
        )

    # Check for FusedSearchResult or duck-typed result
    if hasattr(candidate, "passage_id") and hasattr(candidate, "passage"):
        passage_obj = getattr(candidate, "passage", None)
        text = passage_obj.text if (passage_obj and hasattr(passage_obj, "text")) else ""
        rank = getattr(candidate, "rank", default_index + 1)
        return str(candidate.passage_id), text, passage_obj, rank

    if isinstance(candidate, dict):
        pid = str(candidate.get("passage_id") or candidate.get("id") or f"cand_{default_index}")
        # Text might be top-level or inside a nested 'passage' dict
        text = candidate.get("text")
        if not text and isinstance(candidate.get("passage"), dict):
            text = candidate["passage"].get("text", "")
        text = str(text or "")

        # Try reconstructing Passage if dict has sufficient info
        passage_obj = None
        try:
            passage_obj = Passage.from_dict(candidate.get("passage", candidate))
        except Exception:
            if text:
                passage_obj = Passage(passage_id=pid, text=text)

        rank = candidate.get("rank", default_index + 1)
        return pid, text, passage_obj, rank

    # General fallback
    pid = getattr(candidate, "passage_id", getattr(candidate, "id", f"cand_{default_index}"))
    text = getattr(candidate, "text", "")
    return str(pid), str(text), None, default_index + 1


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Configurable Cross-Encoder Relevance Reranker for VERITAS P4.

    Evaluates full interaction between a claim and candidate passages by forming
    (claim, passage_text) sentence pairs and scoring them with a cross-encoder
    architecture. Scores are normalized to [0.0, 1.0] and candidates are ranked
    descending by relevance.

    IMPORTANT:
    ----------
    Normalized scores represent relative ranking signals. They are NOT calibrated
    probabilities of factual accuracy or relevance.
    """

    def __init__(
        self,
        config: RerankerConfig | dict[str, Any] | None = None,
        model: CrossEncoderProtocol | None = None,
    ) -> None:
        """Initialize the cross-encoder reranker.

        Args:
            config: Configuration instance or dictionary. If None, default is used.
            model: Optional pre-instantiated model conforming to CrossEncoderProtocol
                   (e.g., MockCrossEncoder or custom CrossEncoder). If None, the real
                   CrossEncoder is loaded lazily on first prediction.
        """
        if config is None:
            self.config = RerankerConfig()
        elif isinstance(config, dict):
            self.config = RerankerConfig.from_dict(config)
        elif isinstance(config, RerankerConfig):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config).__name__}")

        # Resolve device
        if self.config.device == "auto":
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = self.config.device

        self._model = model

    @property
    def model(self) -> CrossEncoderProtocol:
        """Lazy-load the underlying CrossEncoder model if not already injected."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(
                    "Loading CrossEncoder '%s' (version=%s) on %s...",
                    self.config.model_name,
                    self.config.model_version,
                    self.device,
                )
                self._model = CrossEncoder(
                    self.config.model_name,
                    max_length=self.config.max_length,
                    device=self.device,
                )
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required to load a real CrossEncoder. "
                    "Install it with 'pip install sentence-transformers' or supply "
                    "a MockCrossEncoder for testing."
                ) from exc
        return self._model

    def get_metadata(self) -> dict[str, Any]:
        """Return run metadata for auditability and configuration logging."""
        return {
            "component": "P4_relevance_reranker",
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "device": self.device,
            "normalization_method": self.config.normalization_method,
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "top_n": self.config.top_n,
            "tie_breaker": self.config.tie_breaker,
        }

    def rerank(
        self,
        claim: str,
        candidates: Sequence[Any],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Score and rerank candidate passages against a claim.

        Args:
            claim: The factual claim string.
            candidates: Sequence of candidate items (Passage, SearchResult,
                        FusedSearchResult, or dicts).
            top_n: Optional override for the number of candidates to return.
                   If None, falls back to config.top_n (or all if both are None).

        Returns:
            List of RerankResult objects ordered descending by relevance score,
            clipped to top_n if configured.
        """
        # Handle empty candidates immediately without invoking model
        if not candidates:
            return []

        # Handle empty claim
        clean_claim = str(claim).strip() if claim is not None else ""
        if not clean_claim:
            logger.warning("Empty or whitespace claim passed to rerank(); returning candidates with score=0.0")
            results = []
            for idx, c in enumerate(candidates):
                pid, _, passage_obj, orig_rank = _extract_candidate_info(c, idx)
                results.append(
                    RerankResult(
                        passage_id=pid,
                        score=0.0,
                        rank=idx + 1,
                        raw_score=0.0,
                        passage=passage_obj,
                        original_rank=orig_rank,
                        metadata={"claim_empty": True},
                    )
                )
            effective_top_n = top_n if top_n is not None else self.config.top_n
            if effective_top_n is not None and effective_top_n > 0:
                results = results[:effective_top_n]
            return results

        # Determine effective top_n
        effective_top_n = top_n if top_n is not None else self.config.top_n
        if effective_top_n is not None and effective_top_n <= 0:
            return []

        # Extract candidate details and form pairs
        extracted: list[tuple[str, str, Passage | None, int | None, int]] = []
        pairs: list[tuple[str, str]] = []

        for idx, candidate in enumerate(candidates):
            pid, text, passage_obj, orig_rank = _extract_candidate_info(candidate, idx)
            extracted.append((pid, text, passage_obj, orig_rank, idx))
            pairs.append((clean_claim, text))

        # Model inference
        raw_scores_arr = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        raw_scores = np.asarray(raw_scores_arr, dtype=np.float32)

        # Normalize scores to [0.0, 1.0]
        norm_scores = normalize_scores(raw_scores, method=self.config.normalization_method)

        # Build intermediate records for sorting
        scored_records: list[dict[str, Any]] = []
        for i, (pid, text, passage_obj, orig_rank, orig_idx) in enumerate(extracted):
            scored_records.append(
                {
                    "passage_id": pid,
                    "raw_score": float(raw_scores[i]),
                    "norm_score": float(norm_scores[i]),
                    "passage": passage_obj,
                    "orig_rank": orig_rank,
                    "orig_idx": orig_idx,
                }
            )

        # Deterministic sorting with tie-breaking
        # Primary key: -norm_score (descending)
        # Secondary key: depends on tie_breaker strategy
        if self.config.tie_breaker == "passage_id":
            scored_records.sort(key=lambda x: (-x["norm_score"], x["passage_id"]))
        else:  # "original"
            scored_records.sort(key=lambda x: (-x["norm_score"], x["orig_idx"]))

        # Truncate to top_n if requested
        if effective_top_n is not None and effective_top_n > 0:
            scored_records = scored_records[:effective_top_n]

        # Construct final typed RerankResult list with 1-indexed ranks
        results: list[RerankResult] = []
        for new_rank, item in enumerate(scored_records, start=1):
            results.append(
                RerankResult(
                    passage_id=item["passage_id"],
                    score=float(item["norm_score"]),
                    rank=new_rank,
                    raw_score=float(item["raw_score"]),
                    passage=item["passage"],
                    original_rank=item["orig_rank"],
                    metadata={
                        "model_name": self.config.model_name,
                        "model_version": self.config.model_version,
                        "normalization": self.config.normalization_method,
                    },
                )
            )

        return results


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def rerank_candidates(
    claim: str,
    candidates: Sequence[Any],
    top_n: int | None = None,
    config: RerankerConfig | dict[str, Any] | None = None,
    model: CrossEncoderProtocol | None = None,
) -> list[RerankResult]:
    """Convenience function to rerank candidate passages against a claim.

    Args:
        claim: Factual claim string.
        candidates: Candidate passages or results.
        top_n: Optional maximum number of results to return.
        config: Optional RerankerConfig or mapping.
        model: Optional CrossEncoderProtocol instance (e.g. MockCrossEncoder).

    Returns:
        List of RerankResult objects ordered by descending relevance.
    """
    reranker = CrossEncoderReranker(config=config, model=model)
    return reranker.rerank(claim=claim, candidates=candidates, top_n=top_n)
