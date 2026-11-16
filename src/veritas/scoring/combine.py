"""Score combination module for VERITAS (P4).

Combines relevance, domain credibility, and temporal alignment into a unified
score for candidate evidence passages:

    combined = alpha * relevance + beta * credibility + gamma * temporal

Guarantees:
  1. All component scores (relevance, credibility, temporal) and the combined
     score are strictly in [0.0, 1.0].
  2. When beta = gamma = 0, the pipeline performs pure relevance ranking.
  3. Weights (alpha, beta, gamma) are normalized so their sum equals 1.0.
  4. Full provenance and audit metadata are retained on every scored candidate.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from veritas.retrieval.reranker import normalize_scores
from veritas.retrieval.schema import Passage, SearchResult
from veritas.scoring.credibility import (
    CredibilityConfig,
    CredibilityScorer,
    extract_domain,
)
from veritas.scoring.temporal import (
    TemporalConfig,
    TemporalScorer,
)

logger = logging.getLogger("veritas.scoring.combine")


# ---------------------------------------------------------------------------
# Weights Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoreWeights:
    """Weights governing the relative contribution of each scoring dimension.

    Formula:
        combined = alpha * relevance + beta * credibility + gamma * temporal

    Attributes:
        alpha: Weight for relevance score (from BM25, Dense, or CrossEncoder).
        beta: Weight for domain credibility tier score.
        gamma: Weight for temporal recency alignment score.
        normalize_weights: When True, normalizes weights such that alpha + beta + gamma = 1.0.
    """

    alpha: float = 0.60
    beta: float = 0.25
    gamma: float = 0.15
    normalize_weights: bool = True

    def __post_init__(self) -> None:
        if self.alpha < 0.0 or self.beta < 0.0 or self.gamma < 0.0:
            raise ValueError(
                f"Weights must be non-negative: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}"
            )

    def effective_weights(self) -> tuple[float, float, float]:
        """Compute the effective (optionally normalized) weights (alpha, beta, gamma)."""
        a, b, g = float(self.alpha), float(self.beta), float(self.gamma)
        total = a + b + g
        if total == 0.0:
            return 0.0, 0.0, 0.0
        if self.normalize_weights:
            return a / total, b / total, g / total
        return a, b, g

    def to_dict(self) -> dict[str, Any]:
        eff_a, eff_b, eff_g = self.effective_weights()
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "normalize_weights": self.normalize_weights,
            "effective_alpha": round(eff_a, 4),
            "effective_beta": round(eff_b, 4),
            "effective_gamma": round(eff_g, 4),
        }


# ---------------------------------------------------------------------------
# ScoredCandidate Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoredCandidate:
    """A candidate passage with individual and combined relevance/credibility/temporal scores.

    Attributes:
        passage_id: Unique identifier of the evidence passage.
        combined_score: Final unified score in [0.0, 1.0].
        rank: 1-indexed rank after combination sorting (1 = highest score).
        relevance_score: Normalized relevance component in [0.0, 1.0].
        credibility_score: Domain credibility component in [0.0, 1.0].
        temporal_score: Recency alignment component in [0.0, 1.0].
        passage: Source Passage object if available.
        domain: Extracted source domain string.
        evidence_date: Extracted evidence publication date string.
        claim_date: Claim temporal anchor date string.
        metadata: Audit metadata including effective weights and raw scores.
    """

    passage_id: str
    combined_score: float
    rank: int
    relevance_score: float
    credibility_score: float
    temporal_score: float
    passage: Passage | None = None
    domain: str | None = None
    evidence_date: str | None = None
    claim_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert ScoredCandidate to a serializable dictionary."""
        return {
            "passage_id": self.passage_id,
            "combined_score": round(float(self.combined_score), 6),
            "rank": int(self.rank),
            "relevance_score": round(float(self.relevance_score), 6),
            "credibility_score": round(float(self.credibility_score), 6),
            "temporal_score": round(float(self.temporal_score), 6),
            "domain": self.domain,
            "evidence_date": self.evidence_date,
            "claim_date": self.claim_date,
            "passage": self.passage.to_dict() if self.passage else None,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# CombinedScorerConfig
# ---------------------------------------------------------------------------

@dataclass
class CombinedScorerConfig:
    """Full configuration for P4 score combination.

    Attributes:
        weights: ScoreWeights instance defining alpha, beta, gamma.
        credibility: CredibilityConfig instance.
        temporal: TemporalConfig instance.
        relevance_normalization: Method for mapping raw relevance scores to [0.0, 1.0].
    """

    weights: ScoreWeights = field(default_factory=ScoreWeights)
    credibility: CredibilityConfig = field(default_factory=CredibilityConfig.default_provisional)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    relevance_normalization: str = "sigmoid"

    @classmethod
    def from_yaml(cls, path: str | Path) -> CombinedScorerConfig:
        """Load configuration from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Scoring config file not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        scoring_data = raw.get("scoring", raw)

        # Weights
        w_data = scoring_data.get("weights", {})
        weights = ScoreWeights(
            alpha=float(w_data.get("alpha", 0.60)),
            beta=float(w_data.get("beta", 0.25)),
            gamma=float(w_data.get("gamma", 0.15)),
            normalize_weights=bool(scoring_data.get("normalize_weights", True)),
        )

        # Credibility
        c_data = scoring_data.get("credibility", {})
        tier_path = c_data.get("tier_config_path")
        if tier_path:
            # Resolve relative to p.parent or root
            tier_file = Path(tier_path)
            if not tier_file.is_absolute():
                tier_file = p.parent / tier_file
                if not tier_file.exists():
                    tier_file = Path(tier_path)  # try as-is
            credibility = CredibilityConfig.from_yaml(tier_file)
            if "unknown_domain_score" in c_data:
                credibility.unknown_domain_score = float(c_data["unknown_domain_score"])
            if "random_mode" in c_data:
                credibility.random_mode = bool(c_data["random_mode"])
            if "random_seed" in c_data:
                credibility.random_seed = int(c_data["random_seed"])
        else:
            credibility = CredibilityConfig.default_provisional()

        # Temporal
        t_data = scoring_data.get("temporal", {})
        temporal = TemporalConfig.from_dict(t_data)

        rel_norm = scoring_data.get("relevance_normalization", "sigmoid")

        return cls(
            weights=weights,
            credibility=credibility,
            temporal=temporal,
            relevance_normalization=rel_norm,
        )


# ---------------------------------------------------------------------------
# CombinedScorer Class
# ---------------------------------------------------------------------------

class CombinedScorer:
    """Unified evidence scorer combining relevance, credibility, and temporal signals.

    Score:
        combined = alpha * relevance + beta * credibility + gamma * temporal
    """

    def __init__(
        self,
        config: CombinedScorerConfig | dict[str, Any] | None = None,
        credibility_scorer: CredibilityScorer | None = None,
        temporal_scorer: TemporalScorer | None = None,
    ) -> None:
        if config is None:
            self.config = CombinedScorerConfig()
        elif isinstance(config, dict):
            # Parse dict
            w_dict = config.get("weights", {})
            weights = ScoreWeights(
                alpha=float(w_dict.get("alpha", 0.60)),
                beta=float(w_dict.get("beta", 0.25)),
                gamma=float(w_dict.get("gamma", 0.15)),
                normalize_weights=bool(config.get("normalize_weights", True)),
            )
            c_config = config.get("credibility")
            if isinstance(c_config, dict):
                c_dict = dict(c_config)
                tier_path = c_dict.pop("tier_config_path", None)
                if tier_path:
                    tier_file = Path(tier_path)
                    if not tier_file.is_absolute():
                        from veritas.config import ROOT_DIR
                        tier_file = ROOT_DIR / tier_file
                    if tier_file.exists():
                        credibility = CredibilityConfig.from_yaml(tier_file)
                        for k, v in c_dict.items():
                            if hasattr(credibility, k):
                                setattr(credibility, k, v)
                    else:
                        credibility = CredibilityConfig(**c_dict)
                else:
                    credibility = CredibilityConfig(**c_dict)
            elif isinstance(c_config, CredibilityConfig):
                credibility = c_config
            else:
                credibility = CredibilityConfig.default_provisional()

            t_config = config.get("temporal")
            if isinstance(t_config, dict):
                temporal = TemporalConfig.from_dict(t_config)
            elif isinstance(t_config, TemporalConfig):
                temporal = t_config
            else:
                temporal = TemporalConfig()

            self.config = CombinedScorerConfig(
                weights=weights,
                credibility=credibility,
                temporal=temporal,
                relevance_normalization=config.get("relevance_normalization", "sigmoid"),
            )
        elif isinstance(config, CombinedScorerConfig):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config).__name__}")

        self.credibility_scorer = credibility_scorer or CredibilityScorer(self.config.credibility)
        self.temporal_scorer = temporal_scorer or TemporalScorer(self.config.temporal)

    def _extract_candidate_data(
        self,
        candidate: Any,
        default_index: int,
    ) -> tuple[str, float, Passage | None, str | None, str | None]:
        """Extract (passage_id, raw_rel_score, passage_obj, domain, date) from candidate."""
        pid = getattr(candidate, "passage_id", getattr(candidate, "id", f"cand_{default_index}"))
        raw_score = getattr(candidate, "score", 0.0)
        passage_obj = getattr(candidate, "passage", None)
        domain = getattr(candidate, "domain", None)
        date = getattr(candidate, "date", getattr(candidate, "publication_date", None))

        if isinstance(candidate, Passage):
            pid = candidate.passage_id
            passage_obj = candidate
            domain = candidate.domain or candidate.url
            date = candidate.date
            raw_score = 1.0  # Default unranked passage score

        elif isinstance(candidate, dict):
            pid = candidate.get("passage_id") or candidate.get("id") or f"cand_{default_index}"
            raw_score = candidate.get("score", 0.0)
            domain = candidate.get("domain") or candidate.get("url")
            date = candidate.get("date") or candidate.get("publication_date")
            p_data = candidate.get("passage")
            if isinstance(p_data, Passage):
                passage_obj = p_data
            elif isinstance(p_data, dict):
                try:
                    passage_obj = Passage.from_dict(p_data)
                except Exception:
                    passage_obj = None

        if passage_obj:
            if not domain:
                domain = passage_obj.domain or passage_obj.url
            if not date:
                date = passage_obj.date

        return str(pid), float(raw_score), passage_obj, domain, date

    def score_candidates(
        self,
        candidates: Sequence[Any],
        claim_date: str | None = None,
        claim_type: str | None = None,
        weights_override: ScoreWeights | tuple[float, float, float] | None = None,
        top_n: int | None = None,
    ) -> list[ScoredCandidate]:
        """Score candidate evidence passages and sort them descending by combined score.

        Formula:
            combined = alpha * relevance + beta * credibility + gamma * temporal

        Args:
            candidates: Sequence of candidates (RerankResult, SearchResult, Passage, dict).
            claim_date: Anchor ISO date of the claim.
            claim_type: Category of claim for temporal decay selection.
            weights_override: Optional custom weights (alpha, beta, gamma).
            top_n: Optional number of top candidates to return.

        Returns:
            List of ScoredCandidate instances ordered descending by combined_score.
        """
        if not candidates:
            return []

        # Resolve weights
        if weights_override is not None:
            if isinstance(weights_override, tuple):
                eff_weights = ScoreWeights(
                    alpha=weights_override[0],
                    beta=weights_override[1],
                    gamma=weights_override[2],
                    normalize_weights=self.config.weights.normalize_weights,
                )
            else:
                eff_weights = weights_override
        else:
            eff_weights = self.config.weights

        alpha, beta, gamma = eff_weights.effective_weights()

        # Step 1: Extract candidate data & raw relevance scores
        extracted = []
        raw_rel_scores = []
        for idx, c in enumerate(candidates):
            pid, raw_score, passage_obj, domain, date = self._extract_candidate_data(c, idx)
            extracted.append((pid, raw_score, passage_obj, domain, date, idx))
            raw_rel_scores.append(raw_score)

        # Step 2: Normalize relevance scores into [0.0, 1.0]
        # If candidate already has score in [0.0, 1.0] (e.g. from RerankResult),
        # check if normalization is needed. If scores already in [0, 1], preserve them.
        arr_rel = np.asarray(raw_rel_scores, dtype=np.float32)
        if np.all(arr_rel >= 0.0) and np.all(arr_rel <= 1.0) and not (np.all(arr_rel == 0.0)):
            norm_rel_scores = arr_rel
        else:
            norm_rel_scores = normalize_scores(arr_rel, method=self.config.relevance_normalization)

        # Step 3: Compute credibility and temporal components
        scored_items: list[ScoredCandidate] = []
        for i, (pid, raw_score, passage_obj, domain, date, orig_idx) in enumerate(extracted):
            r_score = float(np.clip(norm_rel_scores[i], 0.0, 1.0))

            # Credibility component in [0.0, 1.0]
            if beta > 0.0:
                c_score = float(self.credibility_scorer.score_domain(domain))
            else:
                c_score = 0.0

            # Temporal component in [0.0, 1.0]
            if gamma > 0.0:
                t_score = float(self.temporal_scorer.score(claim_date=claim_date, evidence_date=date, claim_type=claim_type))
            else:
                t_score = 0.0

            # Combined score calculation
            # Note: when beta=gamma=0, combined is purely alpha * r_score = 1.0 * r_score = r_score
            combined = alpha * r_score + beta * c_score + gamma * t_score
            combined_clamped = float(np.clip(combined, 0.0, 1.0))

            norm_domain = extract_domain(domain)

            scored_items.append(
                ScoredCandidate(
                    passage_id=pid,
                    combined_score=round(combined_clamped, 6),
                    rank=0,  # assigned after sorting
                    relevance_score=round(r_score, 6),
                    credibility_score=round(c_score, 6),
                    temporal_score=round(t_score, 6),
                    passage=passage_obj,
                    domain=norm_domain or None,
                    evidence_date=str(date) if date else None,
                    claim_date=str(claim_date) if claim_date else None,
                    metadata={
                        "raw_relevance": raw_score,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "orig_idx": orig_idx,
                    },
                )
            )

        # Step 4: Sort descending by combined_score (stable secondary tie-break by orig_idx)
        scored_items.sort(key=lambda x: (-x.combined_score, x.metadata.get("orig_idx", 0)))

        # Assign 1-indexed ranks
        for rank_idx, item in enumerate(scored_items, start=1):
            item.rank = rank_idx

        # Truncate top_n if requested
        if top_n is not None and top_n > 0:
            scored_items = scored_items[:top_n]

        return scored_items
