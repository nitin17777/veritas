"""Validation-only grid search module for VERITAS scoring weights (P4).

Performs hyperparameter search over (alpha, beta, gamma, tau) exclusively on
a validation dataset. STRICTLY PROHIBITS test-set tuning to prevent data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from veritas.scoring.combine import CombinedScorer, CombinedScorerConfig, ScoreWeights
from veritas.scoring.credibility import CredibilityConfig
from veritas.scoring.temporal import TemporalConfig

logger = logging.getLogger("veritas.scoring.grid_search")

VALID_VALIDATION_SPLIT_NAMES = ("val", "validation", "dev", "tune")


# ---------------------------------------------------------------------------
# Simplex Weight Grid Generator
# ---------------------------------------------------------------------------

def generate_simplex_grid(step: float = 0.1) -> list[tuple[float, float, float]]:
    """Generate discrete (alpha, beta, gamma) grid on the simplex: alpha + beta + gamma = 1.0.

    Always includes pure relevance (1.0, 0.0, 0.0) as the initial baseline.
    """
    n_steps = int(round(1.0 / step))
    grid: list[tuple[float, float, float]] = []

    # Pure relevance baseline first
    grid.append((1.0, 0.0, 0.0))

    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            a = round(i * step, 4)
            b = round(j * step, 4)
            g = round(k * step, 4)
            tup = (a, b, g)
            if tup not in grid:
                grid.append(tup)

    return grid


# ---------------------------------------------------------------------------
# Metric Evaluator
# ---------------------------------------------------------------------------

def compute_mrr_at_k(
    ranked_passage_ids: list[str],
    gold_passage_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Reciprocal Rank at k for a single query."""
    if not gold_passage_ids:
        return 0.0
    for rank, pid in enumerate(ranked_passage_ids[:k], start=1):
        if pid in gold_passage_ids:
            return 1.0 / float(rank)
    return 0.0


def compute_recall_at_k(
    ranked_passage_ids: list[str],
    gold_passage_ids: set[str],
    k: int = 5,
) -> float:
    """Compute hit recall at k (1.0 if any gold item is in top-k, else 0.0)."""
    if not gold_passage_ids:
        return 0.0
    top_set = set(ranked_passage_ids[:k])
    return 1.0 if bool(top_set & gold_passage_ids) else 0.0


# ---------------------------------------------------------------------------
# Grid Search Result
# ---------------------------------------------------------------------------

@dataclass
class GridSearchResult:
    """Outcome of validation grid search.

    Attributes:
        best_alpha: Optimal weight for relevance.
        best_beta: Optimal weight for credibility.
        best_gamma: Optimal weight for temporal alignment.
        best_tau: Optimal decay parameter in days.
        best_metric_score: Highest validation metric achieved.
        metric_name: Name of evaluated metric (e.g. 'mrr@10').
        baseline_pure_relevance_score: Metric achieved with pure relevance (1.0, 0.0, 0.0).
        num_validation_queries: Total number of validation queries evaluated.
        all_trials: Records of all hyperparameter combinations evaluated.
        split_name: Name of the validation split (must NOT be 'test').
    """

    best_alpha: float
    best_beta: float
    best_gamma: float
    best_tau: float
    best_metric_score: float
    metric_name: str
    baseline_pure_relevance_score: float
    num_validation_queries: int
    all_trials: list[dict[str, Any]] = field(default_factory=list)
    split_name: str = "validation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_weights": {
                "alpha": self.best_alpha,
                "beta": self.best_beta,
                "gamma": self.best_gamma,
            },
            "best_tau": self.best_tau,
            "best_metric_score": round(self.best_metric_score, 6),
            "baseline_pure_relevance_score": round(self.baseline_pure_relevance_score, 6),
            "metric_name": self.metric_name,
            "num_validation_queries": self.num_validation_queries,
            "split_name": self.split_name,
            "total_trials": len(self.all_trials),
        }


# ---------------------------------------------------------------------------
# Grid Search Engine
# ---------------------------------------------------------------------------

class ScoringGridSearch:
    """Executes hyperparameter grid search for score combination weights exclusively on validation data."""

    def __init__(
        self,
        weight_step: float = 0.1,
        tau_candidates: Sequence[float] | None = None,
        metric: str = "mrr@10",
        k: int = 10,
        credibility_config: CredibilityConfig | None = None,
    ) -> None:
        self.weight_grid = generate_simplex_grid(step=weight_step)
        self.tau_candidates = list(tau_candidates or [30.0, 90.0, 180.0, 365.0])
        self.metric = metric.lower()
        self.k = int(k)
        self.credibility_config = credibility_config or CredibilityConfig.default_provisional()

    def run(
        self,
        validation_data: list[dict[str, Any]],
        split_name: str = "validation",
    ) -> GridSearchResult:
        """Run grid search over validation queries.

        Args:
            validation_data: List of validation query items. Each must contain:
              - 'claim': str
              - 'gold_passage_ids': list of gold IDs
              - 'candidates': list of candidate passages/results
              - optional 'claim_date': str
              - optional 'claim_type': str
            split_name: Identifier of the split. MUST be a validation alias ('val',
                        'validation', 'dev', 'tune'). Will raise ValueError if 'test'
                        is supplied.

        Returns:
            GridSearchResult containing optimal weights, baseline comparison, and trial history.
        """
        # Strict validation-only guard
        clean_split = split_name.strip().lower()
        if clean_split not in VALID_VALIDATION_SPLIT_NAMES:
            if "test" in clean_split:
                raise ValueError(
                    f"TEST-SET TUNING FORBIDDEN: split_name='{split_name}' violates methodological guidelines. "
                    f"Hyperparameter grid search is strictly permitted only on validation data: {VALID_VALIDATION_SPLIT_NAMES}."
                )
            raise ValueError(
                f"Invalid split_name '{split_name}'. Must be one of validation splits: {VALID_VALIDATION_SPLIT_NAMES}"
            )

        if not validation_data:
            raise ValueError("validation_data is empty; cannot run grid search.")

        logger.info(
            "Starting grid search over %d weight combinations and %d tau values on %d validation queries...",
            len(self.weight_grid),
            len(self.tau_candidates),
            len(validation_data),
        )

        all_trials: list[dict[str, Any]] = []
        best_score = -1.0
        best_alpha = 1.0
        best_beta = 0.0
        best_gamma = 0.0
        best_tau = 180.0
        baseline_pure_relevance = -1.0

        for tau in self.tau_candidates:
            # Configure scorer for this tau
            temp_cfg = TemporalConfig(default_tau=tau)
            scorer_cfg = CombinedScorerConfig(
                credibility=self.credibility_config,
                temporal=temp_cfg,
            )
            scorer = CombinedScorer(config=scorer_cfg)

            for alpha, beta, gamma in self.weight_grid:
                weights = ScoreWeights(alpha=alpha, beta=beta, gamma=gamma, normalize_weights=True)

                # Evaluate over all validation queries
                query_scores: list[float] = []

                for item in validation_data:
                    claim_date = item.get("claim_date")
                    claim_type = item.get("claim_type")
                    gold_ids = set(item.get("gold_passage_ids", []))
                    candidates = item.get("candidates", [])

                    ranked_candidates = scorer.score_candidates(
                        candidates=candidates,
                        claim_date=claim_date,
                        claim_type=claim_type,
                        weights_override=weights,
                        top_n=self.k,
                    )
                    ranked_ids = [c.passage_id for c in ranked_candidates]

                    if self.metric.startswith("recall"):
                        q_score = compute_recall_at_k(ranked_ids, gold_ids, k=self.k)
                    else:  # default mrr
                        q_score = compute_mrr_at_k(ranked_ids, gold_ids, k=self.k)

                    query_scores.append(q_score)

                mean_score = float(np.mean(query_scores)) if query_scores else 0.0

                trial_record = {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "tau": tau,
                    "mean_score": round(mean_score, 6),
                }
                all_trials.append(trial_record)

                # Record baseline: pure relevance (1.0, 0.0, 0.0)
                if (alpha == 1.0 and beta == 0.0 and gamma == 0.0) and baseline_pure_relevance < 0:
                    baseline_pure_relevance = mean_score

                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha, best_beta, best_gamma = alpha, beta, gamma
                    best_tau = tau

        logger.info(
            "Grid search complete. Best score: %.4f (alpha=%.2f, beta=%.2f, gamma=%.2f, tau=%.1f) vs baseline pure relevance: %.4f",
            best_score, best_alpha, best_beta, best_gamma, best_tau, baseline_pure_relevance
        )

        return GridSearchResult(
            best_alpha=best_alpha,
            best_beta=best_beta,
            best_gamma=best_gamma,
            best_tau=best_tau,
            best_metric_score=best_score,
            metric_name=self.metric,
            baseline_pure_relevance_score=max(0.0, baseline_pure_relevance),
            num_validation_queries=len(validation_data),
            all_trials=all_trials,
            split_name=split_name,
        )
