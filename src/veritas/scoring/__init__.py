"""VERITAS scoring package (P4).

Implements evidence scoring modules:
- CredibilityScorer: Domain tier and heuristic credibility evaluations.
- TemporalScorer: Exponential decay temporal alignment scoring.
- CombinedScorer: Unified score combination (combined = alpha*rel + beta*cred + gamma*temp).
- ScoringGridSearch: Validation-only hyperparameter tuning for combination weights.
"""

from veritas.scoring.combine import (
    CombinedScorer,
    CombinedScorerConfig,
    ScoreWeights,
    ScoredCandidate,
)
from veritas.scoring.credibility import (
    CredibilityConfig,
    CredibilityScorer,
    extract_domain,
)
from veritas.scoring.grid_search import (
    GridSearchResult,
    ScoringGridSearch,
    generate_simplex_grid,
)
from veritas.scoring.temporal import (
    TemporalConfig,
    TemporalScorer,
    parse_iso_date,
)

__all__ = [
    "CombinedScorer",
    "CombinedScorerConfig",
    "CredibilityConfig",
    "CredibilityScorer",
    "GridSearchResult",
    "ScoreWeights",
    "ScoredCandidate",
    "ScoringGridSearch",
    "TemporalConfig",
    "TemporalScorer",
    "extract_domain",
    "generate_simplex_grid",
    "parse_iso_date",
]
