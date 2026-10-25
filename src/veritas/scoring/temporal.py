"""Temporal scoring module for VERITAS (P4).

Computes recency and temporal alignment scores between a claim date and an
evidence publication date using an exponential decay model:

    temporal_score = exp(-delta_days / tau)

where:
    delta_days = |claim_date - evidence_date| (in days)
    tau        = decay parameter (in days)

Missing dates fall back to a neutral score (default: 0.50) and log the event.
Different tau parameters are supported based on claim type (e.g., breaking news,
health, scientific).
"""

from __future__ import annotations

import datetime
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from veritas.retrieval.schema import Passage

logger = logging.getLogger("veritas.scoring.temporal")

DEFAULT_TAU = 180.0          # ~6 months
DEFAULT_NEUTRAL_SCORE = 0.50  # Neutral prior when dates are unavailable

# Provisional claim-type specific decay parameters (in days)
DEFAULT_TAU_BY_CLAIM_TYPE: dict[str, float] = {
    "breaking": 14.0,       # Fast-moving breaking events (rapid obsolescence)
    "election": 30.0,       # Political campaigns and shifting poll numbers
    "politics": 90.0,       # Policy statements and legislative updates
    "health": 365.0,        # Public health recommendations (moderate stability)
    "science": 730.0,       # Scientific consensus & empirical studies (slow decay)
    "historical": 1825.0,   # Historical events (very long half-life)
}


# ---------------------------------------------------------------------------
# Date Parsing Helper
# ---------------------------------------------------------------------------

def parse_iso_date(date_val: str | datetime.date | datetime.datetime | None) -> datetime.date | None:
    """Parse a flexible date value into a standard datetime.date object.

    Handles:
      - datetime.date and datetime.datetime instances
      - ISO-8601 strings: YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS, YYYY-MM-DDTHH:MM:SSZ
      - Year-only strings: YYYY (interprets as YYYY-01-01)
      - Strings with trailing whitespace or quotes

    Args:
        date_val: Input date string or object.

    Returns:
        datetime.date instance, or None if input cannot be parsed.
    """
    if date_val is None:
        return None

    if isinstance(date_val, datetime.datetime):
        return date_val.date()
    if isinstance(date_val, datetime.date):
        return date_val

    if not isinstance(date_val, str):
        return None

    s = date_val.strip().strip("'\"")
    if not s:
        return None

    # Try full ISO datetime or date
    # Format: YYYY-MM-DD...
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if match:
        try:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return datetime.date(year, month, day)
        except ValueError:
            return None

    # Year-only: YYYY
    if re.match(r"^\d{4}$", s):
        try:
            return datetime.date(int(s), 1, 1)
        except ValueError:
            return None

    return None


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class TemporalConfig:
    """Configuration for temporal exponential decay scoring.

    Attributes:
        default_tau: Default decay rate in days. Must be strictly positive.
        neutral_score: Fallback score assigned when either date is missing.
                       Must be in [0.0, 1.0].
        tau_by_claim_type: Mapping from claim type strings to custom tau values.
    """

    default_tau: float = DEFAULT_TAU
    neutral_score: float = DEFAULT_NEUTRAL_SCORE
    tau_by_claim_type: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TAU_BY_CLAIM_TYPE))

    def __post_init__(self) -> None:
        if self.default_tau <= 0:
            raise ValueError(f"default_tau must be strictly positive, got {self.default_tau}")
        if not (0.0 <= self.neutral_score <= 1.0):
            raise ValueError(f"neutral_score must be in [0.0, 1.0], got {self.neutral_score}")
        for ctype, tau in self.tau_by_claim_type.items():
            if tau <= 0:
                raise ValueError(f"tau for claim_type '{ctype}' must be strictly positive, got {tau}")

    def get_tau(self, claim_type: str | None = None) -> float:
        """Resolve effective tau for a given claim type."""
        if not claim_type or not isinstance(claim_type, str):
            return float(self.default_tau)
        ctype_clean = claim_type.strip().lower()
        return float(self.tau_by_claim_type.get(ctype_clean, self.default_tau))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalConfig:
        """Construct TemporalConfig from dictionary."""
        default_tau = float(data.get("default_tau", DEFAULT_TAU))
        neutral_score = float(data.get("neutral_score", DEFAULT_NEUTRAL_SCORE))
        tau_map = dict(DEFAULT_TAU_BY_CLAIM_TYPE)
        if "tau_by_claim_type" in data and isinstance(data["tau_by_claim_type"], dict):
            for k, v in data["tau_by_claim_type"].items():
                tau_map[str(k).lower()] = float(v)
        return cls(
            default_tau=default_tau,
            neutral_score=neutral_score,
            tau_by_claim_type=tau_map,
        )


# ---------------------------------------------------------------------------
# TemporalScorer Class
# ---------------------------------------------------------------------------

class TemporalScorer:
    """Computes exponential decay temporal scores between claims and evidence.

    Formula:
        score = exp(-delta_days / tau)
        delta_days = |claim_date - evidence_date|

    Guarantees:
      - All output scores are strictly in [0.0, 1.0].
      - When delta_days = 0, score = 1.0.
      - As delta_days -> infinity, score -> 0.0.
      - Missing or unparseable dates return neutral_score (default 0.5) and log an event.
    """

    def __init__(self, config: TemporalConfig | dict[str, Any] | None = None) -> None:
        if config is None:
            self.config = TemporalConfig()
        elif isinstance(config, dict):
            self.config = TemporalConfig.from_dict(config)
        elif isinstance(config, TemporalConfig):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config).__name__}")

    def score(
        self,
        claim_date: str | datetime.date | None,
        evidence_date: str | datetime.date | None,
        claim_type: str | None = None,
        tau_override: float | None = None,
    ) -> float:
        """Calculate temporal alignment score between claim_date and evidence_date.

        Args:
            claim_date: Date associated with the claim statement.
            evidence_date: Publication date of the evidence passage.
            claim_type: Optional category of claim (e.g., 'breaking', 'science')
                        to select appropriate decay parameter tau.
            tau_override: Optional explicit tau override in days.

        Returns:
            Float temporal score strictly in [0.0, 1.0].
        """
        parsed_claim = parse_iso_date(claim_date)
        parsed_evidence = parse_iso_date(evidence_date)

        # Missing date handling
        if parsed_claim is None or parsed_evidence is None:
            missing_items = []
            if parsed_claim is None:
                missing_items.append(f"claim_date='{claim_date}'")
            if parsed_evidence is None:
                missing_items.append(f"evidence_date='{evidence_date}'")
            logger.debug(
                "Missing or unparseable date (%s); falling back to neutral_score=%.2f",
                ", ".join(missing_items),
                self.config.neutral_score,
            )
            return float(self.config.neutral_score)

        # Resolve tau
        if tau_override is not None:
            if tau_override <= 0:
                raise ValueError(f"tau_override must be strictly positive, got {tau_override}")
            effective_tau = float(tau_override)
        else:
            effective_tau = self.config.get_tau(claim_type)

        # Compute absolute day difference
        delta_days = abs((parsed_claim - parsed_evidence).days)

        # Exponential decay: exp(-delta_days / tau)
        exponent = -float(delta_days) / effective_tau
        raw_score = math.exp(exponent)

        # Clamp strictly to [0.0, 1.0]
        score = max(0.0, min(1.0, raw_score))
        return round(score, 6)

    def score_passage(
        self,
        claim_date: str | datetime.date | None,
        passage: Passage | Any,
        claim_type: str | None = None,
        tau_override: float | None = None,
    ) -> float:
        """Score temporal alignment of a candidate Passage or result against claim_date.

        Extracts evidence date from passage.date, passage.publication_date,
        or metadata.
        """
        evidence_date = None
        if hasattr(passage, "date") and passage.date:
            evidence_date = passage.date
        elif hasattr(passage, "publication_date") and passage.publication_date:
            evidence_date = passage.publication_date
        elif hasattr(passage, "passage") and passage.passage:
            p = passage.passage
            evidence_date = getattr(p, "date", None) or getattr(p, "publication_date", None)
        elif isinstance(passage, dict):
            evidence_date = (
                passage.get("date")
                or passage.get("publication_date")
                or passage.get("metadata", {}).get("date")
            )

        return self.score(
            claim_date=claim_date,
            evidence_date=evidence_date,
            claim_type=claim_type,
            tau_override=tau_override,
        )
