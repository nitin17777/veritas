"""Credibility scoring module for VERITAS (P4).

Assigns domain-level credibility scores to candidate evidence passages using
a configurable tier table with an explicit fallback for unknown domains,
plus a pseudo-random mode for empirical ablation studies.

IMPORTANT DISCLAIMER ON OBJECTIVITY & PROVISIONAL STATUS:
---------------------------------------------------------
The domain credibility scores and tier assignments provided in this module
are PROVISIONAL heuristic approximations intended for research and system prototyping.

They do NOT represent objective ground truth, authoritative judgments of journalistic
integrity, or absolute arbiters of factuality. Domain-level heuristics carry
known methodological limitations:
  1. Institutional Bias: Ratings may privilege established Western media organizations
     over independent, local, or specialized investigative journalism.
  2. Granularity: Entire domains are scored uniformly, ignoring article-level variation
     (e.g., opinion columns vs. investigative reporting on the same site).
  3. Temporal Lag: Domain ownership, editorial guidelines, and accuracy fluctuate over time.

All scores and tier assignments are subject to ongoing empirical validation.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from veritas.retrieval.schema import Passage

logger = logging.getLogger("veritas.scoring.credibility")

# Standard default provisional score assignments
DEFAULT_UNKNOWN_DOMAIN_SCORE = 0.50
DEFAULT_RANDOM_SEED = 42

DEFAULT_PROVISIONAL_TIERS: dict[str, dict[str, Any]] = {
    "tier_1": {
        "score": 0.90,
        "domains": [
            "apnews.com",
            "reuters.com",
            "afp.com",
            "snopes.com",
            "politifact.com",
            "factcheck.org",
            "who.int",
            "cdc.gov",
            "nih.gov",
            "nature.com",
            "science.org",
        ],
    },
    "tier_2": {
        "score": 0.75,
        "domains": [
            "bbc.com",
            "bbc.co.uk",
            "nytimes.com",
            "washingtonpost.com",
            "theguardian.com",
            "wsj.com",
            "economist.com",
            "npr.org",
            "pbs.org",
            "bloomberg.com",
            "lemonde.fr",
            "dw.com",
        ],
    },
    "tier_3": {
        "score": 0.50,
        "domains": [
            "usatoday.com",
            "time.com",
            "newsweek.com",
            "huffpost.com",
            "vox.com",
            "theatlantic.com",
            "politico.com",
            "thehill.com",
        ],
    },
    "tier_4": {
        "score": 0.25,
        "domains": [
            "dailymail.co.uk",
            "nypost.com",
            "thesun.co.uk",
            "breitbart.com",
            "dailycaller.com",
            "infowars.com",
            "thegatewaypundit.com",
            "rt.com",
            "sputniknews.com",
        ],
    },
}


# ---------------------------------------------------------------------------
# Domain Extraction Helper
# ---------------------------------------------------------------------------

def extract_domain(url_or_domain: str | None) -> str:
    """Extract and normalize a fully-qualified domain name from a URL or raw domain string.

    Normalization steps:
      - Strips leading/trailing whitespace.
      - Converts to lowercase.
      - Strips protocol scheme (http://, https://).
      - Strips 'www.' prefix.
      - Strips port numbers, URL paths, query parameters, and fragments.

    Args:
        url_or_domain: String representing a URL (e.g. 'https://www.reuters.com/news/123')
                       or raw domain (e.g. 'bbc.co.uk').

    Returns:
        Normalized domain string, or empty string if input is empty or invalid.
    """
    if not url_or_domain or not isinstance(url_or_domain, str):
        return ""

    raw = url_or_domain.strip().lower()
    if not raw:
        return ""

    # Prepend scheme if missing so urlparse handles netloc properly
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", raw):
        raw = "http://" + raw

    try:
        parsed = urlparse(raw)
        host = parsed.netloc or parsed.path
    except Exception:
        host = raw

    # Strip port number if present (e.g. 'example.com:8080')
    host = host.split(":")[0]

    # Strip 'www.' prefix
    if host.startswith("www."):
        host = host[4:]

    # Remove any trailing slashes or path segments
    host = host.split("/")[0].strip()

    return host


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CredibilityConfig:
    """Configuration for domain credibility scoring.

    Attributes:
        unknown_domain_score: Default fallback score for domains not in the tier table.
                              Must be within [0.0, 1.0]. Marked PROVISIONAL.
        random_mode: When True, enables random credibility scoring for ablation experiments.
        random_seed: Fixed seed for reproducible random credibility scoring.
        tier_table: Mapping from domain string to credibility score in [0.0, 1.0].
        is_provisional: Flag documenting that tier ratings are provisional approximations.
    """

    unknown_domain_score: float = DEFAULT_UNKNOWN_DOMAIN_SCORE
    random_mode: bool = False
    random_seed: int = DEFAULT_RANDOM_SEED
    tier_table: dict[str, float] = field(default_factory=dict)
    is_provisional: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.unknown_domain_score <= 1.0):
            raise ValueError(
                f"unknown_domain_score must be in [0.0, 1.0], got {self.unknown_domain_score}"
            )
        for domain, score in self.tier_table.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"Tier score for domain '{domain}' must be in [0.0, 1.0], got {score}"
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> CredibilityConfig:
        """Load CredibilityConfig from a YAML tier table file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Credibility tier file not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        cred_dict = raw.get("credibility", raw)
        unknown_score = float(cred_dict.get("unknown_domain_score", DEFAULT_UNKNOWN_DOMAIN_SCORE))
        random_mode = bool(cred_dict.get("random_mode", False))
        random_seed = int(cred_dict.get("random_seed", DEFAULT_RANDOM_SEED))

        tier_table: dict[str, float] = {}
        tiers = cred_dict.get("tiers", {})
        for tier_name, tier_info in tiers.items():
            if not isinstance(tier_info, dict):
                continue
            score = float(tier_info.get("provisional_score", tier_info.get("score", 0.5)))
            domains = tier_info.get("domains", [])
            for d in domains:
                norm_d = extract_domain(str(d))
                if norm_d:
                    tier_table[norm_d] = score

        return cls(
            unknown_domain_score=unknown_score,
            random_mode=random_mode,
            random_seed=random_seed,
            tier_table=tier_table,
            is_provisional=True,
        )

    @classmethod
    def default_provisional(cls) -> CredibilityConfig:
        """Construct the default built-in provisional configuration."""
        table: dict[str, float] = {}
        for tier_info in DEFAULT_PROVISIONAL_TIERS.values():
            score = float(tier_info["score"])
            for domain in tier_info["domains"]:
                table[domain] = score
        return cls(
            unknown_domain_score=DEFAULT_UNKNOWN_DOMAIN_SCORE,
            random_mode=False,
            random_seed=DEFAULT_RANDOM_SEED,
            tier_table=table,
            is_provisional=True,
        )


# ---------------------------------------------------------------------------
# CredibilityScorer Class
# ---------------------------------------------------------------------------

class CredibilityScorer:
    """Evaluates the credibility of evidence sources based on domain tiers.

    Guarantees:
      - All output scores are strictly in [0.0, 1.0].
      - Unrecognized domains receive the explicit configured unknown_domain_score.
      - In random_mode, scores are generated deterministically from (seed, domain)
        for ablation experimentation.
    """

    def __init__(self, config: CredibilityConfig | dict[str, Any] | None = None) -> None:
        if config is None:
            self.config = CredibilityConfig.default_provisional()
        elif isinstance(config, dict):
            # If path provided inside dict, load from yaml
            tier_path = config.get("tier_config_path")
            if tier_path:
                self.config = CredibilityConfig.from_yaml(tier_path)
                if "unknown_domain_score" in config:
                    self.config.unknown_domain_score = float(config["unknown_domain_score"])
                if "random_mode" in config:
                    self.config.random_mode = bool(config["random_mode"])
                if "random_seed" in config:
                    self.config.random_seed = int(config["random_seed"])
            else:
                self.config = CredibilityConfig(**config)
        elif isinstance(config, CredibilityConfig):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config).__name__}")

    def _get_random_score(self, domain: str) -> float:
        """Generate a deterministic pseudo-random credibility score in [0.0, 1.0].

        Uses SHA-256 over (seed, domain) so that results are order-independent,
        reproducible, and require no mutable random state.
        """
        token = f"{self.config.random_seed}::{domain}".encode("utf-8")
        h = hashlib.sha256(token).digest()
        # Map 4 bytes to float in [0.0, 1.0]
        int_val = int.from_bytes(h[:4], "little")
        return round(float(int_val) / 0xFFFFFFFF, 6)

    def score_domain(self, domain: str | None) -> float:
        """Assign a credibility score in [0.0, 1.0] for a given domain name.

        Matching logic:
          1. If random_mode is enabled, returns deterministic pseudo-random score.
          2. Exact match in tier_table.
          3. Suffix match (e.g. 'news.bbc.co.uk' matches 'bbc.co.uk').
          4. Fallback to configured unknown_domain_score.

        Args:
            domain: Domain or URL string.

        Returns:
            Float credibility score strictly in [0.0, 1.0].
        """
        norm_domain = extract_domain(domain)
        if not norm_domain:
            logger.debug("Empty or unextractable domain '%s'; returning unknown_domain_score %.2f",
                         domain, self.config.unknown_domain_score)
            return self.config.unknown_domain_score

        # Ablation mode
        if self.config.random_mode:
            score = self._get_random_score(norm_domain)
            logger.debug("Random ablation score for '%s' (seed=%d): %.4f",
                         norm_domain, self.config.random_seed, score)
            return score

        # 1. Exact match
        if norm_domain in self.config.tier_table:
            return self.config.tier_table[norm_domain]

        # 2. Suffix / parent domain match (e.g., 'subdomain.apnews.com' -> 'apnews.com')
        for known_domain, score in self.config.tier_table.items():
            if norm_domain.endswith("." + known_domain):
                return score

        # 3. Fallback for unrecognized domains
        logger.debug("Domain '%s' not found in tier table; using fallback score %.2f",
                     norm_domain, self.config.unknown_domain_score)
        return self.config.unknown_domain_score

    def score_passage(self, passage: Passage | Any) -> float:
        """Score the credibility of a candidate Passage or dictionary.

        Inspects passage.domain, passage.url, or metadata in order of availability.

        Args:
            passage: A Passage object, SearchResult, or dictionary.

        Returns:
            Float credibility score in [0.0, 1.0].
        """
        domain = None
        if hasattr(passage, "domain") and passage.domain:
            domain = passage.domain
        elif hasattr(passage, "url") and passage.url:
            domain = passage.url
        elif hasattr(passage, "passage") and passage.passage:
            p = passage.passage
            domain = getattr(p, "domain", None) or getattr(p, "url", None)
        elif isinstance(passage, dict):
            domain = (
                passage.get("domain")
                or passage.get("url")
                or passage.get("metadata", {}).get("domain")
            )

        return self.score_domain(domain)
