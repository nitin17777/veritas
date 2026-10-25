"""Usage example demonstrating the VERITAS P4 Relevance Reranker.

This script demonstrates:
1. Configuring a CrossEncoderReranker with custom parameters.
2. Scoring retrieved candidate passages against a factual claim.
3. Inspecting normalized relevance scores in [0.0, 1.0].
4. Accessing run configuration metadata (model name, version).

IMPORTANT DISCLAIMER:
Normalized scores reflect relative ranking signals on [0.0, 1.0].
They are NOT calibrated probabilities and must not be interpreted as the
likelihood that a claim or passage is factually true.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src/ is on sys.path when run directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from veritas.retrieval.reranker import (
    CrossEncoderReranker,
    MockCrossEncoder,
    RerankerConfig,
    RerankResult,
)
from veritas.retrieval.schema import Passage


def main() -> None:
    # 1. Define a claim and retrieved candidate passages
    claim = "mRNA vaccines significantly reduce COVID-19 hospitalizations and deaths."

    candidates = [
        Passage(
            passage_id="doc_101",
            text="Clinical trials demonstrate that mRNA COVID-19 vaccines offer over 90% "
                 "protection against severe disease and death from SARS-CoV-2.",
        ),
        Passage(
            passage_id="doc_102",
            text="The French Revolution began in 1789 and led to the overthrow of the monarchy.",
        ),
        Passage(
            passage_id="doc_103",
            text="Real-world observational studies confirm that hospitalization rates dropped "
                 "dramatically following the rollout of Pfizer-BioNTech and Moderna vaccines.",
        ),
        Passage(
            passage_id="doc_104",
            text="COVID-19 symptoms can include fever, cough, fatigue, and loss of taste or smell.",
        ),
    ]

    # 2. Configure the reranker
    # (In production, omits 'model' to load sentence-transformers CrossEncoder.
    # Here, we supply MockCrossEncoder with deterministic scores for offline demonstration.)
    config = RerankerConfig(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_version="1.0.0",
        top_n=3,
        normalization_method="sigmoid",
        tie_breaker="original",
    )

    # Mock model assigning realistic logits: high for docs 101/103, low for 102
    mock_model = MockCrossEncoder(
        score_map={
            (claim, candidates[0].text): 4.25,   # Highly relevant logit
            (claim, candidates[1].text): -5.10,  # Irrelevant logit
            (claim, candidates[2].text): 3.90,   # Highly relevant logit
            (claim, candidates[3].text): 0.80,   # Mildly relevant logit
        }
    )

    reranker = CrossEncoderReranker(config=config, model=mock_model)

    # 3. Log run configuration metadata
    metadata = reranker.get_metadata()
    print("=== Reranker Run Configuration ===")
    print(json.dumps(metadata, indent=2))
    print()

    # 4. Perform reranking
    print(f"Claim: '{claim}'\n")
    print(f"Reranking {len(candidates)} candidates (top_n={config.top_n})...\n")

    results: list[RerankResult] = reranker.rerank(claim, candidates)

    print("=== Reranked Results ===")
    for res in results:
        snippet = res.passage.text[:70] if res.passage else ""
        print(f"Rank {res.rank} | ID: {res.passage_id:<8} | "
              f"Score: {res.score:.4f} (raw logit: {res.raw_score:+.2f}) | "
              f"Text: '{snippet}...'")

    print("\nNote: Normalized scores represent relative ranking signals, NOT calibrated probabilities.")


if __name__ == "__main__":
    main()
