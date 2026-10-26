"""Usage example demonstrating VERITAS P5 Claim Extraction and Query Generation.

This script demonstrates:
1. Taking an article / long text input.
2. Extracting check-worthy factual claims while filtering out opinions.
3. Inspecting the generated 2 to 3 retrieval search queries per claim.
4. Demonstrating seamless handoff to the VERITAS verification pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src/ is on sys.path when run directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from veritas.extraction.extract import ClaimExtractor, ExtractionConfig, extract_claims
from veritas.verification.llm_client import MockLLMClient


def main() -> None:
    print("=" * 70)
    print("VERITAS P5: Long Text Claim Extraction & Query Generation")
    print("=" * 70)

    # 1. Sample input text (article discussing health policy and economic data)
    sample_article = (
        "In a press conference on Tuesday, Senator Jane Doe declared that the state's "
        "new clean energy bill was the most exciting legislation in decades. She claimed "
        "that carbon emissions in the state fell by 18 percent between 2020 and 2023. "
        "Doe also stated that the state's solar initiative created exactly 14,200 new green "
        "jobs since its launch in March 2021. In my personal opinion, renewable energy is the "
        "only truly noble path forward for modern society. Later in her speech, the Senator "
        "noted that electric vehicle sales surpassed 50,000 units statewide last year."
    )

    print("\n[1] Input Article:")
    print("-" * 50)
    print(sample_article)
    print("-" * 50)

    # 2. Prepare realistic mock model output for deterministic offline execution
    # In live environments, specify backend='api' with your OPENAI_API_KEY.
    mock_llm_response = {
        "claims": [
            {
                "claim": "State carbon emissions fell by 18 percent between 2020 and 2023.",
                "queries": [
                    "state carbon emissions decline 2020 2023 18 percent",
                    "Department of Environmental Protection greenhouse gas report 2023",
                    "annual carbon reduction emissions data 2020 to 2023",
                ],
                "context": "She claimed that carbon emissions in the state fell by 18 percent between 2020 and 2023.",
                "reason": "Empirical statistical claim with specific baseline years and percentage change.",
            },
            {
                "claim": "The state solar initiative created 14,200 new green jobs since its launch in March 2021.",
                "queries": [
                    "state solar initiative green jobs March 2021 count",
                    "14200 solar energy jobs report employment data",
                ],
                "context": "Doe also stated that the state's solar initiative created exactly 14,200 new green jobs since its launch in March 2021.",
                "reason": "Verifiable job creation statistic tied to a specific policy program and start date.",
            },
            {
                "claim": "Electric vehicle sales surpassed 50,000 units statewide last year.",
                "queries": [
                    "statewide electric vehicle sales 50000 units data",
                    "annual EV registration statistics Department of Motor Vehicles",
                ],
                "context": "the Senator noted that electric vehicle sales surpassed 50,000 units statewide last year.",
                "reason": "Factual claim regarding motor vehicle sales figures and annual threshold.",
            },
        ]
    }

    mock_client = MockLLMClient(canned_responses=[json.dumps(mock_llm_response)])

    # 3. Configure the extractor
    config = ExtractionConfig(
        model_name="mock-gpt4",
        backend="mock",
        prompt_version="v1.0",
        temperature=0.0,
        max_claims=5,
        min_queries=2,
        max_queries=3,
    )

    extractor = ClaimExtractor(config=config, llm_client=mock_client)

    # 4. Execute extraction
    print("\n[2] Running Claim Extraction...")
    result = extractor.extract(sample_article)

    print(f"\nExtraction completed in {result.attempts} attempt(s).")
    print(f"Total Check-Worthy Claims Extracted: {result.total_claims}")
    print(f"Model: {result.model} (Prompt: {result.prompt_version})")

    # 5. Inspect extracted claims and search queries
    print("\n[3] Extracted Check-Worthy Claims & Retrieval Queries:")
    for idx, claim_item in enumerate(result.claims, start=1):
        print(f"\nClaim #{idx}: '{claim_item.claim}'")
        print(f"  Check-worthiness: {claim_item.reason}")
        print(f"  Source context  : \"{claim_item.context}\"")
        print("  Search Queries  :")
        for q_idx, query in enumerate(claim_item.queries, start=1):
            print(f"    [{q_idx}] {query}")

    print("\n[4] Notice:")
    print("  - Subjective opinion ('renewable energy is the only noble path') was filtered out.")
    print("  - Pronoun 'She' was resolved to 'State carbon emissions' and numbers/dates were strictly preserved.")
    print("  - Each claim has 2 to 3 targeted search queries ready for BM25/Dense retrieval.")
    print("=" * 70)


if __name__ == "__main__":
    main()
