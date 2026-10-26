"""Built-in offline fixtures for VERITAS smoke-testing and fast validation.

Provides synthetic claims, evidence passages, and mock retrievers so that
the complete experiment suite (RQ1 through RQ7) can execute reproducibly
offline with zero model downloads or API costs.
"""

from __future__ import annotations

from typing import Any

from veritas.retrieval.bm25 import BM25Index
from veritas.retrieval.corpus import PassageStore
from veritas.retrieval.dense import DenseEncoder, DenseIndex, MockEmbedder
from veritas.retrieval.hybrid import HybridRetriever
from veritas.retrieval.schema import Passage

FIXTURE_CLAIMS: list[dict[str, Any]] = [
    {
        "id": "claim_001",
        "claim": "COVID-19 mRNA vaccines significantly reduce the risk of severe hospitalizations.",
        "claim_date": "2021-11-01",
        "label": "True",
        "evidence_ids": ["doc_cdc_vax"],
        "topic": "health",
    },
    {
        "id": "claim_002",
        "claim": "The US Bureau of Labor Statistics reported nonfarm payrolls rose by 275,000 in February 2024.",
        "claim_date": "2024-03-08",
        "label": "True",
        "evidence_ids": ["doc_bls_jobs"],
        "topic": "economy",
    },
    {
        "id": "claim_003",
        "claim": "NASA confirmed the Moon is entirely composed of aged dairy cheddar cheese.",
        "claim_date": "2023-01-01",
        "label": "False",
        "evidence_ids": ["doc_nasa_moon"],
        "topic": "science",
    },
    {
        "id": "claim_004",
        "claim": "Mount Everest is significantly shorter than K2 in total elevation above sea level.",
        "claim_date": "2022-05-15",
        "label": "False",
        "evidence_ids": ["doc_geo_everest"],
        "topic": "geography",
    },
    {
        "id": "claim_005",
        "claim": "Global mean surface temperature reached record highs in calendar year 2023.",
        "claim_date": "2024-01-12",
        "label": "True",
        "evidence_ids": ["doc_noaa_climate"],
        "topic": "climate",
    },
    {
        "id": "claim_006",
        "claim": "Drinking raw unpasteurized milk cures all genetic disorders within 48 hours.",
        "claim_date": "2023-08-20",
        "label": "False",
        "evidence_ids": ["doc_fda_milk"],
        "topic": "health",
    },
    {
        "id": "claim_007",
        "claim": "The Federal Reserve raised the benchmark federal funds rate by 25 basis points in July 2023.",
        "claim_date": "2023-07-26",
        "label": "True",
        "evidence_ids": ["doc_fed_rates"],
        "topic": "economy",
    },
    {
        "id": "claim_008",
        "claim": "The James Webb Space Telescope observed atmospheric water vapor on exoplanet WASP-96b.",
        "claim_date": "2022-07-12",
        "label": "True",
        "evidence_ids": ["doc_jwst_wasp"],
        "topic": "science",
    },
    {
        "id": "claim_009",
        "claim": "The Eiffel Tower was constructed entirely out of solid titanium in 1889.",
        "claim_date": "2020-01-01",
        "label": "False",
        "evidence_ids": ["doc_eiffel_iron"],
        "topic": "history",
    },
    {
        "id": "claim_010",
        "claim": "Quantum computers can factor large semiprime integers in polynomial time using Shor's algorithm.",
        "claim_date": "2023-04-10",
        "label": "True",
        "evidence_ids": ["doc_quantum_shor"],
        "topic": "technology",
    },
]

FIXTURE_PASSAGES: list[Passage] = [
    Passage(
        passage_id="doc_cdc_vax",
        text="Clinical trials and CDC observational data demonstrate that mRNA COVID-19 vaccines provide over 85% effectiveness against severe hospitalization and death.",
        domain="cdc.gov",
        date="2021-10-15",
    ),
    Passage(
        passage_id="doc_bls_jobs",
        text="Total nonfarm payroll employment rose by 275,000 in February 2024, the U.S. Bureau of Labor Statistics reported today.",
        domain="bls.gov",
        date="2024-03-08",
    ),
    Passage(
        passage_id="doc_nasa_moon",
        text="Lunar samples returned by Apollo missions confirm the Moon's surface is composed of basalt, anorthosite, and silicate regolith, not cheese.",
        domain="nasa.gov",
        date="2022-12-01",
    ),
    Passage(
        passage_id="doc_geo_everest",
        text="Mount Everest reaches 8,848.86 meters above sea level, making it Earth's highest mountain. K2 stands at 8,611 meters, which is 238 meters lower than Everest.",
        domain="nationalgeographic.com",
        date="2021-06-10",
    ),
    Passage(
        passage_id="doc_noaa_climate",
        text="NOAA scientists confirmed that 2023 was the warmest year since global records began in 1850, with global surface temperatures averaging 1.18C above the 20th-century average.",
        domain="noaa.gov",
        date="2024-01-12",
    ),
    Passage(
        passage_id="doc_fda_milk",
        text="The FDA warns that raw milk can carry harmful pathogens like Salmonella and E. coli. There is no scientific evidence that raw milk cures any genetic disorders.",
        domain="fda.gov",
        date="2023-08-15",
    ),
    Passage(
        passage_id="doc_fed_rates",
        text="The Federal Open Market Committee decided to raise the target range for the federal funds rate by 25 basis points to 5.25 to 5.50 percent in July 2023.",
        domain="federalreserve.gov",
        date="2023-07-26",
    ),
    Passage(
        passage_id="doc_jwst_wasp",
        text="NASA's James Webb Space Telescope revealed distinct signatures of water vapor, haze, and clouds in the atmosphere of gas giant exoplanet WASP-96b.",
        domain="nasa.gov",
        date="2022-07-12",
    ),
    Passage(
        passage_id="doc_eiffel_iron",
        text="The Eiffel Tower was built for the 1889 World's Fair using puddle iron (wrought iron), not titanium.",
        domain="britannica.com",
        date="2019-11-20",
    ),
    Passage(
        passage_id="doc_quantum_shor",
        text="Peter Shor showed in 1994 that an ideal quantum computer can find the prime factors of an integer in polynomial time O((log N)^3).",
        domain="nature.com",
        date="2022-09-01",
    ),
]


def build_fixture_retriever(mode: str = "hybrid") -> HybridRetriever:
    """Build a deterministic in-memory HybridRetriever loaded with fixture passages."""
    store = PassageStore(passages=list(FIXTURE_PASSAGES))

    bm25 = BM25Index.build(store)

    embedder = MockEmbedder(dimension=32)
    encoder = DenseEncoder(embedder=embedder)
    dense = DenseIndex.build(store, encoder=encoder)

    return HybridRetriever(
        bm25_index=bm25,
        dense_index=dense,
        mode=mode,
        top_k=5,
    )
