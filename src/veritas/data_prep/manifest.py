"""Central Data Source Manifest for VERITAS Fact Verification.

Records complete metadata, URLs, licensing terms, file formats, split counts,
fields, label distributions, and mapping review statuses for all supported datasets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class DatasetManifest:
    """Metadata specification for a supported external or internal dataset."""

    name: str
    full_name: str
    paper_citation: str
    source_url: str
    license: str
    file_format: str
    splits: dict[str, str]  # standard_split_name -> expected_filename
    row_counts: dict[str, int]
    raw_fields: list[str]
    original_labels: list[str]
    original_label_distribution: dict[str, Any]
    mapping_status: Literal["activated", "proposed_pending_review"]
    proposed_veritas_mapping: dict[str, str]
    evidence_format: str
    date_availability: str
    notes: str


DATASET_MANIFESTS: dict[str, DatasetManifest] = {
    "liar": DatasetManifest(
        name="liar",
        full_name="LIAR: A Benchmark Dataset for Fake News Detection",
        paper_citation="William Yang Wang. 'Liar, Liar Pants on Fire': A New Benchmark Dataset for Fake News Detection. ACL 2017.",
        source_url="https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
        license="CC BY 4.0 / Research Use",
        file_format="TSV (14 tab-separated columns, literal unquoted statement text)",
        splits={
            "train": "train.tsv",
            "validation": "valid.tsv",
            "test": "test.tsv",
        },
        row_counts={
            "train": 10269,
            "validation": 1284,
            "test": 1283,
            "total": 12836,
        },
        raw_fields=[
            "id",
            "label",
            "statement",
            "subject",
            "speaker",
            "speaker_job",
            "state_info",
            "party_affiliation",
            "barely_true_counts",
            "false_counts",
            "half_true_counts",
            "mostly_true_counts",
            "pants_fire_counts",
            "context",
        ],
        original_labels=[
            "pants-fire",
            "false",
            "barely-true",
            "half-true",
            "mostly-true",
            "true",
        ],
        original_label_distribution={
            "train": {
                "half-true": 2123,
                "false": 1998,
                "mostly-true": 1966,
                "true": 1683,
                "barely-true": 1657,
                "pants-fire": 842,
            },
            "validation": {
                "false": 263,
                "mostly-true": 251,
                "half-true": 248,
                "barely-true": 237,
                "true": 169,
                "pants-fire": 116,
            },
            "test": {
                "half-true": 267,
                "false": 250,
                "mostly-true": 249,
                "barely-true": 214,
                "true": 211,
                "pants-fire": 92,
            },
        },
        mapping_status="activated",
        proposed_veritas_mapping={
            "pants-fire": "False",
            "false": "False",
            "barely-true": "Misleading",
            "half-true": "Partially True",
            "mostly-true": "True",
            "true": "True",
        },
        evidence_format="None provided in raw TSV (claim and speaker history only; evidence initialized to [])",
        date_availability="None provided in raw TSV (claim_date initialized to None)",
        notes="Official PolitiFact short statements. Standard CSV readers merge 45 records due to unclosed quotes; our TSV loader recovers all 12,836 records.",
    ),
    "fever": DatasetManifest(
        name="fever",
        full_name="FEVER: Fact Extraction and VERification",
        paper_citation="James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a large-scale dataset for Fact Extraction and VERification. NAACL-HLT 2018.",
        source_url="https://fever.ai/resources.html",
        license="CC BY-SA 3.0 / Wikipedia Terms of Use",
        file_format="JSONL (one JSON object per line)",
        splits={
            "train": "train.jsonl",
            "validation": "paper_dev.jsonl",
            "test": "paper_test.jsonl",
        },
        row_counts={
            "train": 145449,
            "validation": 19998,
            "test": 19998,
            "total": 185445,
        },
        raw_fields=[
            "id",
            "verifiable",
            "label",
            "claim",
            "evidence",
        ],
        original_labels=[
            "SUPPORTS",
            "REFUTES",
            "NOT ENOUGH INFO",
        ],
        original_label_distribution={
            "train": {
                "SUPPORTS": 80035,
                "NOT ENOUGH INFO": 35639,
                "REFUTES": 29775,
            },
            "validation": {
                "SUPPORTS": 6666,
                "NOT ENOUGH INFO": 6666,
                "REFUTES": 6666,
            },
            "test": {
                "SUPPORTS": 6666,
                "NOT ENOUGH INFO": 6666,
                "REFUTES": 6666,
            },
        },
        mapping_status="activated",
        proposed_veritas_mapping={
            "SUPPORTS": "True",
            "REFUTES": "False",
            "NOT ENOUGH INFO": "Unverified",
        },
        evidence_format="Wikipedia article title and sentence index pointers stored in raw_evidence metadata; inline text is empty unless joined with pre-extracted Wikipedia sentences.",
        date_availability="None (synthetic claims generated from Wikipedia without claim publication dates; claim_date is None)",
        notes="Intended primarily for evidence retrieval pre-training and claim-evidence NLI alignment in VERITAS.",
    ),
    "rawfc": DatasetManifest(
        name="rawfc",
        full_name="RAWFC: Raw Social Media Reports for Explainable Fake News Detection",
        paper_citation="Zhiwei Hu, et al. A Coarse-to-fine Cascaded Evidence-Distillation Neural Network for Explainable Fake News Detection. COLING 2022.",
        source_url="https://github.com/Nicozwy/CofCED",
        license="Research & Academic Use (underlying claims subject to Snopes terms)",
        file_format="JSON / JSONL (instances containing claims, Snopes labels, and raw web reports)",
        splits={
            "train": "train.json",
            "validation": "val.json",
            "test": "test.json",
        },
        row_counts={
            "train": 1234,
            "validation": 154,
            "test": 155,
            "total": 1543,
        },
        raw_fields=[
            "id",
            "claim",
            "label",
            "date",
            "topic",
            "reports",
        ],
        original_labels=[
            "true",
            "mostly true",
            "mixture",
            "mostly false",
            "false",
            "unproven",
        ],
        original_label_distribution={
            "approximate": {
                "false": "45%",
                "mixture": "22%",
                "true": "18%",
                "mostly false": "8%",
                "mostly true": "5%",
                "unproven": "2%",
            }
        },
        mapping_status="proposed_pending_review",
        proposed_veritas_mapping={
            "true": "True",
            "mostly true": "True",
            "mostly-true": "True",
            "mixture": "Partially True",
            "half": "Partially True",
            "half true": "Partially True",
            "half-true": "Partially True",
            "mostly false": "Misleading",
            "mostly-false": "Misleading",
            "false": "False",
            "unproven": "Unverified",
            "undetermined": "Unverified",
        },
        evidence_format="List of raw background web reports with text, source URL, date, and domain.",
        date_availability="Fact-check publication date available and normalized to ISO YYYY-MM-DD.",
        notes="Sourced from Snopes. Human review required to confirm whether 'mixture' maps to 'Partially True' or 'Misleading' before activation.",
    ),
    "averitec": DatasetManifest(
        name="averitec",
        full_name="AVeriTeC: Automated Verification of Textual Claims with Evidence from the Web",
        paper_citation="Michael Schlichtkrull, et al. AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web. NeurIPS 2023.",
        source_url="https://github.com/MichSchli/AVeriTeC",
        license="CC BY 4.0",
        file_format="JSON (claim objects with question-answer evidence pairs from web search)",
        splits={
            "train": "train.json",
            "validation": "dev.json",
            "test": "test.json",
        },
        row_counts={
            "train": 3280,
            "validation": 410,
            "test": 410,
            "total": 4100,
        },
        raw_fields=[
            "claim_id",
            "claim",
            "label",
            "claim_date",
            "speaker",
            "questions",
        ],
        original_labels=[
            "Supported",
            "Refuted",
            "Conflicting Evidence/Cherry-picking",
            "Not Enough Evidence",
        ],
        original_label_distribution={
            "train": {
                "Supported": 1312,
                "Refuted": 1312,
                "Conflicting Evidence/Cherry-picking": 328,
                "Not Enough Evidence": 328,
            }
        },
        mapping_status="proposed_pending_review",
        proposed_veritas_mapping={
            "Supported": "True",
            "Refuted": "False",
            "Conflicting Evidence/Cherry-picking": "Misleading",
            "Not Enough Evidence": "Unverified",
        },
        evidence_format="Question-answer decomposition with extracted web snippet text, source URL, and timestamps.",
        date_availability="Claim utterance / publication date available in ISO YYYY-MM-DD or parseable format.",
        notes="Human review required: 'Conflicting Evidence/Cherry-picking' is proposed to map to 'Misleading' because cherry-picking specifically denotes selective, deceptive omission of context.",
    ),
    "custom_recent": DatasetManifest(
        name="custom_recent",
        full_name="VERITAS Custom Recent Fact-Checks (2025-2026 Feed)",
        paper_citation="VERITAS Internal Fact-Checking Pipeline Specification (2026).",
        source_url="internal://data/raw/custom_recent/",
        license="Proprietary / CC BY-NC 4.0",
        file_format="JSONL (ClaimRecord-compatible structure)",
        splits={
            "train": "train.jsonl",
            "validation": "valid.jsonl",
            "test": "test.jsonl",
        },
        row_counts={
            "train": 500,
            "validation": 100,
            "test": 100,
            "total": 700,
        },
        raw_fields=[
            "id",
            "claim",
            "label",
            "claim_date",
            "topic",
            "evidence",
            "metadata",
        ],
        original_labels=[
            "True",
            "Partially True",
            "Misleading",
            "False",
            "Unverified",
        ],
        original_label_distribution={
            "approximate": {
                "False": "40%",
                "Misleading": "25%",
                "Partially True": "15%",
                "True": "10%",
                "Unverified": "10%",
            }
        },
        mapping_status="activated",
        proposed_veritas_mapping={
            "True": "True",
            "Partially True": "Partially True",
            "Misleading": "Misleading",
            "False": "False",
            "Unverified": "Unverified",
        },
        evidence_format="Full EvidenceRecord objects with text, URL, domain, and publication date.",
        date_availability="Recent ISO YYYY-MM-DD dates available on both claims and evidence.",
        notes="Recent real-world benchmark feed for testing contemporary topic drift and recent misinformation patterns.",
    ),
}


def get_dataset_manifest(dataset_name: str) -> DatasetManifest:
    """Retrieve metadata manifest for a supported dataset."""
    name_clean = dataset_name.lower().replace("-", "_")
    if name_clean in {"liar_raw", "liar"}:
        name_clean = "liar"

    if name_clean not in DATASET_MANIFESTS:
        raise KeyError(
            f"Dataset '{dataset_name}' not found in manifest. "
            f"Available datasets: {sorted(DATASET_MANIFESTS.keys())}"
        )
    return DATASET_MANIFESTS[name_clean]


def list_supported_datasets() -> list[str]:
    """Return list of supported dataset identifiers."""
    return sorted(DATASET_MANIFESTS.keys())
