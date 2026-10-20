"""Tests for VERITAS corpus preparation and passage store."""

import json
from pathlib import Path

import pytest

from veritas.retrieval.chunking import ChunkerConfig
from veritas.retrieval.corpus import (
    PassageStore,
    build_corpus_from_processed,
    build_corpus_from_records,
    create_synthetic_corpus,
    extract_domain_from_url,
    extract_passages_from_record,
)
from veritas.retrieval.schema import Passage


def test_extract_domain_from_url():
    assert extract_domain_from_url("https://www.snopes.com/fact-check/test") == "snopes.com"
    assert extract_domain_from_url("http://who.int:8080/reports") == "who.int"
    assert extract_domain_from_url("https://cdc.gov") == "cdc.gov"
    assert extract_domain_from_url("invalid-url") is None
    assert extract_domain_from_url(None) is None


def test_extract_passages_preserves_required_fields():
    record = {
        "id": "rawfc:claim_001",
        "claim": "Claim text here.",
        "label": "False",
        "claim_date": "2024-01-10",
        "topic": "health",
        "split": "train",
        "evidence": [
            {
                "text": "Medical researchers confirmed that the treatment was safe and effective in phase 3 clinical trials.",
                "url": "https://www.nejm.org/doi/full/10.1056/test",
                "date": "2024-01-08",
                "domain": "nejm.org",
            }
        ],
        "metadata": {"dataset": "rawfc"},
    }

    passages = extract_passages_from_record(record)
    assert len(passages) == 1
    p = passages[0]

    assert p.passage_id == "rawfc:claim_001#e0"
    assert p.id == "rawfc:claim_001#e0"  # alias
    assert p.text.startswith("Medical researchers confirmed")
    assert p.url == "https://www.nejm.org/doi/full/10.1056/test"
    assert p.date == "2024-01-08"
    assert p.publication_date == "2024-01-08"  # alias
    assert p.domain == "nejm.org"
    assert p.metadata["source_claim_id"] == "rawfc:claim_001"
    assert p.metadata["dataset"] == "rawfc"


def test_domain_extracted_from_url_when_missing():
    record = {
        "id": "rec_1",
        "evidence": [
            {
                "text": "Evidence body snippet.",
                "url": "https://www.reuters.com/fact-check/article-id",
                "date": "2023-11-20",
                "domain": None,
            }
        ],
    }
    passages = extract_passages_from_record(record)
    assert len(passages) == 1
    assert passages[0].domain == "reuters.com"


def test_never_treat_claim_label_as_evidence():
    """VERITAS scientific principle: claim veracity labels must NEVER be treated as evidence."""
    record = {
        "id": "rec_forbidden",
        "claim": "The moon is made of green cheese.",
        "label": "False",  # Veracity label must NOT leak into evidence
        "split": "train",
        "evidence": [
            {
                "text": "Apollo lunar rock samples contain basalt and anorthosite minerals.",
                "url": "https://nasa.gov/moon-rocks",
                "date": "2021-05-12",
                "domain": "nasa.gov",
            }
        ],
    }

    passages = extract_passages_from_record(record)
    assert len(passages) == 1
    # Check that the label 'False' is completely absent from passage text
    assert "False" not in passages[0].text
    assert "label" not in passages[0].metadata
    # Ensure text is purely the evidence text
    assert passages[0].text == "Apollo lunar rock samples contain basalt and anorthosite minerals."


def test_never_create_fabricated_passages_when_evidence_is_missing():
    """VERITAS scientific principle: Never fabricate evidence passages for datasets without evidence (LIAR)."""
    liar_record = {
        "id": "liar:2635.json",
        "claim": "Says the Annies List political group supports third-trimester abortions on demand.",
        "label": "False",
        "claim_date": None,
        "topic": "abortion",
        "split": "train",
        "evidence": [],  # LIAR has no external evidence passages
        "metadata": {"dataset": "liar"},
    }

    passages = extract_passages_from_record(liar_record)
    assert passages == []  # MUST be empty list

    corpus = build_corpus_from_records([liar_record])
    assert len(corpus) == 0  # 0 passages created, no fabricated text


def test_chunking_long_evidence_creates_multiple_passages():
    # Long evidence of 350 words
    words = [f"info{i}" for i in range(350)]
    long_text = " ".join(words)

    record = {
        "id": "rec_long",
        "evidence": [
            {
                "text": long_text,
                "url": "https://example.org/report",
                "date": "2024-02-01",
                "domain": "example.org",
            }
        ],
    }

    cfg = ChunkerConfig(target_words=150, max_words=200, overlap_words=20)
    passages = extract_passages_from_record(record, chunker_config=cfg)

    assert len(passages) == 3
    assert passages[0].passage_id == "rec_long#e0#c0"
    assert passages[1].passage_id == "rec_long#e0#c1"
    assert passages[2].passage_id == "rec_long#e0#c2"

    for p in passages:
        assert p.url == "https://example.org/report"
        assert p.date == "2024-02-01"
        assert p.domain == "example.org"


def test_passage_store_preserves_ordering_and_indices():
    p1 = Passage(passage_id="p1", text="First text")
    p2 = Passage(passage_id="p2", text="Second text")
    p3 = Passage(passage_id="p3", text="Third text")

    store = PassageStore([p1, p2, p3])

    assert len(store) == 3
    assert store.passage_ids == ["p1", "p2", "p3"]
    assert store[0].passage_id == "p1"
    assert store[1].passage_id == "p2"
    assert store[2].passage_id == "p3"

    assert store.get_index_by_id("p1") == 0
    assert store.get_index_by_id("p2") == 1
    assert store.get_index_by_id("p3") == 2

    assert store.get_by_id("p2").text == "Second text"


def test_passage_store_duplicate_id_raises():
    p1 = Passage(passage_id="dup_id", text="Text 1")
    p2 = Passage(passage_id="dup_id", text="Text 2")
    store = PassageStore([p1])
    with pytest.raises(ValueError):
        store.add(p2)


def test_passage_store_jsonl_roundtrip(tmp_path: Path):
    target_file = tmp_path / "passages.jsonl"
    p1 = Passage(
        passage_id="psg:101",
        text="Sample passage one.",
        url="https://site.org/1",
        date="2024-01-01",
        domain="site.org",
        metadata={"custom": 42},
    )
    p2 = Passage(
        passage_id="psg:102",
        text="Sample passage two.",
        url=None,
        date=None,
        domain=None,
    )

    store = PassageStore([p1, p2])
    store.save_jsonl(target_file)

    loaded_store = PassageStore.load_jsonl(target_file)
    assert len(loaded_store) == 2
    assert loaded_store.passage_ids == ["psg:101", "psg:102"]

    loaded_p1 = loaded_store[0]
    assert loaded_p1.passage_id == "psg:101"
    assert loaded_p1.id == "psg:101"
    assert loaded_p1.text == "Sample passage one."
    assert loaded_p1.url == "https://site.org/1"
    assert loaded_p1.date == "2024-01-01"
    assert loaded_p1.publication_date == "2024-01-01"
    assert loaded_p1.domain == "site.org"
    assert loaded_p1.metadata["custom"] == 42


def test_passage_store_fingerprint_deterministic():
    p1 = Passage(passage_id="p1", text="Text A")
    p2 = Passage(passage_id="p2", text="Text B")

    store1 = PassageStore([p1, p2])
    store2 = PassageStore([p1, p2])
    assert store1.compute_fingerprint() == store2.compute_fingerprint()

    # Changing text or order must change fingerprint
    store_reversed = PassageStore([p2, p1])
    assert store1.compute_fingerprint() != store_reversed.compute_fingerprint()

    p2_modified = Passage(passage_id="p2", text="Text B modified")
    store_modified = PassageStore([p1, p2_modified])
    assert store1.compute_fingerprint() != store_modified.compute_fingerprint()


def test_create_synthetic_corpus():
    synthetic_items = [
        {"passage_id": "syn:1", "text": "Passage text 1", "url": "https://syn.org/1"},
        Passage(passage_id="syn:2", text="Passage text 2"),
        {
            "id": "mock_claim",
            "evidence": [{"text": "Extracted evidence text", "url": "https://syn.org/3"}],
        },
    ]

    store = create_synthetic_corpus(synthetic_items)
    assert len(store) == 3
    assert store.passage_ids == ["syn:1", "syn:2", "mock_claim#e0"]


def test_build_corpus_from_processed_directory(tmp_path: Path):
    # Setup mock processed directory with liar (no evidence) and rawfc (with evidence)
    processed_dir = tmp_path / "processed"
    liar_dir = processed_dir / "liar"
    rawfc_dir = processed_dir / "rawfc"
    liar_dir.mkdir(parents=True)
    rawfc_dir.mkdir(parents=True)

    # LIAR record (empty evidence)
    liar_rec = {
        "id": "liar:1",
        "claim": "Claim without evidence",
        "label": "True",
        "claim_date": None,
        "topic": "politics",
        "split": "train",
        "evidence": [],
    }
    with (liar_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(liar_rec) + "\n")

    # RAWFC record (with evidence)
    rawfc_rec = {
        "id": "rawfc:101",
        "claim": "Claim with evidence",
        "label": "False",
        "claim_date": "2024-01-01",
        "topic": "health",
        "split": "train",
        "evidence": [
            {
                "text": "Health authorities debunked the viral claim with clinical evidence.",
                "url": "https://health.gov/debunk",
                "date": "2023-12-30",
                "domain": "health.gov",
            }
        ],
    }
    with (rawfc_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(rawfc_rec) + "\n")

    store, summary = build_corpus_from_processed(processed_dir=processed_dir)

    assert summary["datasets"]["liar"]["evidence_items_found"] == 0
    assert summary["datasets"]["rawfc"]["evidence_items_found"] == 1
    assert len(store) == 1
    assert store[0].passage_id == "rawfc:101#e0"
    assert store[0].domain == "health.gov"
