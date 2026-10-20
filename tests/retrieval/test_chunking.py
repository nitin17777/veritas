"""Tests for VERITAS retrieval document chunking."""

import pytest

from veritas.retrieval.chunking import ChunkerConfig, chunk_text


def test_chunker_config_valid():
    cfg = ChunkerConfig(target_words=150, max_words=200, overlap_words=20, min_words=20)
    assert cfg.target_words == 150
    assert cfg.max_words == 200
    assert cfg.overlap_words == 20
    assert cfg.min_words == 20


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_words": 0},
        {"target_words": 200, "max_words": 150},
        {"overlap_words": -5},
        {"overlap_words": 150, "target_words": 150},
        {"min_words": -1},
    ],
)
def test_chunker_config_invalid_raises(kwargs):
    with pytest.raises(ValueError):
        ChunkerConfig(**kwargs)


def test_chunk_text_empty_and_whitespace():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []
    assert chunk_text(None) == []


def test_chunk_text_under_max_words_is_single_chunk():
    # 50 words document
    text = " ".join([f"word{i}" for i in range(50)])
    chunks = chunk_text(text, target_words=150, max_words=200, overlap_words=20)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_exact_max_words_is_single_chunk():
    text = " ".join([f"token{i}" for i in range(200)])
    chunks = chunk_text(text, target_words=150, max_words=200, overlap_words=20)
    assert len(chunks) == 1
    assert len(chunks[0].split()) == 200


def test_chunk_text_sliding_window_overlap():
    # 350 words: target 150, max 200, overlap 20
    text = " ".join([f"w{i}" for i in range(350)])
    chunks = chunk_text(text, target_words=150, max_words=200, overlap_words=20)

    assert len(chunks) == 3

    words_0 = chunks[0].split()
    words_1 = chunks[1].split()
    words_2 = chunks[2].split()

    assert len(words_0) == 150
    assert len(words_1) == 150
    assert len(words_2) == 90  # 350 - 260 = 90

    # Overlap between chunk 0 and chunk 1 must be exactly the 20 words: w130..w149
    overlap_0_1 = set(words_0[-20:])
    assert overlap_0_1 == set(words_1[:20])
    assert words_1[:20] == [f"w{i}" for i in range(130, 150)]

    # Overlap between chunk 1 and chunk 2 must be exactly the 20 words: w260..w279
    overlap_1_2 = set(words_1[-20:])
    assert overlap_1_2 == set(words_2[:20])
    assert words_2[:20] == [f"w{i}" for i in range(260, 280)]


def test_chunk_text_merges_small_trailing_remainder():
    # 225 words with step 130:
    # Chunk 0: 0..150
    # Remaining at i=130: 225 - 130 = 95 words. (>= min_words 20 -> chunk 1: 130..225, 95 words)
    text = " ".join([f"tok{i}" for i in range(225)])
    chunks = chunk_text(text, target_words=150, max_words=200, overlap_words=20, min_words=20)
    assert len(chunks) == 2

    # Now test with a tiny remainder: 160 words
    # Chunk 0: 0..150 (words 0..149)
    # i=130: remaining words from 130 is 160 - 130 = 30 words (>= 20)
    # What if text is 205 words?
    # Chunk 0: 0..150. At i=130, remaining is 75 words.
    # What if text has 275 words?
    # Chunk 0: 0..150. Chunk 1: 130..280 -> capped at 275.
    text_275 = " ".join([f"w{i}" for i in range(275)])
    c = chunk_text(text_275, target_words=150, max_words=200, overlap_words=20, min_words=20)
    assert len(c) == 2
    assert len(c[0].split()) == 150
    assert len(c[1].split()) == 145


def test_chunk_text_tiny_orphan_merged_into_previous():
    # Target 100, step 80, overlap 20, min_words 20
    # Text length 185 words (exceeds max 180):
    # Chunk 0: 0..100.
    # i=80: 80..180 (100 words).
    # i=160: remaining 185-160 = 5 words (< min_words 20) -> merged into Chunk 1!
    text = " ".join([f"w{i}" for i in range(185)])
    chunks = chunk_text(text, target_words=100, max_words=180, overlap_words=20, min_words=20)
    assert len(chunks) == 2
    words_1 = chunks[1].split()
    assert words_1[-5:] == [f"w{i}" for i in range(180, 185)]
