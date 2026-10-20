"""Document and evidence text chunking for VERITAS retrieval.

Implements sliding window text chunking targeting 100–200 words with 20-word overlap,
ensuring consistent passage length and semantic overlap across chunk boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkerConfig:
    """Configuration parameters for document text chunking."""

    target_words: int = 150
    max_words: int = 200
    overlap_words: int = 20
    min_words: int = 20

    def __post_init__(self) -> None:
        if self.target_words <= 0:
            raise ValueError(f"target_words must be positive, got {self.target_words}")
        if self.max_words < self.target_words:
            raise ValueError(
                f"max_words ({self.max_words}) must be >= target_words ({self.target_words})"
            )
        if self.overlap_words < 0:
            raise ValueError(f"overlap_words must be non-negative, got {self.overlap_words}")
        if self.overlap_words >= self.target_words:
            raise ValueError(
                f"overlap_words ({self.overlap_words}) must be less than target_words ({self.target_words})"
            )
        if self.min_words < 0:
            raise ValueError(f"min_words must be non-negative, got {self.min_words}")


def chunk_text(
    text: str,
    target_words: int = 150,
    max_words: int = 200,
    overlap_words: int = 20,
    min_words: int = 20,
) -> list[str]:
    """Chunk source text into approximately 100–200 words with 20-word overlap.

    Args:
        text: Raw document or evidence text string.
        target_words: Target word count per chunk (default 150, within 100–200).
        max_words: Maximum word threshold before splitting into chunks (default 200).
        overlap_words: Word overlap between successive chunks (default 20).
        min_words: Minimum words required for an independent trailing chunk.
                   Chunks shorter than this are merged into the preceding chunk.

    Returns:
        List of text chunks, preserving order and word overlaps.
    """
    if not text or not isinstance(text, str):
        return []

    # Normalize arbitrary whitespace
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    words = cleaned.split()
    total_words = len(words)

    # Documents within max_words are preserved as a single passage
    if total_words <= max_words:
        return [cleaned]

    step = max(1, target_words - overlap_words)
    chunks: list[str] = []
    i = 0

    while i < total_words:
        target_end = i + target_words
        # If the remaining un-covered words after target_end is fewer than min_words,
        # and absorbing them does not exceed max_words, include them in this chunk
        if target_end < total_words and (total_words - target_end) < min_words and (total_words - i) <= max_words:
            end = total_words
        else:
            end = min(target_end, total_words)

        chunk_words = words[i:end]
        chunks.append(" ".join(chunk_words))

        if end >= total_words:
            break

        i += step

    return chunks
