"""VERITAS retrieval data contracts and schema definitions.

Defines Passage, SearchResult, and IndexMetadata classes conforming to the
VERITAS specification for evidence retrieval and passage indexing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Passage:
    """An evidence passage stored in the VERITAS retrieval corpus.

    Attributes:
        passage_id: Unique string identifier for the passage.
        text: Normalized body text of the evidence passage.
        url: Source web URL from which evidence was obtained, if available.
        date: ISO 8601 publication date string (YYYY-MM-DD), if available.
        domain: Fully-qualified internet domain of the source, if available.
        metadata: Arbitrary additional provenance metadata (source claim, dataset, chunk index, etc.).
    """

    passage_id: str
    text: str
    url: str | None = None
    date: str | None = None
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Alias for passage_id to maintain compatibility with legacy access patterns."""
        return self.passage_id

    @property
    def publication_date(self) -> str | None:
        """Alias for date representing the publication date."""
        return self.date

    def to_dict(self) -> dict[str, Any]:
        """Convert passage to a serializable dictionary preserving both aliases."""
        return {
            "passage_id": self.passage_id,
            "id": self.passage_id,
            "text": self.text,
            "url": self.url,
            "date": self.date,
            "publication_date": self.date,
            "domain": self.domain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Passage:
        """Reconstruct a Passage from a dictionary."""
        if not isinstance(data, dict):
            raise ValueError(f"Passage data must be a dictionary, got {type(data).__name__}")

        passage_id = data.get("passage_id") or data.get("id")
        if not passage_id or not isinstance(passage_id, str):
            raise ValueError("Passage data must contain a non-empty string 'passage_id' or 'id'")

        text = data.get("text")
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Passage data must contain non-empty string 'text'")

        date = data.get("date") or data.get("publication_date")
        if date is not None:
            date = str(date).strip() or None

        url = data.get("url")
        if url is not None:
            url = str(url).strip() or None

        domain = data.get("domain")
        if domain is not None:
            domain = str(domain).strip() or None

        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        return cls(
            passage_id=str(passage_id).strip(),
            text=str(text).strip(),
            url=url,
            date=date,
            domain=domain,
            metadata=dict(metadata),
        )


@dataclass
class SearchResult:
    """A scored passage result retrieved by a search query."""

    passage_id: str
    score: float
    rank: int
    passage: Passage

    def to_dict(self) -> dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "passage_id": self.passage_id,
            "score": float(self.score),
            "rank": int(self.rank),
            "passage": self.passage.to_dict(),
        }


@dataclass
class IndexMetadata:
    """Metadata detailing the configuration, provenance, and stats of an index."""

    index_name: str
    backend: str = "BM25Okapi"
    total_passages: int = 0
    corpus_fingerprint: str = ""
    config_fingerprint: str = ""
    created_at: str = ""
    k1: float = 1.5
    b: float = 0.75
    chunk_size: int = 150
    chunk_overlap: int = 20
    source_datasets: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexMetadata:
        if not isinstance(data, dict):
            raise ValueError(f"IndexMetadata data must be a dict, got {type(data).__name__}")
        return cls(
            index_name=str(data.get("index_name", "")),
            backend=str(data.get("backend", "BM25Okapi")),
            total_passages=int(data.get("total_passages", 0)),
            corpus_fingerprint=str(data.get("corpus_fingerprint", "")),
            config_fingerprint=str(data.get("config_fingerprint", "")),
            created_at=str(data.get("created_at", "")),
            k1=float(data.get("k1", 1.5)),
            b=float(data.get("b", 0.75)),
            chunk_size=int(data.get("chunk_size", 150)),
            chunk_overlap=int(data.get("chunk_overlap", 20)),
            source_datasets=list(data.get("source_datasets", [])),
            files=dict(data.get("files", {})),
            extra=dict(data.get("extra", {})),
        )
