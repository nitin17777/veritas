"""Corpus preparation and passage store for VERITAS evidence retrieval.

Extracts, normalizes, chunks, and indexes evidence passages from available
dataset evidence records while strictly adhering to VERITAS constraints:
- Never treat a claim's veracity label as evidence.
- Never fabricate passages to fill missing evidence (e.g. LIAR).
- Preserve passage ID, text, URL, publication date, and domain.
- Maintain deterministic ordering so index indices map 1:1 to passage IDs.
- Support small synthetic corpora for testing.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from veritas.data_prep.schema import validate_claim_record
from veritas.data_prep.text_cleaning import clean_text
from veritas.retrieval.chunking import ChunkerConfig, chunk_text
from veritas.retrieval.schema import Passage

logger = logging.getLogger("veritas.retrieval.corpus")


def extract_domain_from_url(url: str | None) -> str | None:
    """Extract normalized internet domain from a URL string."""
    if not url or not isinstance(url, str):
        return None
    try:
        parsed = urlsplit(url.strip())
        netloc = parsed.netloc.lower()
        # Strip port if present
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        # Strip leading www.
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or None
    except Exception:
        return None


class PassageStore:
    """In-memory and file-backed store for evidence passages.

    Preserves strict sequential ordering to ensure BM25 internal indices
    align perfectly with passage IDs and metadata.
    """

    def __init__(self, passages: list[Passage] | None = None) -> None:
        self._passages: list[Passage] = list(passages) if passages else []
        self._id_to_passage: dict[str, Passage] = {}
        self._id_to_index: dict[str, int] = {}
        self._rebuild_index_lookups()

    def _rebuild_index_lookups(self) -> None:
        self._id_to_passage.clear()
        self._id_to_index.clear()
        for idx, passage in enumerate(self._passages):
            self._id_to_passage[passage.passage_id] = passage
            self._id_to_index[passage.passage_id] = idx

    @property
    def passages(self) -> list[Passage]:
        """Ordered list of passages."""
        return self._passages

    @property
    def passage_ids(self) -> list[str]:
        """Ordered list of passage IDs."""
        return [p.passage_id for p in self._passages]

    def __len__(self) -> int:
        return len(self._passages)

    def __getitem__(self, idx: int) -> Passage:
        return self._passages[idx]

    def __iter__(self) -> Iterator[Passage]:
        return iter(self._passages)

    def get_by_id(self, passage_id: str) -> Passage | None:
        """Retrieve a passage by its unique passage ID."""
        return self._id_to_passage.get(passage_id)

    def get_index_by_id(self, passage_id: str) -> int | None:
        """Retrieve the zero-based corpus index for a passage ID."""
        return self._id_to_index.get(passage_id)

    def add(self, passage: Passage) -> None:
        """Add a passage to the end of the store."""
        if passage.passage_id in self._id_to_passage:
            raise ValueError(f"Duplicate passage_id '{passage.passage_id}' cannot be added.")
        idx = len(self._passages)
        self._passages.append(passage)
        self._id_to_passage[passage.passage_id] = passage
        self._id_to_index[passage.passage_id] = idx

    def compute_fingerprint(self) -> str:
        """Compute a deterministic SHA-256 hash of the ordered passages."""
        hasher = hashlib.sha256()
        for p in self._passages:
            hasher.update(p.passage_id.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(p.text.encode("utf-8"))
            hasher.update(b"\x00")
            if p.url:
                hasher.update(p.url.encode("utf-8"))
            hasher.update(b"\x00")
            if p.date:
                hasher.update(p.date.encode("utf-8"))
            hasher.update(b"\x00")
            if p.domain:
                hasher.update(p.domain.encode("utf-8"))
            hasher.update(b"\x01")
        return hasher.hexdigest()

    def save_jsonl(self, path: Path | str) -> None:
        """Serialize passage store to a JSONL file preserving exact sequence."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for p in self._passages:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path | str) -> PassageStore:
        """Load passage store from a JSONL file preserving exact sequence."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Passage store file not found: {target}")

        passages: list[Passage] = []
        with target.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    passages.append(Passage.from_dict(data))
                except Exception as exc:
                    raise ValueError(
                        f"Malformed passage record at line {line_no} in {target}: {exc}"
                    ) from exc

        return cls(passages)


def extract_passages_from_record(
    record: dict[str, Any],
    chunker_config: ChunkerConfig | None = None,
) -> list[Passage]:
    """Extract chunked Passage objects strictly from a claim record's evidence items.

    CONSTRAINTS ENFORCED:
    - Never uses record['label'] as evidence (veracity ground truth is never evidence text).
    - If record['evidence'] is empty, returns an empty list [] (never creates fabricated passages).
    - Preserves passage ID, text, URL, publication date, and domain.
    """
    if chunker_config is None:
        chunker_config = ChunkerConfig()

    claim_id = str(record.get("id", "claim_unknown")).strip()
    evidence_list = record.get("evidence", [])

    if not isinstance(evidence_list, list) or len(evidence_list) == 0:
        return []

    passages: list[Passage] = []

    for ev_idx, ev in enumerate(evidence_list):
        if not isinstance(ev, dict):
            continue

        raw_text = ev.get("text")
        if not raw_text or not isinstance(raw_text, str):
            continue

        cleaned_text = clean_text(raw_text)
        if not cleaned_text:
            continue

        raw_url = ev.get("url")
        url = str(raw_url).strip() if raw_url else None

        raw_date = ev.get("date") or ev.get("publication_date")
        date = str(raw_date).strip() if raw_date else None

        domain = ev.get("domain")
        if domain:
            domain = str(domain).strip().lower()
        elif url:
            domain = extract_domain_from_url(url)

        # Chunk the evidence document targeting 100-200 words with 20-word overlap
        text_chunks = chunk_text(
            cleaned_text,
            target_words=chunker_config.target_words,
            max_words=chunker_config.max_words,
            overlap_words=chunker_config.overlap_words,
            min_words=chunker_config.min_words,
        )

        for chunk_idx, chunk_str in enumerate(text_chunks):
            # Unique, deterministic passage ID
            if len(text_chunks) == 1:
                psg_id = f"{claim_id}#e{ev_idx}"
            else:
                psg_id = f"{claim_id}#e{ev_idx}#c{chunk_idx}"

            psg_meta = {
                "source_claim_id": claim_id,
                "evidence_index": ev_idx,
                "chunk_index": chunk_idx,
                "total_chunks": len(text_chunks),
            }
            if "metadata" in record and isinstance(record["metadata"], dict):
                if "dataset" in record["metadata"]:
                    psg_meta["dataset"] = record["metadata"]["dataset"]

            passages.append(
                Passage(
                    passage_id=psg_id,
                    text=chunk_str,
                    url=url,
                    date=date,
                    domain=domain,
                    metadata=psg_meta,
                )
            )

    return passages


def build_corpus_from_records(
    records: list[dict[str, Any]],
    chunker_config: ChunkerConfig | None = None,
    deduplicate: bool = True,
) -> PassageStore:
    """Build a PassageStore from normalized ClaimRecord dictionaries.

    Args:
        records: List of normalized ClaimRecords.
        chunker_config: Chunking configuration.
        deduplicate: If True, consolidates identical (url, text) chunks into a single passage,
                     tracking all referring claim IDs in metadata['referring_claims'].

    Returns:
        Populated PassageStore.
    """
    if chunker_config is None:
        chunker_config = ChunkerConfig()

    seen_signatures: dict[str, Passage] = {}
    ordered_passages: list[Passage] = []

    for record in records:
        passages = extract_passages_from_record(record, chunker_config=chunker_config)
        for psg in passages:
            if deduplicate:
                # Content signature: URL + text
                sig = f"{psg.url or ''}:::{psg.text}"
                if sig in seen_signatures:
                    existing = seen_signatures[sig]
                    ref_claims = existing.metadata.setdefault("referring_claims", [existing.metadata.get("source_claim_id")])
                    src_id = psg.metadata.get("source_claim_id")
                    if src_id and src_id not in ref_claims:
                        ref_claims.append(src_id)
                    continue
                seen_signatures[sig] = psg

            ordered_passages.append(psg)

    return PassageStore(ordered_passages)


def build_corpus_from_processed(
    processed_dir: Path | str = "data/processed",
    datasets: list[str] | None = None,
    splits: list[str] | None = None,
    chunker_config: ChunkerConfig | None = None,
    deduplicate: bool = True,
) -> tuple[PassageStore, dict[str, Any]]:
    """Build a PassageStore by ingesting processed JSONL dataset files.

    Inspects evidence availability across datasets. For datasets with 0 evidence
    (such as LIAR), records the absence and never fabricates synthetic passages.

    Args:
        processed_dir: Directory containing processed dataset folders.
        datasets: Subsets of datasets to inspect (e.g. ['rawfc', 'averitec', 'liar']).
                  If None, inspects all subdirectories found in processed_dir.
        splits: List of splits to load (e.g. ['train', 'validation', 'test', 'heldout_topic']).
                If None, loads all available *.jsonl files in each dataset folder.
        chunker_config: Chunking configuration.
        deduplicate: Whether to deduplicate identical evidence text passages.

    Returns:
        Tuple of (PassageStore, summary_audit_dict).
    """
    processed_root = Path(processed_dir)
    if not processed_root.is_dir():
        logger.warning("Processed directory '%s' does not exist.", processed_root)
        return PassageStore([]), {"total_passages": 0, "datasets": {}}

    if datasets is None:
        datasets = [d.name for d in processed_root.iterdir() if d.is_dir()]

    summary: dict[str, Any] = {
        "datasets": {},
        "total_records_scanned": 0,
        "total_evidence_items_found": 0,
        "total_passages_created": 0,
    }

    all_records: list[dict[str, Any]] = []

    for ds_name in datasets:
        ds_dir = processed_root / ds_name
        if not ds_dir.is_dir():
            logger.info("Dataset folder '%s' not found. Skipping.", ds_dir)
            continue

        if splits:
            jsonl_files = [ds_dir / f"{s}.jsonl" for s in splits if (ds_dir / f"{s}.jsonl").is_file()]
        else:
            jsonl_files = sorted(ds_dir.glob("*.jsonl"))

        ds_records = 0
        ds_evidence = 0

        for jsonl_file in jsonl_files:
            try:
                with jsonl_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        rec = json.loads(line_str)
                        all_records.append(rec)
                        ds_records += 1
                        evs = rec.get("evidence", [])
                        if isinstance(evs, list):
                            ds_evidence += len(evs)
            except Exception as exc:
                logger.warning("Error reading %s: %s", jsonl_file, exc)

        summary["datasets"][ds_name] = {
            "records_scanned": ds_records,
            "evidence_items_found": ds_evidence,
            "has_evidence": ds_evidence > 0,
        }

        if ds_evidence == 0:
            logger.info(
                "Dataset '%s': %d claims loaded with 0 evidence records. "
                "Per VERITAS scientific policy, no fabricated passages will be created.",
                ds_name,
                ds_records,
            )
        else:
            logger.info(
                "Dataset '%s': %d claims loaded with %d evidence records.",
                ds_name,
                ds_records,
                ds_evidence,
            )

        summary["total_records_scanned"] += ds_records
        summary["total_evidence_items_found"] += ds_evidence

    passage_store = build_corpus_from_records(
        all_records,
        chunker_config=chunker_config,
        deduplicate=deduplicate,
    )

    summary["total_passages_created"] = len(passage_store)
    logger.info("Total passages extracted for corpus: %d", len(passage_store))

    return passage_store, summary


def create_synthetic_corpus(
    items: list[dict[str, Any] | Passage],
    chunker_config: ChunkerConfig | None = None,
) -> PassageStore:
    """Create a PassageStore from synthetic passages or mock claim records for tests.

    Supports:
    - List of Passage instances directly.
    - List of Passage-like dictionaries containing 'passage_id' (or 'id') and 'text'.
    - List of ClaimRecord dictionaries containing 'id' and 'evidence'.
    """
    passages: list[Passage] = []

    for idx, item in enumerate(items):
        if isinstance(item, Passage):
            passages.append(item)
        elif isinstance(item, dict):
            if "evidence" in item:
                # Treated as a ClaimRecord
                extracted = extract_passages_from_record(item, chunker_config=chunker_config)
                passages.extend(extracted)
            else:
                # Treated as a Passage dictionary
                p_id = item.get("passage_id") or item.get("id") or f"synthetic_{idx}"
                text = item.get("text", "")
                passages.append(
                    Passage(
                        passage_id=str(p_id),
                        text=str(text),
                        url=item.get("url"),
                        date=item.get("date") or item.get("publication_date"),
                        domain=item.get("domain") or extract_domain_from_url(item.get("url")),
                        metadata=dict(item.get("metadata", {})),
                    )
                )
        else:
            raise TypeError(f"Unsupported item type in synthetic corpus: {type(item)}")

    return PassageStore(passages)
