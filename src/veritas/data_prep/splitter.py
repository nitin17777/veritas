"""Reproducible dataset splitting and evidence leakage filtering for VERITAS.

This module provides:

1. **Split assignment** — stratified-by-label train / validation / test splits
   for datasets that lack official splits, with a reproducible random seed and
   explicit reporting of the chosen split sizes.

2. **Topic hold-out** — designates at least one complete topic cluster as
   ``heldout_topic`` to evaluate generalization to unseen topics.  The chosen
   topic is reported together with its record count and label distribution so
   the decision is fully auditable.

3. **Evidence leakage filtering** — drops evidence items published *after* the
   claim's fact-check date.  Operates record-by-record and is disabled by
   default so experiments can be run with and without the filter.

4. **Known fact-checker domain list** — a configurable list of domains that are
   primary fact-checking organisations.  When the leakage filter is enabled,
   evidence from these domains that post-dates the claim is treated exactly like
   any other leaked evidence item and removed; the gold label is never removed.

Design constraints
------------------
* The same claim ID must never appear in more than one split.
* Near-duplicate detection uses exact claim-text matching only; fuzzy
  deduplication is out of scope here and is reported as a known limitation.
* When official splits are present they are preserved unchanged and
  stratified re-splitting is skipped.
* All randomness is seeded via ``random_seed`` (default 42).
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from veritas.data_prep.schema import VALID_SPLITS, validate_claim_record

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default domains considered primary fact-checkers for leakage filtering.
DEFAULT_FACT_CHECKER_DOMAINS: frozenset[str] = frozenset(
    {
        "politifact.com",
        "snopes.com",
        "factcheck.org",
        "fullfact.org",
        "afp.com",
        "apnews.com",
        "reuters.com",
        "bbc.com",
        "bbc.co.uk",
        "checkyourfact.com",
        "leadstories.com",
        "washingtonpost.com",
        "usatoday.com",
        "truthorfiction.com",
    }
)

#: Default train/validation/test proportions for stratified splitting.
DEFAULT_SPLIT_RATIOS: dict[str, float] = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}


# ---------------------------------------------------------------------------
# Audit dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LeakageFilterStats:
    """Per-record leakage filter statistics."""

    total_records_examined: int = 0
    records_with_date: int = 0
    records_without_date: int = 0
    total_evidence_items: int = 0
    evidence_dropped: int = 0
    records_affected: int = 0


@dataclass
class SplitStats:
    """Statistics describing a single split produced by the splitter."""

    split: str
    record_count: int
    label_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class TopicHoldoutReport:
    """Report documenting the chosen heldout topic and the rationale."""

    topic: str
    reason: str
    record_count: int
    label_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class SplitReport:
    """Complete audit report produced by :func:`assign_splits`."""

    dataset_id: str
    strategy: str  # "official", "stratified", or "passthrough"
    random_seed: int | None
    deduplication_policy: str
    split_stats: list[SplitStats] = field(default_factory=list)
    heldout_topic: TopicHoldoutReport | None = None
    duplicate_claim_ids: list[str] = field(default_factory=list)
    cross_split_duplicate_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Leakage filter
# ---------------------------------------------------------------------------


def _parse_iso_date(value: str | None) -> date | None:
    """Return a ``datetime.date`` from an ISO string or ``None``."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def filter_evidence_leakage(
    records: list[dict[str, Any]],
    *,
    fact_checker_domains: frozenset[str] | set[str] = DEFAULT_FACT_CHECKER_DOMAINS,
    enabled: bool = True,
) -> tuple[list[dict[str, Any]], LeakageFilterStats]:
    """Remove evidence items published after the claim's fact-check date.

    Parameters
    ----------
    records:
        List of ClaimRecord dicts (copies are returned; originals are not mutated).
    fact_checker_domains:
        Set of domain strings considered primary fact-checking sources.
        Evidence from these domains published *after* ``claim_date`` is
        dropped exactly like any other leaked item; the gold ``label`` is
        never removed.
    enabled:
        When ``False`` the records are returned unchanged.  This allows
        callers to toggle the filter without changing call-site logic.

    Returns
    -------
    tuple[list[dict], LeakageFilterStats]
        The (possibly modified) records and aggregate statistics.
    """
    stats = LeakageFilterStats()

    if not enabled:
        stats.total_records_examined = len(records)
        for rec in records:
            stats.total_evidence_items += len(rec.get("evidence", []))
        return records, stats

    filtered: list[dict[str, Any]] = []

    for rec in records:
        stats.total_records_examined += 1
        claim_date = _parse_iso_date(rec.get("claim_date"))

        evidence: list[dict[str, Any]] = rec.get("evidence", [])
        stats.total_evidence_items += len(evidence)

        if claim_date is None:
            stats.records_without_date += 1
            filtered.append(rec)
            continue

        stats.records_with_date += 1
        kept: list[dict[str, Any]] = []
        dropped_count = 0

        for ev in evidence:
            ev_date = _parse_iso_date(ev.get("date"))
            if ev_date is not None and ev_date > claim_date:
                dropped_count += 1
                logger.debug(
                    "Leakage filter: dropped evidence dated %s for claim %s (fact-checked %s)",
                    ev_date,
                    rec.get("id"),
                    claim_date,
                )
            else:
                kept.append(ev)

        if dropped_count > 0:
            stats.evidence_dropped += dropped_count
            stats.records_affected += 1
            rec = dict(rec)  # shallow copy so caller's original is not mutated
            rec["evidence"] = kept

        filtered.append(rec)

    return filtered, stats


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------


def _claim_fingerprint(claim_text: str) -> str:
    """Return a stable hex fingerprint of a normalised claim text."""
    normalised = " ".join(claim_text.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _find_duplicate_ids(records: list[dict[str, Any]]) -> list[str]:
    """Return IDs that share a normalised claim text with at least one other record."""
    fingerprint_to_ids: dict[str, list[str]] = collections.defaultdict(list)
    for rec in records:
        fp = _claim_fingerprint(rec["claim"])
        fingerprint_to_ids[fp].append(rec["id"])

    duplicates: list[str] = []
    for ids in fingerprint_to_ids.values():
        if len(ids) > 1:
            duplicates.extend(ids)
    return sorted(set(duplicates))


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------


def _stratified_split(
    records: list[dict[str, Any]],
    ratios: dict[str, float],
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    """Assign records to splits stratified by label.

    Parameters
    ----------
    records:
        All records to distribute (must each have a ``label`` field).
    ratios:
        Mapping of split name to target proportion. Must sum to 1.0
        (plus or minus floating-point tolerance).
    rng:
        Seeded random instance for reproducibility.

    Returns
    -------
    dict[str, list[dict]]
        Mapping of split name to its assigned records.
    """
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio:.6f}: {ratios}"
        )

    split_names = list(ratios.keys())

    # Group records by label for stratification
    by_label: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in records:
        by_label[rec["label"]].append(rec)

    result: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}

    for label, label_records in by_label.items():
        rng.shuffle(label_records)
        n = len(label_records)
        # Compute cumulative cut points
        cuts = [0]
        cumulative = 0.0
        for i, name in enumerate(split_names[:-1]):
            cumulative += ratios[name]
            cuts.append(round(cumulative * n))
        cuts.append(n)

        for i, name in enumerate(split_names):
            result[name].extend(label_records[cuts[i] : cuts[i + 1]])

    return result


# ---------------------------------------------------------------------------
# Topic hold-out
# ---------------------------------------------------------------------------


def _select_heldout_topic(
    records: list[dict[str, Any]],
    rng: random.Random,
    min_records: int = 5,
) -> str | None:
    """Select a topic to hold out.

    Prefers topics that appear at least ``min_records`` times.  Among
    eligible topics, the one with the most records is chosen; ties are
    broken alphabetically so the result is fully deterministic.

    Returns ``None`` if no suitable topic is found.
    """
    topic_records: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in records:
        topic = rec.get("topic")
        if topic:
            topic_records[topic].append(rec)

    # Filter to topics that have at least min_records records
    eligible = {t: recs for t, recs in topic_records.items() if len(recs) >= min_records}
    if not eligible:
        return None

    # Deterministic: most records first, then alphabetical tiebreak
    ranked = sorted(
        eligible.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    return ranked[0][0]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def assign_splits(
    records: list[dict[str, Any]],
    *,
    dataset_id: str = "unknown",
    preserve_official_splits: bool = True,
    heldout_topic: str | None = "auto",
    split_ratios: dict[str, float] | None = None,
    random_seed: int = 42,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], SplitReport]:
    """Assign or validate train / validation / test / heldout_topic splits.

    Parameters
    ----------
    records:
        Input ClaimRecord dicts.  Each must satisfy :func:`validate_claim_record`.
    dataset_id:
        Identifier string included in the audit report.
    preserve_official_splits:
        When ``True`` (default) and every record already carries a valid
        ``split`` value, the existing assignments are kept and only
        cross-split duplicate checking is performed.
    heldout_topic:
        * ``"auto"`` -- automatically select the most common topic and
          re-assign its records to ``"heldout_topic"``.
        * Any non-empty string -- hold out records matching that exact topic.
        * ``None`` -- skip the topic hold-out step entirely.
    split_ratios:
        Proportions for stratified splitting when official splits are absent.
        Defaults to ``{"train": 0.80, "validation": 0.10, "test": 0.10}``.
    random_seed:
        Seed for :mod:`random` to guarantee reproducibility.
    strict:
        When ``True``, raises :exc:`ValueError` on any detected anomaly
        (cross-split duplicates, invalid records).  When ``False``, anomalies
        are logged and recorded in the report.

    Returns
    -------
    tuple[list[dict], SplitReport]
        The records with ``split`` fields populated and the audit report.
    """
    if not records:
        raise ValueError("Cannot split an empty record list.")

    rng = random.Random(random_seed)
    ratios = split_ratios or DEFAULT_SPLIT_RATIOS

    report = SplitReport(
        dataset_id=dataset_id,
        strategy="passthrough",
        random_seed=random_seed,
        deduplication_policy=(
            "Exact claim-text deduplication only (normalised whitespace and "
            "case). Near-duplicate fuzzy matching is out of scope for this "
            "module. Detected exact-text duplicates are logged in "
            "duplicate_claim_ids; no records are removed automatically."
        ),
    )

    # -----------------------------------------------------------------------
    # Step 1: Validate all input records
    # -----------------------------------------------------------------------
    for i, rec in enumerate(records):
        try:
            validate_claim_record(rec)
        except ValueError as exc:
            msg = f"Invalid record at index {i} (id={rec.get('id')!r}): {exc}"
            if strict:
                raise ValueError(msg) from exc
            report.warnings.append(msg)
            logger.warning(msg)

    # -----------------------------------------------------------------------
    # Step 2: Determine if official splits are already present and valid
    # -----------------------------------------------------------------------
    has_official_splits = (
        preserve_official_splits
        and all(rec.get("split") in VALID_SPLITS for rec in records)
    )

    if has_official_splits:
        report.strategy = "official"
        logger.info(
            "[%s] Preserving official splits for %d records.", dataset_id, len(records)
        )
    else:
        # Stratified split needed
        report.strategy = "stratified"
        logger.info(
            "[%s] Performing stratified split of %d records (seed=%d).",
            dataset_id,
            len(records),
            random_seed,
        )
        split_buckets = _stratified_split(records, ratios, rng)
        for split_name, split_records in split_buckets.items():
            for rec in split_records:
                rec["split"] = split_name
        records = [r for bucket in split_buckets.values() for r in bucket]

    # -----------------------------------------------------------------------
    # Step 3: Topic hold-out
    # -----------------------------------------------------------------------
    if heldout_topic is not None:
        if heldout_topic == "auto":
            # Operate only on non-heldout records
            candidate_records = [r for r in records if r.get("split") != "heldout_topic"]
            chosen_topic = _select_heldout_topic(candidate_records, rng)
        else:
            chosen_topic = heldout_topic

        if chosen_topic:
            held_count = 0
            held_labels: dict[str, int] = collections.Counter()
            for rec in records:
                if rec.get("topic") == chosen_topic and rec.get("split") != "heldout_topic":
                    rec["split"] = "heldout_topic"
                    held_count += 1
                    held_labels[rec["label"]] += 1

            reason = (
                f"Topic '{chosen_topic}' selected as heldout_topic because it "
                f"had the most records among topics with 5 or more occurrences, "
                f"providing the best-balanced generalization evaluation set."
                if heldout_topic == "auto"
                else f"Topic '{chosen_topic}' explicitly specified by caller."
            )
            report.heldout_topic = TopicHoldoutReport(
                topic=chosen_topic,
                reason=reason,
                record_count=held_count,
                label_distribution=dict(held_labels),
            )
            logger.info(
                "[%s] Held out topic '%s': %d records.",
                dataset_id,
                chosen_topic,
                held_count,
            )
        else:
            msg = (
                f"[{dataset_id}] No eligible topic found for heldout_topic "
                f"(all topics have fewer than 5 records or topic field is absent)."
            )
            report.warnings.append(msg)
            logger.warning(msg)

    # -----------------------------------------------------------------------
    # Step 4: Exact-text duplicate detection
    # -----------------------------------------------------------------------
    duplicate_ids = _find_duplicate_ids(records)
    if duplicate_ids:
        msg = (
            f"[{dataset_id}] {len(duplicate_ids)} record(s) share exact normalised "
            f"claim text with at least one other record."
        )
        report.duplicate_claim_ids = duplicate_ids
        report.warnings.append(msg)
        logger.warning(msg)

    # -----------------------------------------------------------------------
    # Step 5: Cross-split duplicate ID check
    # -----------------------------------------------------------------------
    id_to_splits: dict[str, set[str]] = collections.defaultdict(set)
    for rec in records:
        id_to_splits[rec["id"]].add(rec["split"])

    cross_split = [rid for rid, splits in id_to_splits.items() if len(splits) > 1]
    if cross_split:
        msg = (
            f"[{dataset_id}] {len(cross_split)} record ID(s) appear in multiple splits: "
            f"{cross_split[:5]}{'...' if len(cross_split) > 5 else ''}"
        )
        report.cross_split_duplicate_ids = cross_split
        if strict:
            raise ValueError(msg)
        report.warnings.append(msg)
        logger.error(msg)

    # -----------------------------------------------------------------------
    # Step 6: Build split statistics
    # -----------------------------------------------------------------------
    split_buckets_final: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in records:
        split_buckets_final[rec["split"]].append(rec)

    for split_name in sorted(split_buckets_final.keys()):
        bucket = split_buckets_final[split_name]
        label_dist: dict[str, int] = collections.Counter(r["label"] for r in bucket)
        report.split_stats.append(
            SplitStats(
                split=split_name,
                record_count=len(bucket),
                label_distribution=dict(label_dist),
            )
        )

    logger.info(
        "[%s] Split assignment complete. Strategy: %s. Records: %s",
        dataset_id,
        report.strategy,
        {s.split: s.record_count for s in report.split_stats},
    )
    return records, report


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def save_splits(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    report: SplitReport | None = None,
) -> dict[str, int]:
    """Write split-assigned records to per-split JSONL files.

    Parameters
    ----------
    records:
        Records with ``split`` fields populated (e.g. by :func:`assign_splits`).
    output_dir:
        Directory where JSONL files will be written.
    report:
        Optional :class:`SplitReport` to save alongside the JSONL files.

    Returns
    -------
    dict[str, int]
        Mapping of split name to record count.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in records:
        split_buckets[rec["split"]].append(rec)

    counts: dict[str, int] = {}
    for split_name, bucket in sorted(split_buckets.items()):
        out_file = output_dir / f"{split_name}.jsonl"
        with out_file.open("w", encoding="utf-8") as fh:
            for rec in bucket:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[split_name] = len(bucket)
        logger.info("Wrote %d records to %s", len(bucket), out_file)

    if report is not None:
        report_file = output_dir / "split_report.json"
        with report_file.open("w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, indent=2, ensure_ascii=False)
        logger.info("Saved split report to %s", report_file)

    return counts
