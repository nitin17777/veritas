"""VERITAS P8 batch runner with crash-resume support.

Design
------
* Reads a JSONL file of claim records (``id``, ``claim``, optional
  ``claim_date``).
* Writes one Verdict per line to an output JSONL file.
* **Resume support**: on startup the runner reads all ``claim_id`` values
  already present in the output file and skips those claims.  A crash
  mid-batch can therefore be resumed by re-running the same command without
  any extra flags.
* Atomic write: each verdict is flushed immediately after processing so
  partial runs are always valid JSONL.
* Keeps a checkpoint file ``<output>.ckpt.json`` that records the run
  configuration, the start timestamp, total/done/failed counts, and the path
  to the output file.  This lets an operator inspect progress without
  tailing the output JSONL.

Usage (CLI)
-----------
    python scripts/run_pipeline_batch.py \\
        --input  data/claims.jsonl \\
        --output results/verdicts.jsonl \\
        --config configs/pipeline.yaml

    # Resume a crashed run by re-running the same command:
    python scripts/run_pipeline_batch.py \\
        --input  data/claims.jsonl \\
        --output results/verdicts.jsonl \\
        --config configs/pipeline.yaml
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from veritas.pipeline.pipeline import PipelineConfig, Verdict, VeritasPipeline
from veritas.retrieval.hybrid import HybridRetriever
from veritas.verification.llm_client import BaseLLMClient

logger = logging.getLogger("veritas.pipeline.batch")


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

@dataclass
class BatchCheckpoint:
    """Lightweight progress checkpoint persisted to disk as JSON."""

    run_id: str
    input_path: str
    output_path: str
    started_at: str
    total_claims: int = 0
    done: int = 0
    failed: int = 0
    skipped: int = 0
    finished_at: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "started_at": self.started_at,
            "total_claims": self.total_claims,
            "done": self.done,
            "failed": self.failed,
            "skipped": self.skipped,
            "finished_at": self.finished_at,
            "config_snapshot": self.config_snapshot,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BatchCheckpoint:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


# ---------------------------------------------------------------------------
# Input reading
# ---------------------------------------------------------------------------

def _iter_claims(input_path: Path) -> Iterator[dict[str, Any]]:
    """Yield raw claim dicts from a JSONL file, skipping blank lines."""
    with input_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if not isinstance(record, dict):
                    logger.warning("Line %d: expected JSON object, skipping.", lineno)
                    continue
                yield record
            except json.JSONDecodeError as exc:
                logger.warning("Line %d: malformed JSON, skipping – %s", lineno, exc)


def _load_completed_ids(output_path: Path) -> set[str]:
    """Return the set of claim_ids already present in the output file."""
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
                cid = rec.get("claim_id")
                if cid:
                    completed.add(str(cid))
            except Exception:
                pass
    return completed


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

class BatchRunner:
    """Crash-resumable batch runner for the VERITAS pipeline.

    Parameters
    ----------
    pipeline:
        Pre-built ``VeritasPipeline`` instance.  Sharing a single pipeline
        object avoids repeated index loading across thousands of claims.
    output_path:
        Path to the JSONL file where verdicts are appended.
    run_id:
        Unique identifier for this batch run (used in verdict records and
        checkpoint file).
    """

    def __init__(
        self,
        pipeline: VeritasPipeline,
        output_path: Path | str,
        run_id: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.output_path.with_suffix(".ckpt.json")
        self.run_id = run_id or _make_run_id()

    def run(
        self,
        input_path: Path | str,
        show_progress: bool = True,
    ) -> BatchCheckpoint:
        """Process all claims in *input_path*, appending verdicts to *output_path*.

        Already-processed claim IDs (detected from the output file) are
        skipped, enabling transparent crash-resume.

        Returns the final checkpoint.
        """
        input_path = Path(input_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Count total lines for progress reporting
        all_claims = list(_iter_claims(input_path))
        total = len(all_claims)

        # Identify already-done claim IDs for resume
        completed_ids = _load_completed_ids(self.output_path)
        n_skipped = len(completed_ids)
        if n_skipped:
            logger.info(
                "Resuming batch run '%s': %d/%d claims already completed.",
                self.run_id, n_skipped, total,
            )

        ckpt = BatchCheckpoint(
            run_id=self.run_id,
            input_path=str(input_path),
            output_path=str(self.output_path),
            started_at=datetime.now(timezone.utc).isoformat(),
            total_claims=total,
            skipped=n_skipped,
            config_snapshot=self.pipeline.config.to_dict(),
        )
        ckpt.save(self.ckpt_path)

        done = 0
        failed = 0

        with self.output_path.open("a", encoding="utf-8") as out_fh:
            for idx, record in enumerate(all_claims, start=1):
                claim_id = str(record.get("id") or record.get("claim_id") or f"claim_{idx}")
                if claim_id in completed_ids:
                    continue  # resume: skip already-done

                claim = str(record.get("claim", "")).strip()
                claim_date = record.get("claim_date") or record.get("date")

                if not claim:
                    logger.warning("claim_id=%s has empty claim text; skipping.", claim_id)
                    n_skipped += 1
                    continue

                try:
                    verdict = self.pipeline.verify(
                        claim=claim,
                        claim_date=claim_date,
                        claim_id=claim_id,
                        run_id=self.run_id,
                    )
                    out_fh.write(json.dumps(verdict.to_dict(), ensure_ascii=False) + "\n")
                    out_fh.flush()
                    done += 1
                except Exception as exc:
                    logger.error(
                        "Unhandled exception for claim_id=%s: %s", claim_id, exc,
                    )
                    # Write a failure sentinel so the claim is not re-attempted
                    sentinel = {
                        "claim_id": claim_id,
                        "claim": claim,
                        "claim_date": claim_date,
                        "label": "Unverified",
                        "confidence": 0.0,
                        "evidence_ids": [],
                        "explanation": f"Pipeline exception: {exc}",
                        "parse_failed": True,
                        "retrieved_passages": [],
                        "queries_used": [],
                        "stage_timings_s": {},
                        "model_metadata": {},
                        "run_id": self.run_id,
                        "pipeline_version": "p8.0",
                    }
                    out_fh.write(json.dumps(sentinel, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    failed += 1

                if show_progress and (idx % 10 == 0 or idx == total):
                    logger.info(
                        "Progress: %d/%d | done=%d failed=%d skipped=%d",
                        idx, total, done, failed, n_skipped,
                    )

                # Update checkpoint periodically
                ckpt.done = done
                ckpt.failed = failed
                ckpt.skipped = n_skipped
                if idx % 10 == 0:
                    ckpt.save(self.ckpt_path)

        ckpt.done = done
        ckpt.failed = failed
        ckpt.skipped = n_skipped
        ckpt.finished_at = datetime.now(timezone.utc).isoformat()
        ckpt.save(self.ckpt_path)

        logger.info(
            "Batch run '%s' complete: total=%d done=%d failed=%d skipped=%d",
            self.run_id, total, done, failed, n_skipped,
        )
        return ckpt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_id() -> str:
    from datetime import datetime, timezone
    import uuid
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def run_batch(
    input_path: str | Path,
    output_path: str | Path,
    config: PipelineConfig | dict[str, Any] | str | Path | None = None,
    retriever: HybridRetriever | None = None,
    llm_client: BaseLLMClient | None = None,
    run_id: str | None = None,
    show_progress: bool = True,
) -> BatchCheckpoint:
    """High-level convenience function for running a batch without building
    a ``VeritasPipeline`` manually.

    Suppresses the per-verdict ``save_verdicts`` in the pipeline (the batch
    runner writes to its own output path) to avoid duplicate writes.
    """
    if isinstance(config, (str, Path)):
        cfg = PipelineConfig.from_yaml(config)
    elif isinstance(config, dict):
        cfg = PipelineConfig.from_dict(config)
    elif isinstance(config, PipelineConfig):
        cfg = config
    else:
        cfg = PipelineConfig()

    # Disable pipeline-level verdict saving; the batch runner owns the sink.
    cfg.save_verdicts = False

    pipeline = VeritasPipeline(config=cfg, retriever=retriever, llm_client=llm_client)
    runner = BatchRunner(pipeline=pipeline, output_path=output_path, run_id=run_id)
    return runner.run(input_path=input_path, show_progress=show_progress)
