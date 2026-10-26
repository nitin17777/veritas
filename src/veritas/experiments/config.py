"""Experiment configuration models and factor isolation validation for VERITAS.

Ensures:
- Every experiment has a strongly typed configuration specifying its research question.
- Strict isolation of comparison factors (checks that ablations differ only in intended variables).
- Prevention of evaluation data leakage (split validation: never allows tuning on 'test').
- Clear reproducibility metadata (seeds, versions, pipeline parameters).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any

from veritas.config import ROOT_DIR, load_config
from veritas.pipeline.pipeline import PipelineConfig

logger = logging.getLogger("veritas.experiments.config")


class ExperimentConfigError(ValueError):
    """Raised when an experiment configuration fails validation or violates isolation."""


@dataclass
class ExperimentConfig:
    """Configuration for a specific VERITAS research experiment.

    Attributes:
        name: Unique identifier for the experiment run (e.g. 'rq1_llm_only').
        research_question: RQ identifier (e.g. 'RQ1', 'RQ2', ..., 'RQ7', 'Overall').
        description: Brief description of the experimental hypothesis.
        comparison_factor: The single independent variable under study.
        factor_value: The value assigned to the independent variable in this run.
        dataset_path: Path to the evaluation claims JSONL file.
        dataset_name: Dataset name identifier (e.g. 'liar', 'fever', 'fixture').
        dataset_version: Dataset release version string.
        split: Dataset split ('test' or 'val'; NEVER 'train').
        pipeline: Complete PipelineConfig instance.
        results_dir: Directory where predictions, metrics, and logs will be written.
        seed: Random seed for deterministic reproducibility.
        smoke_test: Whether this run is in smoke-test mode with reduced sample size.
        max_claims: Optional upper limit on evaluated claims (useful for fast smoke-testing).
        version: Experiment schema version.
    """

    name: str
    research_question: str = "RQ1"
    description: str = ""
    comparison_factor: str = ""
    factor_value: Any = None
    dataset_path: str = "data/processed/liar_test.jsonl"
    dataset_name: str = "liar"
    dataset_version: str = "v1.0"
    split: str = "test"
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    results_dir: str = "results/experiments"
    seed: int = 42
    smoke_test: bool = False
    max_claims: int | None = None
    version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ExperimentConfigError("Experiment name must not be empty.")
        if self.split.strip().lower() == "train":
            raise ExperimentConfigError(
                "Experiment evaluation split must NOT be 'train'. Never tune or evaluate on the training split."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Construct ExperimentConfig from a raw dictionary (e.g. loaded from YAML)."""
        exp = data.get("experiment", {})
        ds = data.get("dataset", {})
        paths = data.get("paths", {})
        smoke = data.get("smoke_test", {})

        name = str(exp.get("name", "unnamed_experiment")).strip()
        rq = str(exp.get("research_question", exp.get("rq", "RQ1"))).strip()
        desc = str(exp.get("description", "")).strip()
        comp_factor = str(exp.get("comparison_factor", "")).strip()
        factor_val = exp.get("factor_value")

        ds_path = str(ds.get("path", "data/processed/liar_test.jsonl"))
        ds_name = str(ds.get("name", "liar"))
        ds_ver = str(ds.get("version", "v1.0"))
        split = str(ds.get("split", "test")).strip().lower()

        base_results_dir = str(paths.get("results_dir", "results/experiments"))
        exp_results_dir = str(Path(base_results_dir) / name)

        seed = int(exp.get("seed", 42))
        is_smoke = bool(smoke.get("enabled", False))
        max_claims = smoke.get("max_claims")
        if max_claims is not None:
            max_claims = int(max_claims)

        # Build nested PipelineConfig from raw dictionary
        pipeline_cfg = PipelineConfig.from_dict(data)
        pipeline_cfg.results_dir = exp_results_dir

        return cls(
            name=name,
            research_question=rq,
            description=desc,
            comparison_factor=comp_factor,
            factor_value=factor_val,
            dataset_path=ds_path,
            dataset_name=ds_name,
            dataset_version=ds_ver,
            split=split,
            pipeline=pipeline_cfg,
            results_dir=exp_results_dir,
            seed=seed,
            smoke_test=is_smoke,
            max_claims=max_claims,
            version=str(exp.get("version", "1.0")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load and parse an experiment YAML configuration."""
        raw = load_config(path)
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary representation."""
        res = asdict(self)
        res["pipeline"] = self.pipeline.to_dict()
        return res


def validate_factor_isolation(
    base_config: ExperimentConfig,
    variant_config: ExperimentConfig,
    allowed_difference_keys: list[str] | None = None,
) -> list[str]:
    """Validate that two compared experiment configurations differ ONLY in the intended factor.

    A key scientific requirement is ensuring that an ablation or comparative experiment
    differs strictly in the factor being studied, without confounding unintended changes.

    Args:
        base_config: Reference baseline experiment configuration.
        variant_config: Experimental variant configuration.
        allowed_difference_keys: Explicit list of dot-notated keys allowed to differ.
            If omitted, uses variant_config.comparison_factor.

    Returns:
        List of unintended difference descriptions (empty if isolation is strictly preserved).
    """
    allowed_keys = set(allowed_difference_keys or [])
    if variant_config.comparison_factor:
        allowed_keys.add(variant_config.comparison_factor)

    # Standard administrative keys that naturally differ between runs
    ignorable_keys = {
        "name",
        "description",
        "results_dir",
        "pipeline.results_dir",
        "factor_value",
        "comparison_factor",
    }

    def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for k, v in d.items():
            curr_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_dict(v, curr_key))
            else:
                flat[curr_key] = v
        return flat

    flat_base = flatten_dict(base_config.to_dict())
    flat_var = flatten_dict(variant_config.to_dict())

    all_keys = set(flat_base.keys()) | set(flat_var.keys())
    violations: list[str] = []

    for k in sorted(all_keys):
        if k in ignorable_keys or any(k.startswith(f"{ign}.") for ign in ignorable_keys):
            continue

        base_val = flat_base.get(k)
        var_val = flat_var.get(k)

        if base_val != var_val:
            # Check if this key or its parent is explicitly allowed
            is_allowed = any(
                k == ak or k.startswith(f"{ak}.") or ak.startswith(f"{k}.")
                for ak in allowed_keys
            )
            if not is_allowed:
                violations.append(
                    f"Unintended configuration difference at '{k}': baseline={base_val!r} vs variant={var_val!r}"
                )

    return violations
