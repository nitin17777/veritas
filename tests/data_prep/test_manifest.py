"""Tests for the Data Source Manifest registry."""

import pytest

from veritas.data_prep.manifest import (
    DATASET_MANIFESTS,
    get_dataset_manifest,
    list_supported_datasets,
)


def test_manifest_contains_required_datasets():
    """All required project datasets must be registered in the manifest."""
    datasets = list_supported_datasets()
    assert "liar" in datasets
    assert "fever" in datasets
    assert "rawfc" in datasets
    assert "averitec" in datasets
    assert "custom_recent" in datasets


@pytest.mark.parametrize("name", ["liar", "fever", "rawfc", "averitec", "custom_recent"])
def test_manifest_metadata_fields_populated(name):
    """Every manifest entry must have required documentation fields."""
    meta = get_dataset_manifest(name)

    assert meta.name == name
    assert meta.source_url.startswith("http") or meta.source_url.startswith("internal://")
    assert meta.license
    assert meta.file_format
    assert len(meta.splits) >= 1
    assert len(meta.row_counts) >= 1
    assert len(meta.raw_fields) >= 1
    assert len(meta.original_labels) >= 1
    assert meta.mapping_status in {"activated", "proposed_pending_review"}


def test_get_manifest_aliases():
    """Manifest lookup must handle hyphens and aliases."""
    meta1 = get_dataset_manifest("liar-raw")
    assert meta1.name == "liar"

    meta2 = get_dataset_manifest("custom_recent")
    assert meta2.name == "custom_recent"


def test_get_manifest_unknown_raises():
    """Unknown dataset name raises KeyError."""
    with pytest.raises(KeyError, match="not found in manifest"):
        get_dataset_manifest("unknown_dataset_xyz")
