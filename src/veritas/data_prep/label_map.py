
LIAR_LABEL_MAP = {
    "pants-fire": "False",
    "false": "False",
    "barely-true": "Misleading",
    "half-true": "Partially True",
    "mostly-true": "True",
    "true": "True",
    "not enough info": "Unverified",
    "not-enough-info": "Unverified",
}

FEVER_LABEL_MAP = {
    "supported": "True",
    "refuted": "False",
    "not enough info": "Unverified",
    "not_enough_info": "Unverified",
}


def normalize_source_label(label: str) -> str:
    """Normalize label spelling and whitespace."""
    return label.strip().lower().replace("_", "-")


def map_label(label: str, dataset: str) -> str:
    """Map a dataset-specific label to a VERITAS label."""
    normalized = normalize_source_label(label)

    if dataset.lower() in {"liar", "liar_raw"}:
        mapping = LIAR_LABEL_MAP
    elif dataset.lower() == "fever":
        mapping = FEVER_LABEL_MAP
    else:
        raise ValueError(
            f"No label mapping configured for dataset: {dataset}"
        )

    # Accept underscore/hyphen variants consistently.
    normalized = normalized.replace("_", "-")

    # Support both FEVER spelling variants.
    if dataset.lower() == "fever" and normalized == "not-enough-info":
        normalized = "not enough info"

    try:
        return mapping[normalized]
    except KeyError:
        raise ValueError(
            f"Unknown label {label!r} for dataset {dataset!r}"
        ) from None
