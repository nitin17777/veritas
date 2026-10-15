"""Label mapping taxonomy for VERITAS fact verification.

Maps dataset-specific veracity labels to the five canonical VERITAS labels:
- True
- Partially True
- Misleading
- False
- Unverified

Per project guidelines:
- LIAR and FEVER mappings are activated.
- RAWFC and AVeriTeC label vocabularies are inspected and proposed below,
  remaining pending human review before default activation.
"""

from __future__ import annotations

LIAR_LABEL_MAP: dict[str, str] = {
    "pants-fire": "False",
    "false": "False",
    "barely-true": "Misleading",
    "half-true": "Partially True",
    "mostly-true": "True",
    "true": "True",
    "not enough info": "Unverified",
    "not-enough-info": "Unverified",
}

FEVER_LABEL_MAP: dict[str, str] = {
    "supported": "True",
    "supports": "True",
    "refuted": "False",
    "refutes": "False",
    "not enough info": "Unverified",
    "not-enough-info": "Unverified",
}

# ==============================================================================
# Proposed Mappings (Pending Human Review - Not activated by default)
# ==============================================================================

# RAWFC (Snopes-sourced claims, Hu et al., COLING 2022)
PROPOSED_RAWFC_LABEL_MAP: dict[str, str] = {
    "true": "True",
    "mostly true": "True",
    "mostly-true": "True",
    "mixture": "Partially True",
    "half": "Partially True",
    "half true": "Partially True",
    "half-true": "Partially True",
    "mostly false": "Misleading",
    "mostly-false": "Misleading",
    "false": "False",
    "unproven": "Unverified",
    "undetermined": "Unverified",
}

# AVeriTeC (Real-world web claims, Schlichtkrull et al., NeurIPS 2023)
PROPOSED_AVERITEC_LABEL_MAP: dict[str, str] = {
    "supported": "True",
    "refuted": "False",
    # Note for Human Review: "Conflicting Evidence/Cherry-picking" represents nuanced
    # misrepresentation. We propose "Misleading" (or "Partially True").
    "conflicting evidence/cherry-picking": "Misleading",
    "conflicting-evidence/cherry-picking": "Misleading",
    "conflicting evidence": "Misleading",
    "conflicting": "Misleading",
    "not enough evidence": "Unverified",
    "not-enough-evidence": "Unverified",
}

CUSTOM_LABEL_MAP: dict[str, str] = {
    "true": "True",
    "partially true": "Partially True",
    "partially-true": "Partially True",
    "misleading": "Misleading",
    "false": "False",
    "unverified": "Unverified",
}


def normalize_source_label(label: str) -> str:
    """Normalize label spelling and whitespace."""
    return label.strip().lower().replace("_", "-")


def map_label(
    label: str,
    dataset: str,
    custom_mapping: dict[str, str] | None = None,
) -> str:
    """Map a dataset-specific label to a canonical VERITAS label.

    Parameters:
        label: The raw source label string.
        dataset: The dataset identifier ('liar', 'fever', 'rawfc', 'averitec', 'custom').
        custom_mapping: Optional explicit mapping dictionary to override or provide
                        mappings for datasets pending human review.
    """
    normalized = normalize_source_label(label)
    ds = dataset.lower()

    if custom_mapping is not None:
        # Check normalized with hyphens, spaces, and original
        for candidate in (normalized, normalized.replace("-", " "), label.strip().lower(), label.strip()):
            if candidate in custom_mapping:
                return custom_mapping[candidate]
        raise ValueError(
            f"Unknown label {label!r} in custom mapping for dataset {dataset!r}"
        )

    if ds in {"liar", "liar_raw", "liar-raw"}:
        mapping = LIAR_LABEL_MAP
    elif ds == "fever":
        mapping = FEVER_LABEL_MAP
        if normalized == "not-enough-info":
            normalized = "not enough info"
    elif ds == "custom":
        mapping = CUSTOM_LABEL_MAP
        normalized = normalized.replace("-", " ")
    elif ds in {"rawfc", "averitec"}:
        raise ValueError(
            f"Label mapping for dataset {dataset!r} has not been activated. "
            f"Proposed mappings exist in PROPOSED_{dataset.upper()}_LABEL_MAP and "
            f"require human review before activation. Pass custom_mapping to proceed."
        )
    else:
        raise ValueError(
            f"No label mapping configured for dataset: {dataset}"
        )

    try:
        return mapping[normalized]
    except KeyError:
        # Also try space variant
        space_var = normalized.replace("-", " ")
        if space_var in mapping:
            return mapping[space_var]
        raise ValueError(
            f"Unknown label {label!r} for dataset {dataset!r}"
        ) from None
