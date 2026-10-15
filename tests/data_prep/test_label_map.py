
import pytest

from veritas.data_prep.label_map import map_label


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("pants-fire", "False"),
        ("false", "False"),
        ("barely-true", "Misleading"),
        ("half-true", "Partially True"),
        ("mostly-true", "True"),
        ("true", "True"),
        ("not enough info", "Unverified"),
    ],
)
def test_liar_label_mapping(source, expected):
    assert map_label(source, "liar") == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("SUPPORTED", "True"),
        ("refuted", "False"),
        ("NOT ENOUGH INFO", "Unverified"),
        ("not_enough_info", "Unverified"),
    ],
)
def test_fever_label_mapping(source, expected):
    assert map_label(source, "fever") == expected


def test_unknown_label_fails():
    with pytest.raises(ValueError):
        map_label("unknown-label", "liar")


def test_unconfigured_dataset_fails():
    with pytest.raises(ValueError):
        map_label("true", "rawfc")
