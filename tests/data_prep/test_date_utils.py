
from datetime import date, datetime

import pytest

from veritas.data_prep.date_utils import normalize_date


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2025-03-04", "2025-03-04"),
        ("2025/03/04", "2025-03-04"),
        ("03/04/2025", "2025-03-04"),
        ("Mar 4, 2025", "2025-03-04"),
        (date(2025, 3, 4), "2025-03-04"),
        (datetime(2025, 3, 4, 15, 30), "2025-03-04"),
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_normalize_date(source, expected):
    assert normalize_date(source) == expected


def test_invalid_date_fails():
    with pytest.raises(ValueError):
        normalize_date("not-a-date")
