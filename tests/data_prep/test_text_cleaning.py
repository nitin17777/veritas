
import pytest

from veritas.data_prep.text_cleaning import clean_text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("  Hello   world  ", "Hello world"),
        ("<p>Hello <b>world</b></p>", "Hello world"),
        ("A &amp; B", "A & B"),
        ("The CLAIM is TRUE", "The CLAIM is TRUE"),
        (None, ""),
        ("", ""),
    ],
)
def test_clean_text(source, expected):
    assert clean_text(source) == expected
