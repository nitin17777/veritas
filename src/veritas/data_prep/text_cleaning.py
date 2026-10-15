
import re
from html import unescape


def clean_text(text: str | None) -> str:
    """Remove HTML, normalize whitespace, and preserve original casing."""
    if text is None:
        return ""

    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
