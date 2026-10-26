"""VERITAS P11 Demo package."""

from veritas.demo.web_search import (
    LiveSearchHit,
    fetch_page_text,
    hits_to_passages,
    search_web,
)
from veritas.demo.engine import (
    DemoEvidenceCard,
    DemoVerificationResponse,
    VeritasDemoEngine,
)

__all__ = [
    "LiveSearchHit",
    "fetch_page_text",
    "search_web",
    "hits_to_passages",
    "DemoEvidenceCard",
    "DemoVerificationResponse",
    "VeritasDemoEngine",
]
