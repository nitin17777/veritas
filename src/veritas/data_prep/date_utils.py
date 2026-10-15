
from datetime import date, datetime


def normalize_date(value: str | date | datetime | None) -> str | None:
    """Return a date as ISO YYYY-MM-DD, or None when missing."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date().isoformat()

    if isinstance(value, date):
        return value.isoformat()

    value = value.strip()
    if not value:
        return None

    # Try common date formats first.
    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%b %d, %Y",
        "%B %d, %Y",
    )

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue

    raise ValueError(f"Unsupported date format: {value!r}")
