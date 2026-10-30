"""Aspect-level record contract shared by data, models and evaluation."""

from absa.config import SENTIMENT_LABELS

RECORD_FIELDS = ("review", "aspect", "sentiment")


def validate_record(record, allowed_labels=SENTIMENT_LABELS):
    """Return ``record`` unchanged if valid, otherwise raise ``ValueError``.

    A valid record is a mapping with non-empty string ``review`` and
    ``aspect`` fields and a ``sentiment`` drawn from ``allowed_labels``.
    """
    if not isinstance(record, dict):
        raise ValueError(f"record must be a dict, got {type(record).__name__}")
    missing = [f for f in RECORD_FIELDS if f not in record]
    if missing:
        raise ValueError(f"record is missing fields: {missing}")
    for field in ("review", "aspect"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field!r} must be a non-empty string")
    if record["sentiment"] not in allowed_labels:
        raise ValueError(
            f"unknown sentiment {record['sentiment']!r}; expected one of {allowed_labels}"
        )
    return record
