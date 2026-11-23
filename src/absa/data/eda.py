"""Exploratory statistics over aspect-level records.

Pure, dependency-free counting functions so the same code can be used from
the exploration notebook, a script or a test, and every reported number
comes from the data actually passed in rather than being assumed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence


def class_balance(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Return the count of each ``sentiment`` label across ``records``."""
    return dict(Counter(str(record["sentiment"]) for record in records))


def aspect_frequency(
    records: Sequence[Mapping[str, object]], top_n: int | None = None
) -> list[tuple[str, int]]:
    """Return ``(aspect, count)`` pairs, most frequent first.

    Ties are broken by first appearance in ``records``, so the result is
    deterministic for a given input order.
    """
    counts = Counter(str(record["aspect"]) for record in records)
    # sorted() is stable and Counter preserves first-seen order, so equal counts keep
    # their first-appearance order (this is what makes the result deterministic).
    ordered = sorted(counts.items(), key=lambda pair: -pair[1])
    return ordered[:top_n] if top_n is not None else ordered


def domain_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Return the count of records for each ``domain``."""
    return dict(Counter(str(record["domain"]) for record in records))


def review_lengths(records: Sequence[Mapping[str, object]]) -> list[int]:
    """Return the character length of each record's ``review``, in input order."""
    return [len(str(record["review"])) for record in records]
