"""Guards against train/test leakage, which silently inflates evaluation scores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def normalise_review(text: str) -> str:
    """Lowercase ``text`` and collapse whitespace, the key used to compare reviews."""
    return " ".join(text.lower().split())


def find_overlap(
    train: Iterable[Mapping[str, object]], test: Iterable[Mapping[str, object]]
) -> set[str]:
    """Return normalised review texts present in both splits.

    Aspect-level records share a review, so overlap is judged on the review text.
    """
    train_reviews = {normalise_review(str(record["review"])) for record in train}
    test_reviews = {normalise_review(str(record["review"])) for record in test}
    return train_reviews & test_reviews


def assert_no_leakage(
    train: Iterable[Mapping[str, object]], test: Iterable[Mapping[str, object]]
) -> None:
    """Raise ``ValueError`` if any review appears in both the train and test splits."""
    overlap = find_overlap(train, test)
    if overlap:
        sample = sorted(overlap)[:3]
        msg = f"{len(overlap)} review(s) appear in both train and test, e.g. {sample}"
        raise ValueError(msg)
