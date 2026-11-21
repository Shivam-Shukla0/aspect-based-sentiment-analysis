"""BERT sentence-pair example construction and label mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from absa.config import SENTIMENT_LABELS
from absa.data.schema import validate_record

SENTIMENT_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
ID_TO_SENTIMENT: dict[int, str] = dict(enumerate(SENTIMENT_LABELS))


@dataclass(frozen=True)
class BertPairExample:
    """A single sentence-pair example for BERT aspect sentiment classification."""

    example_id: str
    text_a: str
    text_b: str
    label: int
    sentiment: str

    def to_dict(self) -> dict[str, object]:
        """Return example attributes as a dictionary."""
        return {
            "example_id": self.example_id,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "label": self.label,
            "sentiment": self.sentiment,
        }


def _label_ids(labels: Sequence[str]) -> dict[str, int]:
    """Map each label to its position in ``labels``."""
    return {label: idx for idx, label in enumerate(labels)}


def _sentence_id(record: Mapping[str, object]) -> str | None:
    """Return the record's sentence id, preferring ``sentence_id`` over ``id``."""
    for key in ("sentence_id", "id"):
        value = record.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return None


def build_pair_example(
    record: Mapping[str, object],
    example_id: str | None = None,
    labels: Sequence[str] = SENTIMENT_LABELS,
) -> BertPairExample:
    """Build a BertPairExample from an aspect record dictionary.

    Args:
        record: A dictionary with 'review', 'aspect' and 'sentiment', and optionally
            'sentence_id' (or 'id').
        example_id: Identifier for the example. Defaults to the record's sentence id,
            or '0' if it has none. Use ``build_pair_dataset`` to get ids that are unique
            per aspect.
        labels: Allowed sentiment labels. A label's id is its position in this sequence,
            so pass the same ``labels`` the model head was sized for (for example without
            'conflict').

    Raises:
        ValueError: If the record is invalid or its sentiment is not in ``labels``.
    """
    validated = validate_record(record, labels)
    sentiment = str(validated["sentiment"])
    if example_id is None:
        example_id = _sentence_id(record) or "0"
    return BertPairExample(
        example_id=example_id,
        text_a=str(validated["review"]),
        text_b=str(validated["aspect"]),
        label=_label_ids(labels)[sentiment],
        sentiment=sentiment,
    )


def build_pair_dataset(
    records: Sequence[Mapping[str, object]],
    labels: Sequence[str] = SENTIMENT_LABELS,
) -> list[BertPairExample]:
    """Convert a sequence of aspect records into BertPairExample instances.

    Each example id is ``<sentence id>_<aspect index>``, where the aspect index counts the
    records seen so far for the same sentence id, so two aspects of one sentence never
    share an id. Records without a ``sentence_id`` (or ``id``) use their position in
    ``records`` as the sentence id.
    """
    examples: list[BertPairExample] = []
    seen: dict[str, int] = {}
    for position, rec in enumerate(records):
        sentence_id = _sentence_id(rec) or str(position)
        aspect_index = seen.get(sentence_id, 0)
        seen[sentence_id] = aspect_index + 1
        examples.append(
            build_pair_example(rec, example_id=f"{sentence_id}_{aspect_index}", labels=labels)
        )
    return examples
