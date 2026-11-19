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


def build_pair_example(
    record: Mapping[str, object],
    example_id: str | None = None,
) -> BertPairExample:
    """Build a BertPairExample from an aspect record dictionary.

    Args:
        record: A validated dictionary with 'review', 'aspect', and 'sentiment'.
        example_id: Optional unique identifier; defaults to record['id'] or '0'.
    """
    validated = validate_record(record)
    sentiment = str(validated["sentiment"])
    label_id = SENTIMENT_TO_ID[sentiment]
    rec_id = str(example_id or record.get("id") or "0")
    return BertPairExample(
        example_id=rec_id,
        text_a=str(validated["review"]),
        text_b=str(validated["aspect"]),
        label=label_id,
        sentiment=sentiment,
    )


def build_pair_dataset(
    records: Sequence[Mapping[str, object]],
) -> list[BertPairExample]:
    """Convert a sequence of aspect records into BertPairExample instances."""
    examples: list[BertPairExample] = []
    for idx, rec in enumerate(records):
        rec_id = str(rec.get("id") or idx)
        examples.append(build_pair_example(rec, example_id=rec_id))
    return examples
