"""Sentences, grouping, splitting and tag decoding for the aspect extractor.

Everything here is plain Python so it can be tested without a GPU or a model: the
extractor itself lives in ``absa.models.bert_extractor`` and only produces tags.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from absa.extraction.bio import spans_from_tags, tokenize_with_offsets

Span = tuple[int, int]


@dataclass(frozen=True)
class Sentence:
    """A review sentence with its gold aspect spans and their sentiments."""

    sentence_id: str
    text: str
    spans: tuple[tuple[int, int, str], ...] = ()

    @property
    def gold_spans(self) -> list[Span]:
        """Return the gold character spans in text order."""
        return sorted((start, end) for start, end, _ in self.spans)


def group_sentences(
    records: Sequence[Mapping[str, object]], no_aspect: Sequence[Mapping[str, object]] = ()
) -> list[Sentence]:
    """Group aspect records by sentence and add the sentences without any aspect.

    Records need ``review``, ``sentence_id``, ``start``, ``end`` and ``sentiment``; sentences
    keep the order in which they first appear.
    """
    grouped: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    for record in records:
        key = (str(record["sentence_id"]), str(record["review"]))
        grouped.setdefault(key, []).append(
            (int(str(record["start"])), int(str(record["end"])), str(record["sentiment"]))
        )
    sentences = [Sentence(sid, text, tuple(spans)) for (sid, text), spans in grouped.items()]
    sentences.extend(Sentence(str(r["sentence_id"]), str(r["review"])) for r in no_aspect)
    return sentences


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def split_sentences(
    sentences: Sequence[Sentence], val_fraction: float, seed: int
) -> tuple[list[Sentence], list[Sentence]]:
    """Split into train and validation sentences, keeping identical texts together.

    Texts are compared after lower-casing and collapsing whitespace, so a repeated review
    never lands on both sides. The split is deterministic for a given seed.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    texts = sorted({_normalized(s.text) for s in sentences})
    random.Random(seed).shuffle(texts)  # noqa: S311  (a reproducible split, not security)
    val_texts = set(texts[: max(1, round(len(texts) * val_fraction))])
    train = [s for s in sentences if _normalized(s.text) not in val_texts]
    val = [s for s in sentences if _normalized(s.text) in val_texts]
    if not train or not val:
        raise ValueError("the split left the train or the validation side empty")
    return train, val


def repair_tags(tags: Sequence[str]) -> list[str]:
    """Make a tag sequence valid: an ``I-ASP`` that does not continue a span starts one."""
    repaired: list[str] = []
    for tag in tags:
        starts_span = tag == "I-ASP" and (not repaired or repaired[-1] == "O")
        repaired.append("B-ASP" if starts_span else tag)
    return repaired


def decode_spans(text: str, tags: Sequence[str]) -> list[Span]:
    """Decode word-level BIO tags into character spans of ``text``.

    ``tags`` has one entry per regex token (see ``tokenize_with_offsets``). Missing trailing
    tags, for example after truncation, count as ``O``. Invalid sequences are repaired by
    ``repair_tags``, so the result is deterministic and never raises on tag order.

    Raises:
        ValueError: If there are more tags than tokens or a tag is unknown.
    """
    tokens = tokenize_with_offsets(text)
    if len(tags) > len(tokens):
        raise ValueError(f"got {len(tags)} tags for {len(tokens)} tokens")
    padded = [*tags, *["O"] * (len(tokens) - len(tags))]
    return spans_from_tags(tokens, repair_tags(padded))
