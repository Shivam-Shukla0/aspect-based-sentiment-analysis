"""BIO sequence tagging and dataset preparation for aspect extraction."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

BIO_TAGS: tuple[str, ...] = ("O", "B-ASP", "I-ASP")
TAG_TO_ID: dict[str, int] = {tag: idx for idx, tag in enumerate(BIO_TAGS)}
ID_TO_TAG: dict[int, str] = dict(enumerate(BIO_TAGS))

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class TokenSpan:
    """A token and its character offsets in the source text."""

    token: str
    start: int
    end: int


@dataclass(frozen=True)
class BioExample:
    """Sequence tagging example with aligned tokens and BIO labels."""

    example_id: str
    tokens: tuple[str, ...]
    tags: tuple[str, ...]
    tag_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Return attributes as a dictionary."""
        return {
            "example_id": self.example_id,
            "tokens": list(self.tokens),
            "tags": list(self.tags),
            "tag_ids": list(self.tag_ids),
        }


def tokenize_with_offsets(text: str) -> list[TokenSpan]:
    """Tokenize text into words and punctuation while tracking character offsets."""
    return [
        TokenSpan(token=m.group(), start=m.start(), end=m.end()) for m in _TOKEN_RE.finditer(text)
    ]


def align_bio_tags(
    tokens: Sequence[TokenSpan],
    aspect_spans: Sequence[tuple[int, int]],
) -> list[str]:
    """Assign BIO tags ('O', 'B-ASP', 'I-ASP') to tokens matching character spans.

    Args:
        tokens: Sequence of TokenSpan with start and end offsets.
        aspect_spans: Sequence of (start_char, end_char) aspect boundaries.

    Returns:
        List of BIO tags matching the length of tokens.
    """
    tags = ["O"] * len(tokens)
    sorted_spans = sorted(aspect_spans, key=lambda s: (s[0], s[1]))

    for span_start, span_end in sorted_spans:
        first_token = True
        for idx, tok in enumerate(tokens):
            if tok.end > span_start and tok.start < span_end:
                if first_token:
                    tags[idx] = "B-ASP"
                    first_token = False
                elif tags[idx] == "O":
                    tags[idx] = "I-ASP"
    return tags


def build_bio_example(
    text: str,
    aspect_spans: Sequence[tuple[int, int]],
    example_id: str = "0",
) -> BioExample:
    """Build a BioExample from review text and character-level aspect spans."""
    token_spans = tokenize_with_offsets(text)
    tags = align_bio_tags(token_spans, aspect_spans)
    tag_ids = [TAG_TO_ID[t] for t in tags]
    return BioExample(
        example_id=example_id,
        tokens=tuple(t.token for t in token_spans),
        tags=tuple(tags),
        tag_ids=tuple(tag_ids),
    )
