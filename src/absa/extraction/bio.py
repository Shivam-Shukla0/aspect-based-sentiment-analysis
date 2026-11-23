"""BIO sequence tagging and dataset preparation for aspect extraction."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# BIO scheme: O = outside any aspect, B-ASP = first token of an aspect, I-ASP = continuation.
# Only one entity type (ASP) is needed here, so there are just three tags.
BIO_TAGS: tuple[str, ...] = ("O", "B-ASP", "I-ASP")
TAG_TO_ID: dict[str, int] = {tag: idx for idx, tag in enumerate(BIO_TAGS)}
ID_TO_TAG: dict[int, str] = dict(enumerate(BIO_TAGS))

# A token is either a run of word characters or a single punctuation mark; whitespace is
# skipped. Simple and deterministic, but NOT the same as BERT word pieces.
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

    A token is tagged if it overlaps a span at all, so a span that cuts through a token
    (for example ``(0, 2)`` on ``"OSX works"``) tags the whole token ``OSX``. SemEval has
    such spans, and this whole-token behaviour is intentional.

    Tokens come from a regular expression, not from a BERT tokenizer. Tags are per word,
    so a BERT extractor must re-align them to word pieces (see the extractor issue #31).

    Args:
        tokens: Sequence of TokenSpan with start and end offsets.
        aspect_spans: Sequence of (start_char, end_char) aspect boundaries.

    Returns:
        List of BIO tags matching the length of tokens.

    Raises:
        ValueError: If a span is empty or inverted, overlaps no token (for example it lies
            outside the text), overlaps another span, or reaches a token that an earlier
            span already tagged (for example ``(0, 7)`` and ``(8, 15)`` on
            ``"hard drive fast"`` both cut ``drive``).
    """
    # Start with everything outside; spans then overwrite the tokens they cover.
    tags = ["O"] * len(tokens)
    claimed: set[int] = set()
    # Process spans left to right so the overlap check below only needs the previous end.
    sorted_spans = sorted(aspect_spans, key=lambda s: (s[0], s[1]))

    previous_end = 0
    for span_start, span_end in sorted_spans:
        if span_start < 0 or span_end <= span_start:
            raise ValueError(f"invalid aspect span {(span_start, span_end)}")
        if span_start < previous_end:
            raise ValueError(f"aspect span {(span_start, span_end)} overlaps an earlier span")
        previous_end = span_end
        # A token is covered if its interval intersects the span (strict inequalities because
        # ends are exclusive); this is why a span cutting a token still tags the whole token.
        covered = [i for i, t in enumerate(tokens) if t.end > span_start and t.start < span_end]
        if not covered:
            raise ValueError(f"aspect span {(span_start, span_end)} does not cover any token")
        if claimed.intersection(covered):
            raise ValueError(f"aspect span {(span_start, span_end)} reaches a token already tagged")
        claimed.update(covered)
        # First covered token begins the aspect, the rest continue it.
        tags[covered[0]] = "B-ASP"
        for idx in covered[1:]:
            tags[idx] = "I-ASP"
    return tags


def spans_from_tags(
    tokens: Sequence[TokenSpan],
    tags: Sequence[str],
) -> list[tuple[int, int]]:
    """Rebuild character spans from BIO tags and the tokens they were aligned to.

    Each span runs from the start of its ``B-ASP`` token to the end of its last
    ``I-ASP`` token. For spans that cover whole tokens this is the inverse of
    ``align_bio_tags``. A span that cut through a token comes back widened to the token.

    Args:
        tokens: Tokens with character offsets, as given to ``align_bio_tags``.
        tags: One BIO tag per token.

    Returns:
        List of (start_char, end_char) spans in text order.

    Raises:
        ValueError: If the lengths differ, a tag is unknown, or ``I-ASP`` does not
            follow a ``B-ASP`` or ``I-ASP`` tag.
    """
    if len(tokens) != len(tags):
        raise ValueError(f"got {len(tags)} tags for {len(tokens)} tokens")
    spans: list[tuple[int, int]] = []
    previous = "O"
    for token, tag in zip(tokens, tags, strict=True):
        if tag not in TAG_TO_ID:
            raise ValueError(f"unknown BIO tag {tag!r}")
        if tag == "B-ASP":
            spans.append((token.start, token.end))  # open a new span at this token
        elif tag == "I-ASP":
            # I-ASP is only legal directly after B-ASP/I-ASP; otherwise there is no span to extend.
            if previous == "O":
                raise ValueError("I-ASP tag does not follow B-ASP or I-ASP")
            # Extend the span opened last: keep its start, move its end to this token.
            spans[-1] = (spans[-1][0], token.end)
        previous = tag
    return spans


def build_bio_example(
    text: str,
    aspect_spans: Sequence[tuple[int, int]],
    example_id: str = "0",
) -> BioExample:
    """Build a BioExample from review text and character-level aspect spans.

    The tags are on regex tokens (words and punctuation), not on BERT word pieces. The
    aspect extractor (issue #31) must re-align them to word pieces before training.

    Raises:
        ValueError: If a span ends after the end of ``text`` or is otherwise invalid.
    """
    # Validate against the raw text first: align_bio_tags only sees tokens and could not
    # tell that a span runs past the end of the sentence.
    for start, end in aspect_spans:
        if end > len(text):
            raise ValueError(f"aspect span ({start}, {end}) ends after the text ({len(text)})")
    token_spans = tokenize_with_offsets(text)
    tags = align_bio_tags(token_spans, aspect_spans)
    # Integer ids are what a model's loss function consumes; strings are for humans.
    tag_ids = [TAG_TO_ID[t] for t in tags]
    return BioExample(
        example_id=example_id,
        tokens=tuple(t.token for t in token_spans),
        tags=tuple(tags),
        tag_ids=tuple(tag_ids),
    )
