"""Tests for BIO-tagged aspect extraction dataset preparation."""

from __future__ import annotations

import pytest

from absa.extraction.bio import (
    BIO_TAGS,
    ID_TO_TAG,
    TAG_TO_ID,
    align_bio_tags,
    build_bio_example,
    spans_from_tags,
    tokenize_with_offsets,
)


def test_bio_tag_vocabulary() -> None:
    assert BIO_TAGS == ("O", "B-ASP", "I-ASP")
    assert TAG_TO_ID["O"] == 0
    assert TAG_TO_ID["B-ASP"] == 1
    assert TAG_TO_ID["I-ASP"] == 2
    assert ID_TO_TAG[1] == "B-ASP"


def test_tokenize_with_offsets() -> None:
    text = "Great battery life!"
    tokens = tokenize_with_offsets(text)
    assert [t.token for t in tokens] == ["Great", "battery", "life", "!"]
    assert tokens[1].start == 6
    assert tokens[1].end == 13
    assert text[tokens[1].start : tokens[1].end] == "battery"


def test_align_bio_tags_multi_word() -> None:
    text = "The battery life is exceptional."
    tokens = tokenize_with_offsets(text)
    # "battery life" is from 4 to 16
    spans = [(4, 16)]
    tags = align_bio_tags(tokens, spans)
    expected = ["O", "B-ASP", "I-ASP", "O", "O", "O"]
    assert tags == expected


def test_build_bio_example_multi_aspect() -> None:
    text = "The food was delicious, but service was slow."
    # "food" -> (4, 8), "service" -> (28, 35)
    spans = [(4, 8), (28, 35)]
    example = build_bio_example(text, spans, example_id="review_1")

    assert example.example_id == "review_1"
    assert "B-ASP" in example.tags
    food_idx = example.tokens.index("food")
    service_idx = example.tokens.index("service")
    assert example.tags[food_idx] == "B-ASP"
    assert example.tags[service_idx] == "B-ASP"
    assert example.tag_ids[food_idx] == TAG_TO_ID["B-ASP"]


def test_build_bio_example_no_aspects() -> None:
    text = "Everything was perfectly fine."
    example = build_bio_example(text, aspect_spans=[])
    assert all(t == "O" for t in example.tags)
    assert all(tid == 0 for tid in example.tag_ids)


@pytest.mark.parametrize(
    ("text", "span"),
    [("abc", (2, 1)), ("abc", (1, 1)), ("abc", (-1, 2)), ("abc", (5, 9))],
)
def test_align_rejects_invalid_or_out_of_range_spans(text: str, span: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="aspect span"):
        align_bio_tags(tokenize_with_offsets(text), [span])


def test_align_rejects_overlapping_spans() -> None:
    tokens = tokenize_with_offsets("hard drive fast")
    with pytest.raises(ValueError, match="overlaps an earlier span"):
        align_bio_tags(tokens, [(0, 10), (5, 10)])


def test_align_tags_whole_token_when_span_cuts_through_it() -> None:
    tokens = tokenize_with_offsets("OSX works")
    assert align_bio_tags(tokens, [(0, 2)]) == ["B-ASP", "O"]


@pytest.mark.parametrize(
    ("text", "spans"),
    [
        ("hard drive fast", [(0, 7), (8, 15)]),
        ("OSX", [(0, 1), (1, 2)]),
    ],
)
def test_align_rejects_spans_that_reach_the_same_token(
    text: str, spans: list[tuple[int, int]]
) -> None:
    with pytest.raises(ValueError, match="already tagged"):
        align_bio_tags(tokenize_with_offsets(text), spans)


@pytest.mark.parametrize(
    ("text", "spans"),
    [
        ("The battery life is exceptional.", [(4, 16)]),
        ("The food was delicious, but service was slow.", [(4, 8), (28, 35)]),
        ("hard drive fast", [(0, 4), (5, 10)]),
        ("hard drive fast", [(0, 4), (5, 10), (11, 15)]),
        ("great, fast", []),
    ],
)
def test_round_trip_is_exact_for_whole_token_spans(text: str, spans: list[tuple[int, int]]) -> None:
    tokens = tokenize_with_offsets(text)
    assert spans_from_tags(tokens, align_bio_tags(tokens, spans)) == spans


def test_round_trip_keeps_adjacent_spans_apart() -> None:
    tokens = tokenize_with_offsets("hard drive")
    tags = align_bio_tags(tokens, [(0, 4), (5, 10)])
    assert tags == ["B-ASP", "B-ASP"]
    assert spans_from_tags(tokens, tags) == [(0, 4), (5, 10)]


def test_spans_from_tags_widens_to_the_whole_token() -> None:
    tokens = tokenize_with_offsets("OSX works")
    assert spans_from_tags(tokens, align_bio_tags(tokens, [(0, 2)])) == [(0, 3)]


@pytest.mark.parametrize(
    ("tags", "message"),
    [
        (["B-ASP"], "tags for 2 tokens"),
        (["B-ASP", "X"], "unknown BIO tag"),
        (["O", "I-ASP"], "does not follow"),
    ],
)
def test_spans_from_tags_rejects_bad_tags(tags: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        spans_from_tags(tokenize_with_offsets("hard drive"), tags)
