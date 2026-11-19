"""Tests for BIO-tagged aspect extraction dataset preparation."""

from __future__ import annotations

from absa.extraction.bio import (
    BIO_TAGS,
    ID_TO_TAG,
    TAG_TO_ID,
    align_bio_tags,
    build_bio_example,
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
