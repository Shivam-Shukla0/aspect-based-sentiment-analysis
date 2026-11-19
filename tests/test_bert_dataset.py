"""Tests for BERT sentence-pair dataset preparation."""

from __future__ import annotations

import pytest

from absa.config import SENTIMENT_LABELS
from absa.data.bert_dataset import (
    ID_TO_SENTIMENT,
    SENTIMENT_TO_ID,
    BertPairExample,
    build_pair_dataset,
    build_pair_example,
)


def test_label_mapping_consistency() -> None:
    assert len(SENTIMENT_TO_ID) == len(SENTIMENT_LABELS)
    assert len(ID_TO_SENTIMENT) == len(SENTIMENT_LABELS)
    for idx, label in enumerate(SENTIMENT_LABELS):
        assert SENTIMENT_TO_ID[label] == idx
        assert ID_TO_SENTIMENT[idx] == label


def test_build_pair_example() -> None:
    record = {
        "id": "100",
        "review": "The battery life is exceptional.",
        "aspect": "battery life",
        "sentiment": "positive",
    }
    example = build_pair_example(record)
    assert isinstance(example, BertPairExample)
    assert example.example_id == "100"
    assert example.text_a == "The battery life is exceptional."
    assert example.text_b == "battery life"
    assert example.label == SENTIMENT_TO_ID["positive"]
    assert example.sentiment == "positive"
    assert example.to_dict()["label"] == 0


def test_build_pair_example_conflict_label() -> None:
    record = {
        "id": "101",
        "review": "The food was delicious but pricey.",
        "aspect": "food",
        "sentiment": "conflict",
    }
    example = build_pair_example(record)
    assert example.label == SENTIMENT_TO_ID["conflict"]
    assert example.sentiment == "conflict"


def test_build_pair_dataset_multi_aspect() -> None:
    records = [
        {
            "id": "200_0",
            "review": "Fast delivery, but the screen was cracked.",
            "aspect": "delivery",
            "sentiment": "positive",
        },
        {
            "id": "200_1",
            "review": "Fast delivery, but the screen was cracked.",
            "aspect": "screen",
            "sentiment": "negative",
        },
    ]
    dataset = build_pair_dataset(records)
    assert len(dataset) == 2
    assert dataset[0].text_b == "delivery"
    assert dataset[0].label == SENTIMENT_TO_ID["positive"]
    assert dataset[1].text_b == "screen"
    assert dataset[1].label == SENTIMENT_TO_ID["negative"]


def test_build_pair_example_invalid_record() -> None:
    with pytest.raises(ValueError, match="record is missing fields"):
        build_pair_example({"review": "Missing aspect", "sentiment": "neutral"})
