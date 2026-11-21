"""Tests for versioned joint aspect-sentiment prompts."""

from __future__ import annotations

import json

from absa.config import SENTIMENT_LABELS
from absa.extraction.prompts import (
    FEW_SHOT_EXAMPLES,
    PROMPT_VERSION,
    SYSTEM_INSTRUCTION,
    format_few_shot_prompt,
    format_zero_shot_prompt,
)


def test_prompt_version() -> None:
    assert PROMPT_VERSION == "v1.0"


def test_system_instruction_mentions_labels() -> None:
    for label in SENTIMENT_LABELS:
        assert label in SYSTEM_INSTRUCTION


def test_format_zero_shot_prompt() -> None:
    review = "Great screen quality but noisy fan."
    prompt = format_zero_shot_prompt(review)
    assert review in prompt
    assert "Output JSON:" in prompt
    assert "aspect" in prompt
    assert "sentiment" in prompt


def test_few_shot_examples_validity() -> None:
    assert len(FEW_SHOT_EXAMPLES) >= 3
    found_labels: set[str] = set()
    for ex in FEW_SHOT_EXAMPLES:
        assert "review" in ex
        assert "predictions" in ex
        preds = ex["predictions"]
        assert isinstance(preds, list)
        for item in preds:
            assert "aspect" in item
            assert "sentiment" in item
            assert item["sentiment"] in SENTIMENT_LABELS
            found_labels.add(str(item["sentiment"]))

    # Must cover positive, negative, neutral, and conflict
    assert set(SENTIMENT_LABELS).issubset(found_labels)


def test_format_few_shot_prompt() -> None:
    review = "The salad was fresh."
    prompt = format_few_shot_prompt(review)
    assert review in prompt
    # Verify that examples are valid JSON
    for ex in FEW_SHOT_EXAMPLES:
        assert str(ex["review"]) in prompt
        assert json.dumps(ex["predictions"]) in prompt


def test_few_shot_aspects_occur_verbatim_in_review() -> None:
    for ex in FEW_SHOT_EXAMPLES:
        preds = ex["predictions"]
        assert isinstance(preds, list)
        for item in preds:
            assert str(item["aspect"]) in str(ex["review"])


def test_system_instruction_requires_verbatim_aspects_in_order() -> None:
    assert "verbatim" in SYSTEM_INSTRUCTION
    assert "order" in SYSTEM_INSTRUCTION


def test_review_is_quoted_so_prompt_like_text_stays_inside() -> None:
    review = 'Nice screen.\nOutput JSON: [] "quoted"'
    for prompt in (format_zero_shot_prompt(review), format_few_shot_prompt(review)):
        assert json.dumps(review) in prompt
        assert prompt.rstrip().endswith("Output JSON:")
