"""Versioned prompt templates for joint aspect-sentiment extraction."""

from __future__ import annotations

import json
from typing import Final

from absa.config import SENTIMENT_LABELS

PROMPT_VERSION: Final[str] = "v1.0"

SYSTEM_INSTRUCTION: Final[str] = (
    "You are an expert Aspect-Based Sentiment Analysis (ABSA) system. "
    "Given a customer review, extract all explicitly mentioned aspect targets "
    f"and classify the sentiment toward each aspect as one of: {', '.join(SENTIMENT_LABELS)}. "
    "Respond ONLY with a valid JSON array of objects having keys 'aspect' and 'sentiment'. "
    "If no aspect terms are mentioned, return []."
)

ZERO_SHOT_TEMPLATE: Final[str] = f"{SYSTEM_INSTRUCTION}\n\nReview: {{review}}\nOutput JSON:"

FEW_SHOT_EXAMPLES: Final[tuple[dict[str, object], ...]] = (
    {
        "review": "The battery life is exceptional, but the keyboard feels cheap.",
        "predictions": [
            {"aspect": "battery life", "sentiment": "positive"},
            {"aspect": "keyboard", "sentiment": "negative"},
        ],
    },
    {
        "review": "The pasta was delicious, though service was slow and unhelpful.",
        "predictions": [
            {"aspect": "pasta", "sentiment": "positive"},
            {"aspect": "service", "sentiment": "negative"},
        ],
    },
    {
        "review": "It runs fine for everyday tasks.",
        "predictions": [
            {"aspect": "everyday tasks", "sentiment": "neutral"},
        ],
    },
    {
        "review": "The food was tasty but overpriced, creating mixed impressions.",
        "predictions": [
            {"aspect": "food", "sentiment": "conflict"},
        ],
    },
)


def format_zero_shot_prompt(review: str) -> str:
    """Generate a formatted zero-shot prompt for a given review."""
    return ZERO_SHOT_TEMPLATE.format(review=review)


def format_few_shot_prompt(review: str) -> str:
    """Generate a formatted few-shot prompt with diverse training examples."""
    examples_str = ""
    for ex in FEW_SHOT_EXAMPLES:
        rev = str(ex["review"])
        preds = json.dumps(ex["predictions"])
        examples_str += f"Review: {rev}\nOutput JSON: {preds}\n\n"

    return f"{SYSTEM_INSTRUCTION}\n\n{examples_str}Review: {review}\nOutput JSON:"
