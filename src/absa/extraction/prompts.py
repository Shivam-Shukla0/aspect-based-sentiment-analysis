"""Versioned prompt templates for joint aspect-sentiment extraction."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Final

from absa.config import SENTIMENT_LABELS

PROMPT_VERSION: Final[str] = "v1.0"

PROMPT_DIR = files("absa.extraction").joinpath("prompt_files").joinpath(PROMPT_VERSION)


def load_prompt_file(name: str) -> str:
    """Return the text of ``name`` from the prompt files of the current version.

    One trailing newline is dropped so that editors may end the file with a newline.
    """
    text = PROMPT_DIR.joinpath(name).read_text(encoding="utf-8")
    return text.removesuffix("\n")


SYSTEM_INSTRUCTION: Final[str] = load_prompt_file("system.txt").format(
    labels=", ".join(SENTIMENT_LABELS)
)

ZERO_SHOT_TEMPLATE: Final[str] = load_prompt_file("zero_shot.txt").replace(
    "{system}", SYSTEM_INSTRUCTION
)


def _quote(review: str) -> str:
    """Return ``review`` as a JSON string so that its content cannot look like prompt text."""
    return json.dumps(review, ensure_ascii=False)


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
        "review": "It comes with a backlit keyboard and two USB ports.",
        "predictions": [
            {"aspect": "backlit keyboard", "sentiment": "neutral"},
            {"aspect": "USB ports", "sentiment": "neutral"},
        ],
    },
    {
        "review": "The staff were friendly at first but turned rude later.",
        "predictions": [
            {"aspect": "staff", "sentiment": "conflict"},
        ],
    },
)


def format_zero_shot_prompt(review: str) -> str:
    """Generate a formatted zero-shot prompt for a given review."""
    return ZERO_SHOT_TEMPLATE.format(review=_quote(review))


def format_few_shot_prompt(review: str) -> str:
    """Generate a formatted few-shot prompt with hand-written example reviews."""
    examples_str = ""
    for ex in FEW_SHOT_EXAMPLES:
        rev = str(ex["review"])
        preds = json.dumps(ex["predictions"])
        examples_str += f"Review: {_quote(rev)}\nOutput JSON: {preds}\n\n"

    return f"{SYSTEM_INSTRUCTION}\n\n{examples_str}Review: {_quote(review)}\nOutput JSON:"
