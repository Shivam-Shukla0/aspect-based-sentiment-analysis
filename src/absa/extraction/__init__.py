"""Aspect extraction, sequence tagging and prediction contracts."""

from absa.extraction.prompts import (
    FEW_SHOT_EXAMPLES,
    PROMPT_VERSION,
    SYSTEM_INSTRUCTION,
    ZERO_SHOT_TEMPLATE,
    format_few_shot_prompt,
    format_zero_shot_prompt,
)

__all__ = [
    "FEW_SHOT_EXAMPLES",
    "PROMPT_VERSION",
    "SYSTEM_INSTRUCTION",
    "ZERO_SHOT_TEMPLATE",
    "format_few_shot_prompt",
    "format_zero_shot_prompt",
]
