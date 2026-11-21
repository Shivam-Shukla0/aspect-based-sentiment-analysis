"""Aspect extraction: BIO sequence tagging and structured language-model prompts."""

from absa.extraction.bio import (
    BIO_TAGS,
    ID_TO_TAG,
    TAG_TO_ID,
    BioExample,
    TokenSpan,
    align_bio_tags,
    build_bio_example,
    tokenize_with_offsets,
)
from absa.extraction.decoding import (
    Sentence,
    decode_spans,
    group_sentences,
    repair_tags,
    split_sentences,
)
from absa.extraction.prompts import (
    FEW_SHOT_EXAMPLES,
    PROMPT_VERSION,
    SYSTEM_INSTRUCTION,
    ZERO_SHOT_TEMPLATE,
    format_few_shot_prompt,
    format_zero_shot_prompt,
)

__all__ = [
    "BIO_TAGS",
    "FEW_SHOT_EXAMPLES",
    "ID_TO_TAG",
    "PROMPT_VERSION",
    "SYSTEM_INSTRUCTION",
    "TAG_TO_ID",
    "ZERO_SHOT_TEMPLATE",
    "BioExample",
    "Sentence",
    "TokenSpan",
    "align_bio_tags",
    "build_bio_example",
    "decode_spans",
    "format_few_shot_prompt",
    "format_zero_shot_prompt",
    "group_sentences",
    "repair_tags",
    "split_sentences",
    "tokenize_with_offsets",
]
