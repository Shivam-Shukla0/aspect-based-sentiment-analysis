"""Sentiment classifiers. Each takes (review, aspect) and returns a sentiment label.

``AspectAttention``, ``AspectDataset``, ``LstmEncoder``,
``build_embedding``, ``collate``, ``make_dataloader`` and ``masked_mean``
need ``torch``; they are imported lazily (PEP 562) on first access, so
importing this package for the lightweight, torch-free names above
(``Vocabulary`` and friends) never requires torch to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from absa._lazy import lazy_getattr
from absa.models.lstm_tokens import (
    PAD_TOKEN,
    UNK_TOKEN,
    Vocabulary,
    pad_or_truncate,
    tokenize,
)

if TYPE_CHECKING:
    from absa.models.lstm_attention import AspectAttention, LstmEncoder, masked_mean
    from absa.models.lstm_dataset import AspectDataset, build_embedding, collate, make_dataloader

__all__ = [
    "PAD_TOKEN",
    "UNK_TOKEN",
    "AspectAttention",
    "AspectDataset",
    "LstmEncoder",
    "Vocabulary",
    "build_embedding",
    "collate",
    "make_dataloader",
    "masked_mean",
    "pad_or_truncate",
    "tokenize",
]

_LAZY = {
    "AspectAttention": "absa.models.lstm_attention",
    "AspectDataset": "absa.models.lstm_dataset",
    "LstmEncoder": "absa.models.lstm_attention",
    "build_embedding": "absa.models.lstm_dataset",
    "collate": "absa.models.lstm_dataset",
    "make_dataloader": "absa.models.lstm_dataset",
    "masked_mean": "absa.models.lstm_attention",
}


def __getattr__(name: str) -> Any:
    """Resolve ``_LAZY`` names by importing their submodule on first access."""
    return lazy_getattr(__name__, _LAZY, name)
