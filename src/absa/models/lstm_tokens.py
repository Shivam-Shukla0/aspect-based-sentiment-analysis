"""Tokenizer and vocabulary for the LSTM model.

Tokenization lowercases text and splits it into contiguous alphanumeric
runs and single punctuation characters (a simple, documented policy); the
same rule applies to reviews and to multi-word aspect terms, so both are
tokenized consistently.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

PAD_TOKEN = "<pad>"  # noqa: S105 (a vocabulary marker, not a secret)
UNK_TOKEN = "<unk>"  # noqa: S105
_TOKEN_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and split it into word and punctuation tokens.

    Example: ``"Great value!"`` becomes ``["great", "value", "!"]``.
    """
    return _TOKEN_RE.findall(text.lower())


class Vocabulary:
    """A token-to-index mapping with reserved pad and unknown tokens.

    Built from training data only (via ``build``), so the vocabulary does
    not leak information from validation or test reviews.
    """

    def __init__(self, tokens: Sequence[str]) -> None:
        """Wrap an ordered token list that must start with ``[pad, unk]``."""
        if list(tokens[:2]) != [PAD_TOKEN, UNK_TOKEN]:
            raise ValueError(f"tokens must start with [{PAD_TOKEN!r}, {UNK_TOKEN!r}]")
        self._tokens = tuple(tokens)
        self._index = {token: index for index, token in enumerate(self._tokens)}

    @property
    def pad_id(self) -> int:
        """Return the index reserved for the padding token."""
        return 0

    @property
    def unk_id(self) -> int:
        """Return the index reserved for the unknown token."""
        return 1

    @property
    def tokens(self) -> tuple[str, ...]:
        """Return every token, ordered by index."""
        return self._tokens

    def __len__(self) -> int:
        """Return the number of tokens, including the reserved pad and unknown tokens."""
        return len(self._tokens)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Map each token to its index, or to ``unk_id`` if it was never seen."""
        return [self._index.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        """Map each index back to its token."""
        return [self._tokens[index] for index in ids]

    def save(self, path: Path) -> None:
        """Write this vocabulary as JSON to ``path``, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(self._tokens), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        """Load a vocabulary previously written by ``save``."""
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> Vocabulary:
        """Build a vocabulary from training ``texts``.

        Keeps tokens seen at least ``min_freq`` times, ordered by
        descending frequency and then alphabetically, so the result is
        deterministic regardless of the input order.
        """
        counts = Counter(token for text in texts for token in tokenize(text))
        kept = sorted(
            (token for token, count in counts.items() if count >= min_freq),
            key=lambda token: (-counts[token], token),
        )
        return cls([PAD_TOKEN, UNK_TOKEN, *kept])


def pad_or_truncate(ids: Sequence[int], length: int, pad_id: int) -> list[int]:
    """Return ``ids`` truncated or right-padded with ``pad_id`` to exactly ``length``."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if len(ids) >= length:
        return list(ids[:length])
    return [*ids, *([pad_id] * (length - len(ids)))]
