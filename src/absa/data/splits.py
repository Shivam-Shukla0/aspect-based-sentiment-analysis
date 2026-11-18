"""Reproducible train/validation splitting, grouped by sentence id.

The official SemEval test split is never touched here: this only carves a
validation set out of the training records, since the official train/test
separation must be preserved (see ``data/README.md``).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TypeVar

from absa.config import RANDOM_SEED
from absa.data.leakage import assert_no_leakage

RecordT = TypeVar("RecordT", bound=Mapping[str, object])


def split_train_val(
    records: Sequence[RecordT], val_fraction: float = 0.1, seed: int = RANDOM_SEED
) -> tuple[list[RecordT], list[RecordT]]:
    """Split ``records`` into ``(train, val)``, grouped by ``sentence_id``.

    Aspect-level records from the same sentence share its review text,
    so splitting at the record level would leak that text across the
    split; grouping by ``sentence_id`` avoids this. The split is
    deterministic for a given ``seed``, and is checked with
    ``absa.data.leakage.assert_no_leakage`` before being returned.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be strictly between 0 and 1, got {val_fraction}")
    sentence_ids = sorted({str(record["sentence_id"]) for record in records})
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(sentence_ids)
    cut = max(1, round(len(sentence_ids) * val_fraction))
    val_ids = set(sentence_ids[:cut])
    train = [r for r in records if str(r["sentence_id"]) not in val_ids]
    val = [r for r in records if str(r["sentence_id"]) in val_ids]
    assert_no_leakage(train, val)
    return train, val
