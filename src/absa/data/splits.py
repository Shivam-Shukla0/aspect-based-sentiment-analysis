"""Reproducible train/validation splitting, grouped by review text.

The official SemEval test split is never touched here: this only carves a
validation set out of the training records, since the official train/test
separation must be preserved (see ``data/README.md``).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TypeVar

from absa.config import RANDOM_SEED
from absa.data.leakage import assert_no_leakage, normalise_review

RecordT = TypeVar("RecordT", bound=Mapping[str, object])


def split_train_val(
    records: Sequence[RecordT], val_fraction: float = 0.1, seed: int = RANDOM_SEED
) -> tuple[list[RecordT], list[RecordT]]:
    """Split ``records`` into ``(train, val)``, grouped by normalised review text.

    Aspect-level records from the same sentence share its review text, so
    splitting at the record level would leak that text across the split.
    Grouping by the normalised text (not only ``sentence_id``) also keeps
    different sentence ids that carry identical text together, which
    SemEval contains. The split is deterministic for a given ``seed``, and
    is checked with ``absa.data.leakage.assert_no_leakage`` before being returned.
    """
    # No-leakage design: the validation split (20% of training in the experiments) groups records
    # by normalised review text, so two aspects of the same sentence, or duplicate sentences,
    # never fall on both sides. assert_no_leakage below raises if any review text is shared.
    # The official test set is used only once, after all choices are fixed.
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be strictly between 0 and 1, got {val_fraction}")
    # Unit of splitting = unique normalised review text, never an individual record.
    # sorted() first makes the order independent of set iteration, so the seeded
    # shuffle below is reproducible across runs and platforms.
    groups = sorted({normalise_review(str(record["review"])) for record in records})
    # A private Random instance: does not disturb (or depend on) the global RNG state.
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(groups)
    # max(1, ...) guarantees a non-empty validation set even for very small inputs.
    cut = max(1, round(len(groups) * val_fraction))
    val_groups = set(groups[:cut])
    train = [r for r in records if normalise_review(str(r["review"])) not in val_groups]
    val = [r for r in records if normalise_review(str(r["review"])) in val_groups]
    # Safety net: proves the grouping worked before the split is handed to callers.
    assert_no_leakage(train, val)
    return train, val
