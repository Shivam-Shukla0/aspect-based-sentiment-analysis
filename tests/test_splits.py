import pytest

from absa.data import split_train_val
from absa.data.leakage import find_overlap


def _records(n_sentences: int, aspects_per_sentence: int = 2) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for sentence_index in range(n_sentences):
        review = f"Sentence number {sentence_index} with some words."
        records.extend(
            {
                "review": review,
                "aspect": f"aspect{aspect_index}",
                "sentiment": "positive",
                "sentence_id": str(sentence_index),
            }
            for aspect_index in range(aspects_per_sentence)
        )
    return records


def test_no_sentence_appears_in_both_splits() -> None:
    records = _records(50)
    train, val = split_train_val(records, val_fraction=0.2, seed=42)
    assert find_overlap(train, val) == set()


def test_split_is_deterministic_for_a_given_seed() -> None:
    records = _records(50)
    train_a, val_a = split_train_val(records, val_fraction=0.2, seed=7)
    train_b, val_b = split_train_val(records, val_fraction=0.2, seed=7)
    assert train_a == train_b
    assert val_a == val_b


def test_different_seeds_can_give_different_splits() -> None:
    records = _records(50)
    _, val_a = split_train_val(records, val_fraction=0.2, seed=1)
    _, val_b = split_train_val(records, val_fraction=0.2, seed=2)
    ids_a = {r["sentence_id"] for r in val_a}
    ids_b = {r["sentence_id"] for r in val_b}
    assert ids_a != ids_b


def test_all_records_from_one_sentence_stay_together() -> None:
    records = _records(20, aspects_per_sentence=3)
    train, val = split_train_val(records, val_fraction=0.3, seed=42)
    train_ids = {r["sentence_id"] for r in train}
    val_ids = {r["sentence_id"] for r in val}
    assert train_ids.isdisjoint(val_ids)
    for sentence_id in val_ids:
        count = sum(1 for r in val if r["sentence_id"] == sentence_id)
        assert count == 3


def test_every_record_is_placed_in_exactly_one_split() -> None:
    records = _records(30)
    train, val = split_train_val(records, val_fraction=0.25, seed=42)
    assert len(train) + len(val) == len(records)


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_val_fraction_rejected(bad_fraction: float) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        split_train_val(_records(10), val_fraction=bad_fraction)


def test_sentences_with_identical_text_stay_on_one_side() -> None:
    records: list[dict[str, object]] = []
    for index in range(30):
        review = "Great screen." if index < 2 else f"Unique sentence {index}."
        records.append(
            {"review": review, "aspect": "screen", "sentiment": "positive", "sentence_id": str(index)}
        )
    for seed in range(20):
        train, val = split_train_val(records, val_fraction=0.2, seed=seed)
        assert find_overlap(train, val) == set()
        assert len(train) + len(val) == len(records)
