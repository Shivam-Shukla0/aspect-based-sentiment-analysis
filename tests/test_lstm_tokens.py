from pathlib import Path

import pytest

from absa.models import PAD_TOKEN, UNK_TOKEN, Vocabulary, pad_or_truncate, tokenize


def test_tokenize_lowercases_and_splits_punctuation() -> None:
    assert tokenize("Great value!") == ["great", "value", "!"]


def test_tokenize_handles_multi_word_aspect() -> None:
    assert tokenize("battery life") == ["battery", "life"]


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


def test_build_orders_by_frequency_then_alphabetically() -> None:
    vocab = Vocabulary.build(["a a b", "a b c"])
    assert vocab.tokens == (PAD_TOKEN, UNK_TOKEN, "a", "b", "c")


def test_build_respects_min_freq() -> None:
    vocab = Vocabulary.build(["a a b"], min_freq=2)
    assert vocab.tokens == (PAD_TOKEN, UNK_TOKEN, "a")


def test_build_uses_only_given_texts() -> None:
    train_vocab = Vocabulary.build(["only training words"])
    assert "unseen" not in train_vocab.tokens


def test_unknown_token_maps_to_unk_id() -> None:
    vocab = Vocabulary.build(["known word"])
    ids = vocab.encode(["known", "mystery"])
    assert ids[0] != vocab.unk_id
    assert ids[1] == vocab.unk_id


def test_encode_decode_round_trip() -> None:
    vocab = Vocabulary.build(["hello world"])
    tokens = ["hello", "world"]
    assert vocab.decode(vocab.encode(tokens)) == tokens


def test_pad_id_and_unk_id_are_reserved() -> None:
    vocab = Vocabulary.build(["x"])
    assert vocab.pad_id == 0
    assert vocab.unk_id == 1
    assert vocab.tokens[0] == PAD_TOKEN
    assert vocab.tokens[1] == UNK_TOKEN


def test_len_includes_reserved_tokens() -> None:
    vocab = Vocabulary.build(["a b c"])
    assert len(vocab) == 5


def test_vocabulary_rejects_tokens_without_reserved_prefix() -> None:
    with pytest.raises(ValueError, match="must start with"):
        Vocabulary(["word"])


def test_save_load_round_trip_is_identical(tmp_path: Path) -> None:
    vocab = Vocabulary.build(["the battery is weak the screen is great"])
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = Vocabulary.load(path)
    assert loaded.tokens == vocab.tokens


@pytest.mark.parametrize(
    ("ids", "length", "expected"),
    [
        ([1, 2, 3], 5, [1, 2, 3, 0, 0]),
        ([1, 2, 3], 3, [1, 2, 3]),
        ([1, 2, 3, 4], 2, [1, 2]),
    ],
)
def test_pad_or_truncate(ids: list[int], length: int, expected: list[int]) -> None:
    assert pad_or_truncate(ids, length, pad_id=0) == expected


def test_pad_or_truncate_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="length must be positive"):
        pad_or_truncate([1, 2], 0, pad_id=0)
