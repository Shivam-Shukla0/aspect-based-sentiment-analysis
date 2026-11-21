import pytest

pytest.importorskip("torch")
import torch

from absa.config import SENTIMENT_LABELS
from absa.models import AspectDataset, Vocabulary, build_embedding, collate, make_dataloader

RECORDS = [
    {"review": "The battery is weak.", "aspect": "battery", "sentiment": "negative"},
    {"review": "Great screen.", "aspect": "screen", "sentiment": "positive"},
    {"review": "Fast boot time and great speed.", "aspect": "boot time", "sentiment": "positive"},
]


def _vocab() -> Vocabulary:
    return Vocabulary.build([r["review"] for r in RECORDS])


def test_dataset_length() -> None:
    dataset = AspectDataset(RECORDS, _vocab())
    assert len(dataset) == len(RECORDS)


def test_dataset_item_has_expected_keys_and_dtypes() -> None:
    dataset = AspectDataset(RECORDS, _vocab())
    item = dataset[0]
    assert set(item) == {"review_ids", "aspect_ids", "label"}
    assert item["review_ids"].dtype == torch.long
    assert item["label"].dim() == 0


def test_dataset_label_matches_sentiment_index() -> None:
    dataset = AspectDataset(RECORDS, _vocab())
    item = dataset[0]
    assert item["label"].item() == SENTIMENT_LABELS.index("negative")


def test_collate_pads_to_batch_max_length() -> None:
    vocab = _vocab()
    dataset = AspectDataset(RECORDS, vocab)
    batch = [dataset[i] for i in range(len(dataset))]
    result = collate(batch, pad_id=vocab.pad_id)
    assert result["review_ids"].shape[0] == len(RECORDS)
    assert result["review_ids"].shape[1] == result["review_lengths"].max().item()
    assert result["aspect_ids"].shape[1] == result["aspect_lengths"].max().item()


def test_collate_lengths_match_unpadded_sizes() -> None:
    vocab = _vocab()
    dataset = AspectDataset(RECORDS, vocab)
    batch = [dataset[i] for i in range(len(dataset))]
    result = collate(batch, pad_id=vocab.pad_id)
    for i, item in enumerate(batch):
        assert result["review_lengths"][i].item() == len(item["review_ids"])
        assert result["aspect_lengths"][i].item() == len(item["aspect_ids"])


def test_collate_padding_uses_pad_id() -> None:
    vocab = _vocab()
    dataset = AspectDataset(RECORDS, vocab)
    batch = [dataset[i] for i in range(len(dataset))]
    result = collate(batch, pad_id=vocab.pad_id)
    for i in range(len(batch)):
        length = int(result["review_lengths"][i].item())
        padding = result["review_ids"][i, length:]
        assert bool((padding == vocab.pad_id).all())


def test_collate_preserves_labels() -> None:
    vocab = _vocab()
    dataset = AspectDataset(RECORDS, vocab)
    batch = [dataset[i] for i in range(len(dataset))]
    result = collate(batch, pad_id=vocab.pad_id)
    assert result["label"].tolist() == [item["label"].item() for item in batch]


def test_dataloader_batches_have_consistent_shapes() -> None:
    vocab = _vocab()
    dataset = AspectDataset(RECORDS, vocab)
    loader = make_dataloader(dataset, pad_id=vocab.pad_id, batch_size=2)
    batches = list(loader)
    assert batches[0]["review_ids"].shape[0] == 2
    assert batches[1]["review_ids"].shape[0] == 1


def test_dataloader_shuffle_is_deterministic_for_a_given_seed() -> None:
    vocab = _vocab()
    dataset = AspectDataset(RECORDS, vocab)
    loader_a = make_dataloader(dataset, pad_id=vocab.pad_id, batch_size=1, shuffle=True, seed=7)
    loader_b = make_dataloader(dataset, pad_id=vocab.pad_id, batch_size=1, shuffle=True, seed=7)
    order_a = [batch["label"].item() for batch in loader_a]
    order_b = [batch["label"].item() for batch in loader_b]
    assert order_a == order_b


def test_build_embedding_shape_and_padding_idx() -> None:
    vocab = _vocab()
    embedding = build_embedding(len(vocab), embedding_dim=16, pad_id=vocab.pad_id)
    assert embedding.weight.shape == (len(vocab), 16)
    assert embedding.padding_idx == vocab.pad_id
    assert torch.equal(embedding.weight[vocab.pad_id], torch.zeros(16))
