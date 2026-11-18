"""PyTorch dataset, batching and embedding layer for the LSTM classifier.

Each example is one (review, aspect) pair; both are tokenized and encoded
with the same ``Vocabulary`` (see ``absa.models.lstm_tokens``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from absa.config import RANDOM_SEED, SENTIMENT_LABELS
from absa.models.lstm_tokens import Vocabulary, tokenize


class AspectDataset(Dataset[dict[str, Tensor]]):
    """One (review, aspect) pair per example, encoded with a shared vocabulary."""

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        vocab: Vocabulary,
        labels: Sequence[str] = SENTIMENT_LABELS,
    ) -> None:
        """Wrap ``records`` (each needing ``review``, ``aspect`` and ``sentiment``)."""
        self._records = records
        self._vocab = vocab
        self._label_index = {label: index for index, label in enumerate(labels)}

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return the encoded review ids, aspect ids and label for one record."""
        record = self._records[index]
        review_ids = self._vocab.encode(tokenize(str(record["review"])))
        aspect_ids = self._vocab.encode(tokenize(str(record["aspect"])))
        label = self._label_index[str(record["sentiment"])]
        return {
            "review_ids": torch.tensor(review_ids, dtype=torch.long),
            "aspect_ids": torch.tensor(aspect_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def _pad_stack(sequences: Sequence[Tensor], pad_id: int) -> tuple[Tensor, Tensor]:
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = int(lengths.max().item()) if sequences else 0
    padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for row, seq in enumerate(sequences):
        padded[row, : len(seq)] = seq
    return padded, lengths


def collate(batch: Sequence[Mapping[str, Tensor]], pad_id: int) -> dict[str, Tensor]:
    """Pad a batch of variable-length examples to the batch's own max length.

    Returns ``review_ids``, ``review_lengths``, ``aspect_ids``,
    ``aspect_lengths`` and ``label``, each a tensor with the batch
    dimension first. Lengths are kept so padding can be masked or packed
    by the encoder.
    """
    review_ids, review_lengths = _pad_stack([item["review_ids"] for item in batch], pad_id)
    aspect_ids, aspect_lengths = _pad_stack([item["aspect_ids"] for item in batch], pad_id)
    labels = torch.stack([item["label"] for item in batch])
    return {
        "review_ids": review_ids,
        "review_lengths": review_lengths,
        "aspect_ids": aspect_ids,
        "aspect_lengths": aspect_lengths,
        "label": labels,
    }


def make_dataloader(
    dataset: AspectDataset,
    pad_id: int,
    batch_size: int = 32,
    *,
    shuffle: bool = False,
    seed: int = RANDOM_SEED,
) -> DataLoader[dict[str, Tensor]]:
    """Build a ``DataLoader`` with seeded shuffling and this project's collate function."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate(batch, pad_id),
        generator=generator,
    )


def build_embedding(vocab_size: int, embedding_dim: int, pad_id: int) -> nn.Embedding:
    """Build a randomly-initialised embedding layer.

    No pretrained vectors are used here: none were provided by the team
    (see ``data/README.md``); ``padding_idx`` is fixed so the padding
    token's embedding stays at zero and never receives a gradient.
    """
    return nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
