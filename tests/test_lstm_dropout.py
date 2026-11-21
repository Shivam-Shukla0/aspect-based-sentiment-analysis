import pytest

pytest.importorskip("torch")
import torch

from absa.models.lstm_attention import AbsaLstmModel, ModelConfig


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    review = torch.tensor([[2, 3, 4, 5], [2, 3, 0, 0]])
    lengths = torch.tensor([4, 2])
    aspect = torch.tensor([[3], [2]])
    return review, lengths, aspect, torch.tensor([1, 1])


def _model(dropout: float) -> AbsaLstmModel:
    torch.manual_seed(0)
    config = ModelConfig(embedding_dim=8, hidden_size=4, dropout=dropout)
    return AbsaLstmModel(vocab_size=10, pad_id=0, num_classes=4, config=config)


def test_dropout_changes_training_passes_but_not_evaluation_passes() -> None:
    model = _model(0.5)
    inputs = _inputs()
    model.train()
    first, _ = model(*inputs)
    second, _ = model(*inputs)
    assert not torch.equal(first, second)
    model.eval()
    third, _ = model(*inputs)
    fourth, _ = model(*inputs)
    assert torch.equal(third, fourth)


def test_zero_dropout_is_deterministic_in_training_mode() -> None:
    model = _model(0.0)
    inputs = _inputs()
    model.train()
    first, _ = model(*inputs)
    second, _ = model(*inputs)
    assert torch.equal(first, second)
