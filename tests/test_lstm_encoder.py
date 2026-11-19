import pytest

pytest.importorskip("torch")
import torch

from absa.models import LstmEncoder


def test_output_shapes() -> None:
    batch, time, embedding_dim, hidden_size = 3, 5, 8, 6
    encoder = LstmEncoder(embedding_dim, hidden_size, bidirectional=True)
    embedded = torch.randn(batch, time, embedding_dim)
    lengths = torch.tensor([5, 3, 1])
    states, mask = encoder(embedded, lengths)
    assert states.shape == (batch, time, hidden_size * 2)
    assert mask.shape == (batch, time)
    assert mask.dtype == torch.bool


def test_unidirectional_output_size() -> None:
    encoder = LstmEncoder(8, 6, bidirectional=False)
    embedded = torch.randn(2, 4, 8)
    lengths = torch.tensor([4, 2])
    states, _ = encoder(embedded, lengths)
    assert states.shape == (2, 4, 6)


def test_mask_is_true_for_real_tokens_and_false_for_padding() -> None:
    encoder = LstmEncoder(8, 6)
    embedded = torch.randn(2, 5, 8)
    lengths = torch.tensor([5, 2])
    _, mask = encoder(embedded, lengths)
    assert mask[0].tolist() == [True, True, True, True, True]
    assert mask[1].tolist() == [True, True, False, False, False]


def test_padding_values_do_not_affect_real_token_outputs() -> None:
    torch.manual_seed(0)
    encoder = LstmEncoder(8, 6, bidirectional=True)
    encoder.eval()
    batch, time, embedding_dim = 1, 6, 8
    lengths = torch.tensor([3])

    base = torch.randn(batch, time, embedding_dim)
    base[:, 3:, :] = 0.0
    perturbed = base.clone()
    perturbed[:, 3:, :] = torch.randn(batch, time - 3, embedding_dim) * 100

    with torch.no_grad():
        states_base, _ = encoder(base, lengths)
        states_perturbed, _ = encoder(perturbed, lengths)

    assert torch.allclose(states_base[:, :3], states_perturbed[:, :3], atol=1e-6)


def test_runs_on_cpu() -> None:
    encoder = LstmEncoder(8, 6)
    embedded = torch.randn(2, 4, 8, device="cpu")
    lengths = torch.tensor([4, 3], device="cpu")
    states, mask = encoder(embedded, lengths)
    assert states.device.type == "cpu"
    assert mask.device.type == "cpu"


def test_multi_layer_dropout_does_not_warn() -> None:
    encoder = LstmEncoder(8, 6, num_layers=2, dropout=0.5)
    embedded = torch.randn(2, 4, 8)
    lengths = torch.tensor([4, 2])
    encoder(embedded, lengths)


def test_single_layer_with_dropout_argument_does_not_warn() -> None:
    encoder = LstmEncoder(8, 6, num_layers=1, dropout=0.5)
    embedded = torch.randn(2, 4, 8)
    lengths = torch.tensor([4, 2])
    encoder(embedded, lengths)
