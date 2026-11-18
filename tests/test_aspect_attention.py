import torch

from absa.models import AspectAttention, LstmEncoder, masked_mean


def _encoded(
    encoder: LstmEncoder, batch: int, time: int, lengths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    embedded = torch.randn(batch, time, 8)
    states, mask = encoder(embedded, torch.tensor(lengths))
    return states, mask


def test_masked_mean_ignores_padding() -> None:
    states = torch.tensor([[[1.0], [1.0], [100.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.allclose(masked_mean(states, mask), torch.tensor([[1.0]]))


def test_attention_weights_sum_to_one_over_real_tokens() -> None:
    torch.manual_seed(0)
    encoder = LstmEncoder(embedding_dim=8, hidden_size=6, bidirectional=True)
    attention = AspectAttention(hidden_size=12, num_classes=4)
    review_states, review_mask = _encoded(encoder, batch=3, time=5, lengths=[5, 3, 1])
    aspect_states, aspect_mask = _encoded(encoder, batch=3, time=2, lengths=[2, 1, 1])
    _, weights = attention(review_states, review_mask, aspect_states, aspect_mask)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-6)


def test_padding_positions_receive_zero_weight() -> None:
    torch.manual_seed(0)
    encoder = LstmEncoder(embedding_dim=8, hidden_size=6, bidirectional=True)
    attention = AspectAttention(hidden_size=12, num_classes=4)
    review_states, review_mask = _encoded(encoder, batch=2, time=6, lengths=[6, 2])
    aspect_states, aspect_mask = _encoded(encoder, batch=2, time=2, lengths=[2, 1])
    _, weights = attention(review_states, review_mask, aspect_states, aspect_mask)
    padded = weights[~review_mask]
    assert torch.equal(padded, torch.zeros_like(padded))


def test_forward_pass_shapes() -> None:
    encoder = LstmEncoder(embedding_dim=8, hidden_size=6, bidirectional=True)
    attention = AspectAttention(hidden_size=12, num_classes=4)
    review_states, review_mask = _encoded(encoder, batch=2, time=5, lengths=[5, 3])
    aspect_states, aspect_mask = _encoded(encoder, batch=2, time=2, lengths=[2, 1])
    logits, weights = attention(review_states, review_mask, aspect_states, aspect_mask)
    assert logits.shape == (2, 4)
    assert weights.shape == (2, 5)


def test_gradients_flow_to_encoder_and_classifier() -> None:
    torch.manual_seed(0)
    encoder = LstmEncoder(embedding_dim=8, hidden_size=6, bidirectional=True)
    attention = AspectAttention(hidden_size=12, num_classes=4)
    embedding = torch.nn.Embedding(10, 8, padding_idx=0)

    review_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    aspect_ids = torch.tensor([[6, 0], [7, 8]])
    review_lengths = torch.tensor([3, 2])
    aspect_lengths = torch.tensor([1, 2])

    review_states, review_mask = encoder(embedding(review_ids), review_lengths)
    aspect_states, aspect_mask = encoder(embedding(aspect_ids), aspect_lengths)
    logits, _ = attention(review_states, review_mask, aspect_states, aspect_mask)

    loss = logits.sum()
    loss.backward()

    assert embedding.weight.grad is not None
    assert not torch.equal(embedding.weight.grad, torch.zeros_like(embedding.weight.grad))
    assert encoder.lstm.weight_ih_l0.grad is not None
    assert attention.classifier.weight.grad is not None
    assert attention.project_aspect.weight.grad is not None
