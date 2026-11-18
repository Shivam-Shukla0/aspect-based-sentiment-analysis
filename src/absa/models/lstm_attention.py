"""LSTM with aspect attention.

Contract: classify the sentiment of a supplied ``aspect`` within a
``review``, returning one label from ``absa.config.SENTIMENT_LABELS``.
The encoder below produces per-token hidden states and a mask; the
attention module that turns these into a classification is added
separately.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmEncoder(nn.Module):
    """Encodes a padded, embedded sequence into per-token hidden states and a mask.

    Uses a packed sequence internally so padded positions are never fed to
    the LSTM: real-token outputs do not depend on what values (if any) sit
    in the padding.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        *,
        bidirectional: bool = True,
    ) -> None:
        """Configure the underlying ``nn.LSTM``.

        ``dropout`` is only applied between layers, so it is silently
        ignored (passed as 0.0 to ``nn.LSTM``) when ``num_layers`` is 1,
        avoiding the warning PyTorch would otherwise raise.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.output_size = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, embedded: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a padded, embedded batch.

        ``embedded`` is ``(batch, time, embedding_dim)`` and ``lengths``
        is ``(batch,)``, the number of real (non-padding) tokens per
        example — the same shapes produced by
        ``absa.models.lstm_dataset.collate`` after an embedding lookup.

        Returns ``(states, mask)``: ``states`` is ``(batch, time,
        output_size)`` and ``mask`` is a ``(batch, time)`` boolean tensor,
        true at real-token positions and false at padding.
        """
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_states, _ = self.lstm(packed)
        states, _ = pad_packed_sequence(
            packed_states, batch_first=True, total_length=embedded.size(1)
        )
        time_steps = embedded.size(1)
        positions = torch.arange(time_steps, device=lengths.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        return states, mask


def masked_mean(states: Tensor, mask: Tensor) -> Tensor:
    """Average ``states`` over the time dimension, counting only ``mask``-true positions.

    ``states`` is ``(batch, time, hidden)`` and ``mask`` is ``(batch,
    time)``. Every row of ``mask`` must have at least one true position.
    """
    weights = mask.unsqueeze(-1).to(states.dtype)
    counts = weights.sum(dim=1)
    return (states * weights).sum(dim=1) / counts


class AspectAttention(nn.Module):
    """Scores review tokens against an aspect representation and classifies.

    The aspect representation is the masked mean of the aspect's own
    encoded states (the aspect is run through the same kind of encoder as
    the review, sharing its hidden size — see ``docs/lstm_architecture.md``
    for why). Attention uses Luong-style "general" scoring: a learned
    linear map of the aspect representation is dotted with each review
    token's hidden state, masked to ``-inf`` at padding before the
    softmax, so padding gets exactly zero weight.
    """

    def __init__(self, hidden_size: int, num_classes: int) -> None:
        """``hidden_size`` must match the encoder's ``output_size``."""
        super().__init__()
        self.project_aspect = nn.Linear(hidden_size, hidden_size, bias=False)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        review_states: Tensor,
        review_mask: Tensor,
        aspect_states: Tensor,
        aspect_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Classify sentiment, returning ``(logits, attention_weights)``.

        ``review_states``/``aspect_states`` are ``(batch, time, hidden)``
        and ``review_mask``/``aspect_mask`` are the matching ``(batch,
        time)`` boolean masks, all from ``LstmEncoder``.
        """
        aspect_repr = masked_mean(aspect_states, aspect_mask)
        projected_aspect = self.project_aspect(aspect_repr)
        scores = torch.einsum("bth,bh->bt", review_states, projected_aspect)
        scores = scores.masked_fill(~review_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bt,bth->bh", weights, review_states)
        logits = self.classifier(context)
        return logits, weights
