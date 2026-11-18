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
