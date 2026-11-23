"""Training, validation and checkpointing for the LSTM-with-attention model.

Checkpoints and logs go under ``absa.config.OUTPUTS_DIR`` (git-ignored),
never into the repository; see ``CONTRIBUTING.md``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from absa.config import OUTPUTS_DIR, RANDOM_SEED, SENTIMENT_LABELS
from absa.data import split_train_val
from absa.evaluation import evaluate
from absa.models.lstm_attention import AbsaLstmModel, ModelConfig, resolve_device
from absa.models.lstm_dataset import AspectDataset, make_dataloader
from absa.models.lstm_tokens import Vocabulary
from absa.reproducibility import set_seed


@dataclass(frozen=True, slots=True)
class LstmConfig:
    """Hyperparameters for the LSTM-with-attention model and its training run.

    ``class_weighting`` chooses how class imbalance is handled. ``False`` (the default)
    trains with the plain cross-entropy loss. ``True`` weights each class in the training
    loss by ``total / (classes_present * class_count)``, so rare classes count more; classes
    with no training example get weight 0. Validation loss is never weighted, so it stays
    comparable between runs. ``device`` is ``auto`` (CUDA when available, else CPU), ``cpu``
    or ``cuda``. With ``drop_conflict`` the rare ``conflict`` label is removed from the data and
    left out of the validation macro F1 (the usual SemEval-2014 protocol).
    """

    embedding_dim: int = 50
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = True
    min_freq: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-3
    epochs: int = 5
    val_fraction: float = 0.2
    seed: int = RANDOM_SEED
    class_weighting: bool = False
    device: str = "auto"
    drop_conflict: bool = False

    @property
    def metric_labels(self) -> tuple[str, ...]:
        """Return the labels that validation macro F1 is computed over."""
        if self.drop_conflict:
            return tuple(label for label in SENTIMENT_LABELS if label != "conflict")
        return tuple(SENTIMENT_LABELS)

    @property
    def model_config(self) -> ModelConfig:
        """Return the architecture subset of this config, for building ``AbsaLstmModel``."""
        return ModelConfig(
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Training and validation loss and macro F1 for one epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_macro_f1: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Everything one training run produces."""

    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_val_macro_f1: float
    checkpoint_path: Path


def save_checkpoint(
    path: Path, model: AbsaLstmModel, vocab: Vocabulary, config: LstmConfig
) -> None:
    """Save the model weights, vocabulary and config needed to reload the model."""
    # The checkpoint bundles everything needed to rebuild the model elsewhere: weights, the
    # vocabulary (token ids must match) and the config (architecture sizes).
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_tokens": list(vocab.tokens),
            "config": asdict(config),
            "labels": list(SENTIMENT_LABELS),
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[AbsaLstmModel, Vocabulary]:
    """Load a checkpoint written by ``save_checkpoint``, returning ``(model, vocab)``."""
    # weights_only=True refuses to unpickle arbitrary objects: safer for files from others.
    # map_location="cpu" lets a GPU-trained checkpoint load on a machine without a GPU.
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    vocab = Vocabulary(checkpoint["vocab_tokens"])
    config = LstmConfig(**checkpoint["config"])
    model = AbsaLstmModel(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        num_classes=len(checkpoint["labels"]),
        config=config.model_config,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model, vocab


def _class_weights(records: Sequence[Mapping[str, object]]) -> Tensor:
    """Return balanced per-class loss weights, in ``SENTIMENT_LABELS`` order."""
    # "Balanced" weight = total / (present_classes * class_count): rare classes get a larger
    # weight so the loss does not just reward predicting the majority class.
    counts = [sum(1 for r in records if r["sentiment"] == label) for label in SENTIMENT_LABELS]
    present = sum(1 for count in counts if count > 0)
    total = sum(counts)
    return torch.tensor(
        [total / (present * count) if count else 0.0 for count in counts], dtype=torch.float
    )


def _run_epoch(
    model: AbsaLstmModel,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer | None,
    class_weights: Tensor | None = None,
    device: torch.device | None = None,
) -> tuple[float, list[str], list[str]]:
    """Run one pass over ``loader``; trains if ``optimizer`` is given, else evaluates.

    Returns the average per-batch loss and the true/predicted label
    strings for every example seen, for metric computation. ``class_weights`` is
    applied to the loss only when training.
    """
    # One function serves both phases: an optimizer means "train", None means "evaluate".
    # train mode enables dropout; eval mode disables it.
    model.train(mode=optimizer is not None)
    total_loss = 0.0
    num_batches = 0
    y_true: list[str] = []
    y_pred: list[str] = []
    for cpu_batch in loader:
        # Batches are built on CPU by the DataLoader and moved to the target device here.
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        # Track gradients only while training; skipping them in validation saves memory/time.
        with torch.set_grad_enabled(optimizer is not None):
            logits: Tensor
            logits, _ = model(
                batch["review_ids"],
                batch["review_lengths"],
                batch["aspect_ids"],
                batch["aspect_lengths"],
            )
            loss: Tensor = nn.functional.cross_entropy(
                # Class weights apply to the training loss only, so validation loss stays comparable
                # between runs with and without weighting.
                logits,
                batch["label"],
                weight=class_weights if optimizer is not None else None,
            )
        if optimizer is not None:
            # Standard step: clear old gradients, backpropagate, update the weights.
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]  # Tensor.backward lacks a full stub
            optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        # Keep labels as strings so the shared evaluate() function can score them.
        y_true.extend(SENTIMENT_LABELS[index] for index in batch["label"].tolist())
        y_pred.extend(SENTIMENT_LABELS[index] for index in logits.argmax(dim=-1).tolist())
    return total_loss / max(num_batches, 1), y_true, y_pred


def train(
    train_records: Sequence[Mapping[str, object]],
    val_records: Sequence[Mapping[str, object]],
    config: LstmConfig,
) -> tuple[AbsaLstmModel, Vocabulary, tuple[EpochMetrics, ...]]:
    """Train on ``train_records``, validating on ``val_records`` each epoch.

    Seeds every random number generator before building the model or
    dataloaders, so a given ``config.seed`` reproduces the same first
    epoch's training loss. Returns the model with the best validation macro F1
    (the earliest such epoch on a tie), its vocabulary (built from
    ``train_records`` only) and the per-epoch history.
    """
    set_seed(config.seed)  # first, so weight init and shuffling are reproducible
    device = resolve_device(config.device)
    # Vocabulary from TRAINING reviews only: building it on val/test text would leak information.
    vocab = Vocabulary.build([str(r["review"]) for r in train_records], min_freq=config.min_freq)
    model = AbsaLstmModel(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        num_classes=len(SENTIMENT_LABELS),
        config=config.model_config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_dataloader(
        AspectDataset(train_records, vocab),
        vocab.pad_id,
        config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = make_dataloader(AspectDataset(val_records, vocab), vocab.pad_id, config.batch_size)

    # Class weights are computed from the training split only, and only when requested.
    weights = _class_weights(train_records).to(device) if config.class_weighting else None
    history = []
    # Model selection: remember the weights of the epoch with the best validation macro F1.
    best_f1 = float("-inf")
    best_state: dict[str, Tensor] = {}
    for epoch in range(1, config.epochs + 1):
        train_loss, _, _ = _run_epoch(model, train_loader, optimizer, weights, device)
        val_loss, val_true, val_pred = _run_epoch(model, val_loader, None, None, device)
        val_macro_f1 = evaluate(val_true, val_pred, labels=config.metric_labels).macro_f1
        history.append(EpochMetrics(epoch, train_loss, val_loss, val_macro_f1))
        # Strict ">" keeps the EARLIEST epoch on ties. deepcopy is needed because state_dict()
        # returns references that later epochs would overwrite.
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_state = copy.deepcopy(model.state_dict())
    # Restore the best epoch's weights (not the last epoch's) before returning.
    if best_state:
        model.load_state_dict(best_state)
    return model, vocab, tuple(history)


def run_training(
    records: Sequence[Mapping[str, object]], config: LstmConfig, output_dir: Path = OUTPUTS_DIR
) -> TrainingResult:
    """Split ``records``, train, and write the checkpoint and a run log to ``output_dir``.

    The checkpoint holds the weights of the epoch with the best validation macro F1,
    which is the epoch reported as ``best_epoch``; ``history`` records every epoch.
    With ``config.drop_conflict`` the ``conflict`` records are removed first.
    """
    # Optional protocol choice: remove the rare "conflict" class before splitting.
    if config.drop_conflict:
        records = [record for record in records if record["sentiment"] != "conflict"]
    train_records, val_records = split_train_val(
        records, val_fraction=config.val_fraction, seed=config.seed
    )
    model, vocab, history = train(train_records, val_records, config)
    # max() returns the first maximal item, matching the "earliest epoch wins" rule in train().
    best = max(history, key=lambda epoch_metrics: epoch_metrics.val_macro_f1)

    checkpoint_path = output_dir / "lstm_checkpoint.pt"
    save_checkpoint(checkpoint_path, model, vocab, config)

    # Human-readable run log next to the checkpoint, so every result can be traced to its config.
    log_path = output_dir / "lstm_training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "history": [asdict(epoch_metrics) for epoch_metrics in history],
                "best_epoch": best.epoch,
                "best_val_macro_f1": best.val_macro_f1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainingResult(
        history=history,
        best_epoch=best.epoch,
        best_val_macro_f1=best.val_macro_f1,
        checkpoint_path=checkpoint_path,
    )
