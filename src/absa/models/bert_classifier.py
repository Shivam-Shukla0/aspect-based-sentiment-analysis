"""BERT sentence-pair classifier for aspect sentiment.

The input is the pair ``[CLS] review [SEP] aspect [SEP]`` and the classification head
predicts one label for that aspect. Training follows the usual recipe: AdamW with a
linear warm-up and decay, gradient clipping, optional bfloat16 autocast on a GPU, and the
weights of the epoch with the best validation macro F1 are the ones that are kept.

Records are dictionaries with ``review``, ``aspect`` and ``sentiment``. With
``drop_conflict`` the rare ``conflict`` label is removed before the split and the head has
three outputs, which is the usual SemEval-2014 protocol.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from absa.config import OUTPUTS_DIR, RANDOM_SEED, SENTIMENT_LABELS
from absa.data import split_train_val
from absa.evaluation import evaluate
from absa.models.lstm_attention import resolve_device
from absa.models.lstm_evaluation import TestEvaluationReport, analyze_errors
from absa.reproducibility import set_seed

MODEL_SUBDIR = "model"
LOG_NAME = "bert_training_log.json"


# Model summary: bert-base-uncased as a sentence-pair classifier, [CLS] review [SEP] aspect
# [SEP], with a classification layer on [CLS] giving three outputs. AdamW (weight decay 0.01),
# linear schedule with 10% warm-up, gradient clipping 1.0, 4 epochs, batch 32, max length 128
# (only the review is truncated), bfloat16 on the GPU, best epoch kept. The learning rate was
# chosen from 2, 3, 5 x 1e-5; 5e-5 was best.
#
# Why BERT beats the LSTM: BERT is pre-trained on huge text corpora, so it already knows word
# meaning, negation and context; fine-tuning only adapts it. The LSTM starts from random word
# vectors and learns only from 2-4 thousand examples. Test accuracy BERT 0.759/0.829 vs. LSTM
# 0.599/0.715 (laptop/restaurant), about 16 and 11 points better.
#
# Why the sentence-pair format: putting the aspect as a second segment lets BERT's
# self-attention directly relate the aspect tokens to the review tokens, so the same review can
# get different labels for different aspects. This formulation is known to work well for ABSA
# (Sun et al., 2019).
@dataclass(frozen=True, slots=True)
class BertConfig:
    """Hyperparameters for fine-tuning a BERT sentence-pair classifier.

    ``bf16`` turns on bfloat16 autocast, which only has an effect on a CUDA device.
    ``drop_conflict`` removes the ``conflict`` label from the data and from the head.
    """

    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    val_fraction: float = 0.2
    seed: int = RANDOM_SEED
    device: str = "auto"
    drop_conflict: bool = True
    bf16: bool = True

    @property
    def labels(self) -> tuple[str, ...]:
        """Return the labels of the classification head, in id order."""
        if self.drop_conflict:
            return tuple(label for label in SENTIMENT_LABELS if label != "conflict")
        return tuple(SENTIMENT_LABELS)


@dataclass(frozen=True, slots=True)
class BertEpoch:
    """Training and validation figures for one epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float


@dataclass(frozen=True, slots=True)
class BertResult:
    """Everything one fine-tuning run produces."""

    history: tuple[BertEpoch, ...]
    best_epoch: int
    best_val_macro_f1: float
    model_dir: Path


def _loader(
    records: Sequence[Mapping[str, object]],
    tokenizer: object,
    config: BertConfig,
    *,
    shuffle: bool,
) -> DataLoader[dict[str, Tensor]]:
    """Build a loader that tokenizes each batch as (review, aspect) pairs."""
    # Label name -> head output index; must match config.labels (and thus id2label on load).
    label_ids = {label: index for index, label in enumerate(config.labels)}
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    # Tokenization happens per batch (not up front): padding then adapts to each batch's
    # longest pair instead of a global maximum.
    def collate(batch: list[Mapping[str, object]]) -> dict[str, Tensor]:
        encoded = tokenizer(  # type: ignore[operator]
            [str(record["review"]) for record in batch],
            [str(record["aspect"]) for record in batch],
            padding=True,
            # Two segments: [CLS] review [SEP] aspect [SEP]. Truncating only the review keeps
            # the short aspect intact (it is the thing being classified).
            truncation="only_first",
            max_length=config.max_length,
            return_tensors="pt",
        )
        tensors: dict[str, Tensor] = dict(encoded)
        # Labels are optional so the same loader serves training and label-free prediction.
        if all("sentiment" in record for record in batch):
            tensors["labels"] = torch.tensor(
                [label_ids[str(record["sentiment"])] for record in batch], dtype=torch.long
            )
        return tensors

    return DataLoader(
        list(records),  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator,
    )


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    config: BertConfig,
    device: torch.device,
    optimization: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler] | None,
) -> tuple[float, list[str], list[str]]:
    """Run one pass; trains when ``optimization`` is given. Returns loss and label strings."""
    training = optimization is not None
    model.train(mode=training)
    labels = config.labels
    # bf16 autocast only makes sense on CUDA; elsewhere it is silently disabled.
    use_amp = config.bf16 and device.type == "cuda"
    total_loss, batches = 0.0, 0
    y_true: list[str] = []
    y_pred: list[str] = []
    for cpu_batch in loader:
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp),
        ):
            # Passing "labels" makes the HF model compute the cross-entropy loss itself.
            output = model(**batch)
        if optimization is not None:
            optimizer, scheduler = optimization
            optimizer.zero_grad()
            output.loss.backward()
            # Clip the gradient norm to 1.0: avoids rare exploding updates when fine-tuning.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()  # the LR schedule advances per BATCH, not per epoch
        total_loss += float(output.loss.item())
        batches += 1
        y_true.extend(labels[index] for index in batch["labels"].tolist())
        y_pred.extend(labels[index] for index in output.logits.argmax(dim=-1).tolist())
    return total_loss / max(batches, 1), y_true, y_pred


def train_bert(
    train_records: Sequence[Mapping[str, object]],
    val_records: Sequence[Mapping[str, object]],
    config: BertConfig,
) -> tuple[torch.nn.Module, object, tuple[BertEpoch, ...]]:
    """Fine-tune on ``train_records`` and return the best epoch's model, tokenizer and history.

    The best epoch is the one with the highest validation macro F1 (the earliest on a tie).
    The training and validation records must not contain labels outside ``config.labels``.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    labels = config.labels
    # Pretrained weights come from the Hugging Face hub; the classification head on top is
    # new and randomly initialised, which is what fine-tuning then trains.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
    ).to(device)
    train_loader = _loader(train_records, tokenizer, config, shuffle=True)
    val_loader = _loader(val_records, tokenizer, config, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # Linear warm-up (first warmup_ratio of the steps) then linear decay to zero: the usual
    # BERT fine-tuning schedule. max(..., 1) avoids a zero-step schedule on empty data.
    total_steps = max(config.epochs * len(train_loader), 1)
    scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )

    history: list[BertEpoch] = []
    best_f1 = float("-inf")
    best_state: dict[str, Tensor] = {}
    for epoch in range(1, config.epochs + 1):
        train_loss, _, _ = _run_epoch(model, train_loader, config, device, (optimizer, scheduler))
        val_loss, val_true, val_pred = _run_epoch(model, val_loader, config, device, None)
        result = evaluate(val_true, val_pred, labels=labels)
        history.append(BertEpoch(epoch, train_loss, val_loss, result.accuracy, result.macro_f1))
        if result.macro_f1 > best_f1:
            best_f1 = result.macro_f1
            # Copy to CPU so keeping the best weights does not occupy extra GPU memory.
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    if best_state:
        model.load_state_dict(best_state)
    return model, tokenizer, tuple(history)


def run_bert_training(
    records: Sequence[Mapping[str, object]], config: BertConfig, output_dir: Path = OUTPUTS_DIR
) -> BertResult:
    """Split ``records``, fine-tune, and save the model, tokenizer and a run log.

    The saved model is the best epoch by validation macro F1. With ``config.drop_conflict``
    the ``conflict`` records are removed before the split.
    """
    if config.drop_conflict:
        records = [record for record in records if record["sentiment"] != "conflict"]
    train_records, val_records = split_train_val(
        records, val_fraction=config.val_fraction, seed=config.seed
    )
    model, tokenizer, history = train_bert(train_records, val_records, config)
    best = max(history, key=lambda epoch: epoch.val_macro_f1)

    # save_pretrained writes weights + config; the tokenizer goes in the same directory so
    # load_bert needs a single path.
    model_dir = output_dir / MODEL_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)  # type: ignore[operator]
    tokenizer.save_pretrained(model_dir)  # type: ignore[attr-defined]
    log = {
        "config": asdict(config),
        "history": [asdict(epoch) for epoch in history],
        "best_epoch": best.epoch,
        "best_val_macro_f1": best.val_macro_f1,
    }
    (output_dir / LOG_NAME).write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return BertResult(history, best.epoch, best.val_macro_f1, model_dir)


def load_bert(model_dir: Path) -> tuple[torch.nn.Module, object, tuple[str, ...]]:
    """Load a model saved by ``run_bert_training``; returns ``(model, tokenizer, labels)``."""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Recover the label order from the saved config so predictions use the training-time ids.
    id2label = model.config.id2label
    labels = tuple(id2label[index] for index in sorted(id2label))
    return model, tokenizer, labels


def predict_bert(
    model: torch.nn.Module,
    tokenizer: object,
    records: Sequence[Mapping[str, object]],
    config: BertConfig,
) -> list[str]:
    """Return one predicted label per record, in order. Records need no ``sentiment``."""
    device = resolve_device(config.device)
    model.to(device)
    model.eval()
    labels = config.labels
    predictions: list[str] = []
    use_amp = config.bf16 and device.type == "cuda"
    with torch.no_grad():
        for cpu_batch in _loader(records, tokenizer, config, shuffle=False):
            # Drop "labels" if present: prediction must not depend on gold answers.
            batch = {key: value.to(device) for key, value in cpu_batch.items() if key != "labels"}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(**batch).logits
            predictions.extend(labels[index] for index in logits.argmax(dim=-1).tolist())
    return predictions


def evaluate_bert_test(
    model: torch.nn.Module,
    tokenizer: object,
    records: Sequence[Mapping[str, object]],
    config: BertConfig,
) -> TestEvaluationReport:
    """Score ``records`` (the frozen test set): metrics, confusion matrix and error list.

    The report has the same shape as the LSTM one, so the two can be compared directly.
    """
    from sklearn.metrics import confusion_matrix  # noqa: PLC0415  (only needed here)

    labels = config.labels
    y_true = [str(record["sentiment"]) for record in records]
    y_pred = predict_bert(model, tokenizer, records, config)
    return TestEvaluationReport(
        metrics=evaluate(y_true, y_pred, labels=labels),
        confusion=confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        labels=labels,
        errors=analyze_errors(records, y_true, y_pred),
    )
