"""BERT token-classification aspect extractor.

Words come from the regular-expression tokenizer used for the BIO tags (``absa.extraction``);
each word is split into word pieces, only the first piece of a word carries the tag during
training, and the remaining pieces and the special tokens are masked with ``-100``. At
prediction time the tag of a word is read from its first piece and decoded back into
character spans of the original text.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from absa.config import OUTPUTS_DIR, RANDOM_SEED
from absa.evaluation import evaluate_extraction_spans
from absa.extraction import (
    BIO_TAGS,
    ID_TO_TAG,
    Sentence,
    build_bio_example,
    decode_spans,
    split_sentences,
    tokenize_with_offsets,
)
from absa.models.lstm_attention import resolve_device
from absa.reproducibility import set_seed

MODEL_SUBDIR = "model"
LOG_NAME = "extractor_training_log.json"
# PyTorch's cross-entropy skips targets equal to -100, so masked positions add no loss.
IGNORE_INDEX = -100
Span = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ExtractorConfig:
    """Hyperparameters for fine-tuning the token-classification extractor."""

    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 5e-5
    epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    val_fraction: float = 0.2
    seed: int = RANDOM_SEED
    device: str = "auto"
    bf16: bool = True


@dataclass(frozen=True, slots=True)
class ExtractorEpoch:
    """Training loss and validation span scores for one epoch."""

    epoch: int
    train_loss: float
    val_precision: float
    val_recall: float
    val_f1: float


@dataclass(frozen=True, slots=True)
class ExtractorResult:
    """Everything one extractor fine-tuning run produces."""

    history: tuple[ExtractorEpoch, ...]
    best_epoch: int
    best_val_f1: float
    skipped_sentences: int
    model_dir: Path


# How the extractor works: token classification with BIO tags: O (outside), B-ASP (beginning of
# an aspect), I-ASP (inside). BERT splits words into word pieces; each word's tag is placed on
# its first piece and the other pieces and special tokens are ignored in the loss (label -100).
# At prediction time the tag of each word is read from its first piece and decoded back to
# character offsets. It was trained 5 epochs with a learning rate chosen from 3 and 5 x 1e-5.
def align_labels(word_ids: Sequence[int | None], word_tags: Sequence[int]) -> list[int]:
    """Give each word's tag to its first word piece; mask special tokens and other pieces."""
    # BERT splits a word into several word pieces ("keyboard" -> "key", "##board"), but our
    # BIO tags are per WORD. Only the first piece gets the tag; all others are ignored in
    # the loss. word_ids is None for special tokens ([CLS], [SEP], padding).
    labels: list[int] = []
    previous: int | None = None
    for word_id in word_ids:
        if word_id is None or word_id == previous:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(word_tags[word_id])
        previous = word_id
    return labels


def first_piece_tags(
    word_ids: Sequence[int | None], piece_tags: Sequence[int], n_words: int
) -> list[str]:
    """Read one tag per word from the prediction at its first word piece (``O`` if cut off)."""
    # Inverse of align_labels: start with "O" for every word, so words cut off by
    # truncation (no pieces left) default to "outside".
    tags = ["O"] * n_words
    previous: int | None = None
    for position, word_id in enumerate(word_ids):
        if word_id is not None and word_id != previous:
            tags[word_id] = ID_TO_TAG[piece_tags[position]]
        previous = word_id
    return tags


def _encode(
    tokenizer: object, texts: Sequence[str], config: ExtractorConfig
) -> tuple[dict[str, Tensor], list[list[int | None]], list[int]]:
    """Tokenize ``texts`` as pre-split words; return tensors, word ids and word counts."""
    # Pre-split into the same regex words the BIO tags were built on, so tags and word
    # indices stay aligned; the tokenizer then only sub-splits into word pieces.
    words = [[t.token for t in tokenize_with_offsets(text)] for text in texts]
    encoded = tokenizer(  # type: ignore[operator]
        words,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    # word_ids[i][j] = index of the word that piece j belongs to (None for special tokens).
    word_ids = [encoded.word_ids(i) for i in range(len(words))]
    return dict(encoded), word_ids, [len(w) for w in words]


def _train_loader(
    sentences: Sequence[Sentence], tokenizer: object, config: ExtractorConfig
) -> tuple[DataLoader[dict[str, Tensor]], int]:
    """Build the training loader; sentences whose spans cannot be tagged are skipped."""
    examples: list[tuple[str, list[int]]] = []
    for sentence in sentences:
        try:
            tagged = build_bio_example(sentence.text, sentence.gold_spans)
        # Gold spans that cannot be expressed as clean word tags are skipped (and counted
        # in the return value) instead of training on wrong labels.
        except ValueError:  # overlapping or token-splitting gold spans
            continue
        examples.append((sentence.text, list(tagged.tag_ids)))
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    def collate(batch: list[tuple[str, list[int]]]) -> dict[str, Tensor]:
        tensors, word_ids, _ = _encode(tokenizer, [text for text, _ in batch], config)
        rows = [align_labels(ids, tags) for ids, (_, tags) in zip(word_ids, batch, strict=True)]
        tensors["labels"] = torch.tensor(rows, dtype=torch.long)
        return tensors

    loader = DataLoader(
        examples,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    return loader, len(sentences) - len(examples)


def predict_spans(
    model: torch.nn.Module, tokenizer: object, texts: Sequence[str], config: ExtractorConfig
) -> list[list[Span]]:
    """Return the predicted aspect spans for each text, in text order."""
    device = resolve_device(config.device)
    model.to(device)
    model.eval()
    use_amp = config.bf16 and device.type == "cuda"
    spans: list[list[Span]] = []
    # Predict in chunks so memory stays bounded for long input lists.
    for offset in range(0, len(texts), config.batch_size):
        chunk = texts[offset : offset + config.batch_size]
        tensors, word_ids, counts = _encode(tokenizer, chunk, config)
        batch = {key: value.to(device) for key, value in tensors.items()}
        with (
            torch.no_grad(),
            torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp),
        ):
            logits = model(**batch).logits
        # Highest-scoring tag per word piece; first_piece_tags then picks each word's tag, and
        # decode_spans turns word tags back into character offsets of the original text.
        pieces = logits.argmax(dim=-1).tolist()
        for text, ids, tags, count in zip(chunk, word_ids, pieces, counts, strict=True):
            spans.append(decode_spans(text, first_piece_tags(ids, tags, count)))
    return spans


def score_sentences(
    model: torch.nn.Module,
    tokenizer: object,
    sentences: Sequence[Sentence],
    config: ExtractorConfig,
) -> tuple[float, float, float]:
    """Return exact-match ``(precision, recall, f1)`` of the model on ``sentences``."""
    predicted = predict_spans(model, tokenizer, [s.text for s in sentences], config)
    # Exact-match span metrics: the same protocol used for every extraction result.
    metrics = evaluate_extraction_spans([s.gold_spans for s in sentences], predicted)
    return metrics.precision, metrics.recall, metrics.f1


def train_extractor(
    train_sentences: Sequence[Sentence], val_sentences: Sequence[Sentence], config: ExtractorConfig
) -> tuple[torch.nn.Module, object, tuple[ExtractorEpoch, ...], int]:
    """Fine-tune and return the best epoch's model, tokenizer, history and skipped count.

    The best epoch has the highest validation exact-match span F1 (the earliest on a tie).
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(BIO_TAGS),
        id2label=dict(enumerate(BIO_TAGS)),
        label2id={tag: index for index, tag in enumerate(BIO_TAGS)},
    ).to(device)
    loader, skipped = _train_loader(train_sentences, tokenizer, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total = max(config.epochs * len(loader), 1)
    scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer, int(total * config.warmup_ratio), total
    )
    use_amp = config.bf16 and device.type == "cuda"

    history: list[ExtractorEpoch] = []
    best_f1, best_state = float("-inf"), {}
    for epoch in range(1, config.epochs + 1):
        model.train()  # enable dropout for the training pass
        losses: list[float] = []
        for cpu_batch in loader:
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = model(**batch).loss
            # Per batch: clear gradients, backpropagate, clip, update weights, advance LR schedule.
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))
        # Validation uses span F1 (not token accuracy): it is what the task is graded on.
        precision, recall, f1 = score_sentences(model, tokenizer, val_sentences, config)
        history.append(
            ExtractorEpoch(epoch, sum(losses) / max(len(losses), 1), precision, recall, f1)
        )
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    if best_state:
        model.load_state_dict(best_state)
    return model, tokenizer, tuple(history), skipped


def run_extractor_training(
    sentences: Sequence[Sentence], config: ExtractorConfig, output_dir: Path = OUTPUTS_DIR
) -> ExtractorResult:
    """Split ``sentences`` by text, fine-tune, and save the model, tokenizer and a run log."""
    train_sentences, val_sentences = split_sentences(sentences, config.val_fraction, config.seed)
    model, tokenizer, history, skipped = train_extractor(train_sentences, val_sentences, config)
    best = max(history, key=lambda epoch: epoch.val_f1)
    model_dir = output_dir / MODEL_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)  # type: ignore[operator]
    tokenizer.save_pretrained(model_dir)  # type: ignore[attr-defined]
    log = {
        "config": asdict(config),
        "history": [asdict(epoch) for epoch in history],
        "best_epoch": best.epoch,
        "best_val_f1": best.val_f1,
        "skipped_sentences": skipped,
        "train_sentences": len(train_sentences),
        "val_sentences": len(val_sentences),
    }
    (output_dir / LOG_NAME).write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return ExtractorResult(history, best.epoch, best.val_f1, skipped, model_dir)


def load_extractor(model_dir: Path) -> tuple[torch.nn.Module, object]:
    """Load a model saved by ``run_extractor_training``; returns ``(model, tokenizer)``."""
    return (
        AutoModelForTokenClassification.from_pretrained(model_dir),
        AutoTokenizer.from_pretrained(model_dir),
    )
