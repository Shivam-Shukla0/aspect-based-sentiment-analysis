"""Fine-tune the BERT aspect extractor, score extraction, and optionally score end to end.

Usage: python -m scripts.train_extractor --domain laptop [--classifier-dir DIR]

The extractor is tuned on a validation split of the training sentences (exact-match span F1).
The test file is read once. Extraction is scored by exact and overlap span match with an
error breakdown. With ``--classifier-dir`` the predicted spans are then classified by the
fine-tuned BERT sentiment model and scored end to end (span and label must both match).
Gold aspects labelled ``conflict`` are left out of the end-to-end gold, as the classifier
cannot predict that label. Everything is written to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from absa.config import DATA_PROCESSED_DIR, OUTPUTS_DIR, processed_filename
from absa.data.validation import read_jsonl
from absa.evaluation import evaluate_end_to_end, span_report
from absa.extraction import Sentence, group_sentences
from absa.models.bert_extractor import (
    ExtractorConfig,
    load_extractor,
    predict_spans,
    run_extractor_training,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="train_extractor", description=__doc__)
    parser.add_argument("--domain", required=True, choices=("laptop", "restaurant"))
    parser.add_argument("--data-dir", type=Path, default=DATA_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--lr", type=float, nargs="+", default=[3e-5, 5e-5])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--classifier-dir", type=Path, default=None)
    return parser


def _load_sentences(data_dir: Path, domain: str, split: str) -> list[Sentence]:
    records = read_jsonl(data_dir / processed_filename(domain, split))
    # Sentences with no aspect live in a side file (see data/normalize.py); include them if
    # present so the extractor also sees (and is scored on) "nothing to extract" cases.
    no_aspect = data_dir / f"{domain}_{split}_no_aspect.jsonl"
    return group_sentences(records, read_jsonl(no_aspect) if no_aspect.exists() else [])


def end_to_end_report(
    sentences: Sequence[Sentence],
    predicted: Sequence[Sequence[tuple[int, int]]],
    classifier_dir: Path,
    device: str,
) -> dict[str, object]:
    """Classify the predicted spans with a fine-tuned BERT classifier and score end to end."""
    from absa.models.bert_classifier import (  # noqa: PLC0415  (only needed here)
        BertConfig,
        load_bert,
        predict_bert,
    )

    model, tokenizer, labels = load_bert(classifier_dir)
    config = BertConfig(device=device, drop_conflict="conflict" not in labels)
    # Step 1: turn every predicted span into a (review, aspect text) pair for the classifier.
    pairs = [
        {"review": s.text, "aspect": s.text[start:end]}
        for s, spans in zip(sentences, predicted, strict=True)
        for start, end in spans
    ]
    # Step 2: classify all pairs in one batch, then hand the labels back out in the same
    # order (an iterator) while rebuilding the per-sentence structure.
    labels_flat = iter(predict_bert(model, tokenizer, pairs, config) if pairs else [])
    pred_pairs = [[(start, end, next(labels_flat)) for start, end in spans] for spans in predicted]
    # Gold side: "conflict" aspects are excluded because the classifier cannot predict them.
    gold_pairs = [
        [(start, end, label) for start, end, label in s.spans if label != "conflict"]
        for s in sentences
    ]
    return dict(evaluate_end_to_end(gold_pairs, pred_pairs).to_dict())


def main(argv: Sequence[str] | None = None) -> int:
    """Tune the learning rate, score the test file once and write the summary."""
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or OUTPUTS_DIR / f"extractor_{args.domain}"
    train_sentences = _load_sentences(args.data_dir, args.domain, "train")
    base = ExtractorConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    # Tune the learning rate on validation span F1 (training sentences only).
    runs = []
    for index, learning_rate in enumerate(args.lr):
        config = replace(base, learning_rate=learning_rate)
        result = run_extractor_training(train_sentences, config, output_dir / f"run_{index}")
        runs.append((config, result))
        print(f"run {index}: lr {learning_rate} val span F1 {result.best_val_f1:.4f}")
    best_config, best = max(runs, key=lambda run: run[1].best_val_f1)

    # Final one-time look at the test file with the chosen model.
    model, tokenizer = load_extractor(best.model_dir)
    test_sentences = _load_sentences(args.data_dir, args.domain, "test")
    predicted = predict_spans(model, tokenizer, [s.text for s in test_sentences], best_config)
    report = span_report(
        [s.text for s in test_sentences], [s.gold_spans for s in test_sentences], predicted
    )
    (output_dir / "test_report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    summary: dict[str, object] = {
        "domain": args.domain,
        "train_sentences": len(train_sentences),
        "test_sentences": len(test_sentences),
        "skipped_training_sentences": best.skipped_sentences,
        "runs": [
            {"learning_rate": c.learning_rate, "best_epoch": r.best_epoch, "val_f1": r.best_val_f1}
            for c, r in runs
        ],
        "chosen": asdict(best_config),
        "checkpoint": str(best.model_dir),
        "test_exact": report["exact"],
        "test_overlap": report["overlap"],
    }
    # Optional extra: end-to-end scoring only when a trained sentiment classifier is given.
    if args.classifier_dir is not None:
        summary["end_to_end"] = end_to_end_report(
            test_sentences, predicted, args.classifier_dir, args.device
        )
        summary["classifier"] = str(args.classifier_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    exact = report["exact"]
    print(f"{args.domain}: test exact span F1 {exact['f1']:.4f}")  # type: ignore[index]
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
