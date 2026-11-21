"""Fine-tune BERT on one domain, tune on the validation split, score the test file once.

Usage: python -m scripts.train_bert --domain laptop

For each learning rate it fine-tunes a sentence-pair classifier, keeps the setting with the
best validation macro F1 and only then reads the test file. The ``conflict`` label is
dropped, as in the usual SemEval-2014 protocol. Everything is written to ``--output-dir``.
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
from absa.models.bert_classifier import (
    BertConfig,
    evaluate_bert_test,
    load_bert,
    run_bert_training,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="train_bert", description=__doc__)
    parser.add_argument("--domain", required=True, choices=("laptop", "restaurant"))
    parser.add_argument("--data-dir", type=Path, default=DATA_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--lr", type=float, nargs="+", default=[2e-5, 3e-5, 5e-5])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the learning-rate search, evaluate the best model on the test file, write a summary."""
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or OUTPUTS_DIR / f"bert_{args.domain}"
    train_records = read_jsonl(args.data_dir / processed_filename(args.domain, "train"))
    base = BertConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    runs = []
    for index, learning_rate in enumerate(args.lr):
        config = replace(base, learning_rate=learning_rate)
        result = run_bert_training(train_records, config, output_dir=output_dir / f"run_{index}")
        runs.append((config, result))
        print(f"run {index}: lr {learning_rate} val macro F1 {result.best_val_macro_f1:.4f}")
    best_config, best = max(runs, key=lambda run: run[1].best_val_macro_f1)

    model, tokenizer, _ = load_bert(best.model_dir)
    test_records = [
        record
        for record in read_jsonl(args.data_dir / processed_filename(args.domain, "test"))
        if record["sentiment"] != "conflict"
    ]
    report = evaluate_bert_test(model, tokenizer, test_records, best_config)
    report.save(output_dir / "test_report.json")
    summary = {
        "domain": args.domain,
        "train_records": len(train_records),
        "test_records": len(test_records),
        "runs": [
            {
                "learning_rate": config.learning_rate,
                "best_epoch": result.best_epoch,
                "val_macro_f1": result.best_val_macro_f1,
            }
            for config, result in runs
        ],
        "chosen": asdict(best_config),
        "test_accuracy": report.metrics.accuracy,
        "test_macro_f1": report.metrics.macro_f1,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.domain}: chosen lr={best_config.learning_rate}; "
        f"test accuracy {report.metrics.accuracy:.4f} macro F1 {report.metrics.macro_f1:.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
