"""Tune the LSTM on the validation split, then score the official test file once.

Usage: python -m scripts.train_lstm --domain laptop

For one domain it trains every setting in a small grid, keeps the one with the best
validation macro F1 and only then reads the test file. The ``conflict`` label is dropped,
as in the usual SemEval-2014 protocol. Everything is written to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from absa.config import DATA_PROCESSED_DIR, OUTPUTS_DIR, processed_filename
from absa.data.validation import read_jsonl
from absa.models.lstm_evaluation import evaluate_test_set
from absa.models.lstm_training import LstmConfig, load_checkpoint, run_training


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="train_lstm", description=__doc__)
    parser.add_argument("--domain", required=True, choices=("laptop", "restaurant"))
    parser.add_argument("--data-dir", type=Path, default=DATA_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-3, 3e-3])
    parser.add_argument("--dropout", type=float, nargs="+", default=[0.0, 0.3, 0.5])
    parser.add_argument("--weighting", choices=("off", "on", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    return parser


def _grid(args: argparse.Namespace) -> list[LstmConfig]:
    weightings = {"off": (False,), "on": (True,), "both": (False, True)}[args.weighting]
    return [
        LstmConfig(
            embedding_dim=100,
            hidden_size=hidden,
            dropout=dropout,
            learning_rate=lr,
            batch_size=32,
            epochs=args.epochs,
            class_weighting=weighting,
            device=args.device,
            drop_conflict=True,
        )
        for hidden, lr, weighting, dropout in itertools.product(
            args.hidden, args.lr, weightings, args.dropout
        )
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid, evaluate the best setting on the test file and write the summary."""
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or OUTPUTS_DIR / f"lstm_{args.domain}"
    train_records = read_jsonl(args.data_dir / processed_filename(args.domain, "train"))

    runs = []
    for index, config in enumerate(_grid(args)):
        result = run_training(train_records, config, output_dir=output_dir / f"run_{index}")
        runs.append((config, result))
        print(f"run {index}: {asdict(config)} val macro F1 {result.best_val_macro_f1:.4f}")
    best_config, best = max(runs, key=lambda run: run[1].best_val_macro_f1)

    model, vocab = load_checkpoint(best.checkpoint_path)
    test_records = [
        record
        for record in read_jsonl(args.data_dir / processed_filename(args.domain, "test"))
        if record["sentiment"] != "conflict"
    ]
    report = evaluate_test_set(
        model, vocab, test_records, labels=best_config.metric_labels, device=args.device
    )
    report.save(output_dir / "test_report.json")
    summary = {
        "domain": args.domain,
        "train_records": len(train_records),
        "test_records": len(test_records),
        "runs": [
            {
                "config": asdict(config),
                "best_epoch": result.best_epoch,
                "val_macro_f1": result.best_val_macro_f1,
            }
            for config, result in runs
        ],
        "chosen": asdict(best_config),
        "checkpoint": str(best.checkpoint_path),
        "test_accuracy": report.metrics.accuracy,
        "test_macro_f1": report.metrics.macro_f1,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.domain}: chosen hidden={best_config.hidden_size} lr={best_config.learning_rate} "
        f"dropout={best_config.dropout} weighted={best_config.class_weighting}; "
        f"test accuracy {report.metrics.accuracy:.4f} macro F1 {report.metrics.macro_f1:.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
