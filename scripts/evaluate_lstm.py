"""Evaluate a trained LSTM checkpoint on a processed JSONL test file.

Usage: python -m scripts.evaluate_lstm CHECKPOINT TEST_FILE [--output REPORT_JSON]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from absa.data.validation import read_jsonl
from absa.models.lstm_evaluation import evaluate_test_set
from absa.models.lstm_training import load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="evaluate_lstm", description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="checkpoint saved by run_training")
    parser.add_argument("test_file", type=Path, help="processed JSONL test file")
    parser.add_argument("--output", type=Path, default=None, help="optional report JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate the checkpoint on the test file, optionally save a report, and print it."""
    args = build_parser().parse_args(argv)
    # The checkpoint carries the weights, vocabulary and config, so no other input is needed.
    model, vocab = load_checkpoint(args.checkpoint)
    records = read_jsonl(args.test_file)
    report = evaluate_test_set(model, vocab, records)
    if args.output is not None:
        report.save(args.output)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
