"""Validate a processed JSONL file and print a machine-readable summary.

Usage: python -m scripts.validate_processed FILE [--output REPORT_JSON]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from absa.data.validation import read_jsonl, validate_records


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="validate_processed", description=__doc__)
    parser.add_argument("file", type=Path, help="processed JSONL file")
    parser.add_argument("--output", type=Path, default=None, help="optional report JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the given file, optionally save a report, and return the exit code."""
    args = build_parser().parse_args(argv)
    summary = validate_records(read_jsonl(args.file))
    if args.output is not None:
        summary.save(args.output)
    print(json.dumps(summary.to_dict(), indent=2))
    return 1 if summary.errors else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
