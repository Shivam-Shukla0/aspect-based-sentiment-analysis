"""Command line interface: ``python -m absa validate [FILE]``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from typing import TextIO

from absa import __version__
from absa.config import SENTIMENT_LABELS
from absa.data import validate_record


def validate_lines(lines: Iterable[str], allowed_labels: Sequence[str]) -> list[str]:
    """Return one error message per invalid JSON Lines record; blank lines are skipped."""
    errors = []
    # start=1 so messages use human-friendly 1-based line numbers, like an editor shows.
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue  # blank lines (e.g. a trailing newline) are not records
        try:
            validate_record(json.loads(line), allowed_labels)
        except ValueError as exc:  # json.JSONDecodeError is a ValueError
            errors.append(f"line {number}: {exc}")
    return errors


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="absa", description=__doc__)
    parser.add_argument("--version", action="version", version=f"absa {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate JSON Lines aspect records")
    # No file given -> read from stdin, so the command works in shell pipelines.
    validate.add_argument("file", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    validate.add_argument("--drop-conflict", action="store_true", help="reject 'conflict' labels")
    return parser


def main(argv: Sequence[str] | None = None, stderr: TextIO = sys.stderr) -> int:
    """Run the CLI and return the process exit code."""
    args = build_parser().parse_args(argv)
    # --drop-conflict removes "conflict" from the accepted labels for experiments that
    # exclude that class; every other label stays valid.
    labels = [
        label for label in SENTIMENT_LABELS if not (args.drop_conflict and label == "conflict")
    ]
    with args.file as handle:
        errors = validate_lines(handle, labels)
    for error in errors:
        print(error, file=stderr)
    if not errors:
        sys.stdout.write("ok\n")
    # Conventional exit codes: 0 = all records valid, 1 = at least one invalid record.
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
