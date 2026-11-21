"""Build processed JSONL files from one raw SemEval XML file.

Usage: python -m scripts.build_processed INPUT_XML --domain DOMAIN --split SPLIT
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from absa.config import DATA_PROCESSED_DIR
from absa.data.normalize import build_processed_files
from absa.data.xml_parser import parse_laptop_xml, parse_restaurant_xml


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="build_processed", description=__doc__)
    parser.add_argument("input", type=Path, help="raw SemEval XML file")
    parser.add_argument("--domain", required=True, choices=("laptop", "restaurant"))
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--output-dir", type=Path, default=DATA_PROCESSED_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, normalize and write the processed files; return the process exit code."""
    args = build_parser().parse_args(argv)
    sentences = (
        parse_laptop_xml(args.input)
        if args.domain == "laptop"
        else parse_restaurant_xml(args.input)[0]
    )
    records_path, no_aspect_path = build_processed_files(
        sentences, args.domain, args.split, args.output_dir
    )
    print(f"wrote {records_path}")
    print(f"wrote {no_aspect_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
