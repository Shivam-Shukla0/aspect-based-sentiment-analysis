"""Project-wide constants shared by all team members."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DOMAINS = ("laptop", "restaurant")
SPLITS = ("train", "val", "test")

# SemEval-2014 Task 4 polarity labels. "conflict" is kept in the schema;
# experiments that drop it must say so explicitly (see data/README.md).
SENTIMENT_LABELS = ("positive", "negative", "neutral", "conflict")

RANDOM_SEED = 42


def processed_filename(domain: str, split: str) -> str:
    """Return the canonical processed-file name for a domain/split pair.

    For example ``processed_filename("laptop", "train")`` is
    ``"laptop_train.jsonl"``. One aspect record is stored per line.
    """
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return f"{domain}_{split}.jsonl"
