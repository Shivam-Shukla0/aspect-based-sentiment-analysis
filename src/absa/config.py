"""Project-wide constants shared by all team members."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DOMAINS = ("laptop", "restaurant")

# SemEval-2014 Task 4 polarity labels. "conflict" is kept in the schema;
# experiments that drop it must say so explicitly (see data/README.md).
SENTIMENT_LABELS = ("positive", "negative", "neutral", "conflict")

RANDOM_SEED = 42
