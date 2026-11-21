"""Normalize parsed SemEval sentences into the common aspect-level record schema.

Categories (restaurant `<aspectCategories>`) are not part of this schema and
are not written here: only given-aspect classification needs this output.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from absa.config import DATA_PROCESSED_DIR, processed_filename
from absa.data.schema import validate_record

if TYPE_CHECKING:
    # Only used in type hints below; kept out of runtime imports so this
    # module (and anything that imports it) does not require lxml just to
    # normalize already-parsed sentences.
    from absa.data.xml_parser import ParsedSentence


def no_aspect_filename(domain: str, split: str) -> str:
    """Return the file name for sentences with no aspect-term annotation."""
    return f"{processed_filename(domain, split).removesuffix('.jsonl')}_no_aspect.jsonl"


def normalize_sentences(
    sentences: Sequence[ParsedSentence], domain: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Convert parsed sentences into the common aspect-level record schema.

    Returns ``(records, no_aspect_sentences)``. ``records`` has one entry
    per (review, aspect) pair, each validated against
    ``absa.data.validate_record`` and extended with ``sentence_id``,
    ``domain``, ``start`` and ``end``. ``no_aspect_sentences`` holds the
    sentences that carried no aspect-term annotation at all, kept rather
    than dropped.
    """
    records: list[dict[str, object]] = []
    no_aspect: list[dict[str, object]] = []
    for sentence in sentences:
        if not sentence.spans:
            no_aspect.append(
                {"sentence_id": sentence.sentence_id, "domain": domain, "review": sentence.text}
            )
            continue
        for span in sentence.spans:
            record: dict[str, object] = {
                "review": sentence.text,
                "aspect": span.term,
                "sentiment": span.polarity,
                "sentence_id": sentence.sentence_id,
                "domain": domain,
                "start": span.start,
                "end": span.end,
            }
            validate_record(record)
            records.append(record)
    return records, no_aspect


def write_jsonl(records: Sequence[dict[str, object]], path: Path) -> None:
    """Write one JSON object per line to ``path`` (sorted keys), creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def build_processed_files(
    sentences: Sequence[ParsedSentence],
    domain: str,
    split: str,
    processed_dir: Path = DATA_PROCESSED_DIR,
) -> tuple[Path, Path]:
    """Normalize ``sentences`` and write both processed JSONL files.

    Returns the ``(records_path, no_aspect_path)`` written. Writing is
    idempotent: running this twice on the same input overwrites each file
    with byte-identical content.
    """
    records, no_aspect = normalize_sentences(sentences, domain)
    records_path = processed_dir / processed_filename(domain, split)
    no_aspect_path = processed_dir / no_aspect_filename(domain, split)
    write_jsonl(records, records_path)
    write_jsonl(no_aspect, no_aspect_path)
    return records_path, no_aspect_path
