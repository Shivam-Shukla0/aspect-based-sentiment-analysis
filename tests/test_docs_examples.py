import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_documented_extraction_spans_match_their_text() -> None:
    readme = (ROOT / "data" / "README.md").read_text(encoding="utf-8")
    records = [json.loads(block) for block in re.findall(r"```json\n(.*?)```", readme, re.DOTALL)]
    checked = 0
    for record in records:
        for span in record.get("spans", []):
            assert record["text"][span["start"] : span["end"]] == span["term"]
            checked += 1
    assert checked > 0
