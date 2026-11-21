import csv
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
from scripts import compare_results as cmp

PRF = {"precision": 0.8, "recall": 0.6, "f1": 0.686, "tp": 1, "fp": 1, "fn": 1}


def _write(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _outputs(tmp_path: Path, *, with_llm: bool = True) -> Path:
    for domain in ("laptop", "restaurant"):
        for folder in ("lstm", "bert"):
            summary = {"test_accuracy": 0.7, "test_macro_f1": 0.6}
            _write(tmp_path / f"{folder}_{domain}" / "summary.json", summary)
        extractor = {"test_exact": PRF, "end_to_end": PRF}
        _write(tmp_path / f"extractor_{domain}" / "summary.json", extractor)
        if with_llm:
            llm = {"extraction_exact": PRF, "end_to_end": PRF}
            _write(tmp_path / f"llm_{domain}" / "summary.json", llm)
    return tmp_path


def test_every_row_names_its_source_file(tmp_path: Path) -> None:
    rows, missing = cmp.collect(_outputs(tmp_path))
    assert missing == []
    assert {r["setting"] for r in rows} == {"given aspect", "extraction", "end to end"}
    assert all(r["source"].endswith("summary.json") for r in rows)
    sources = {r["source"] for r in rows if r["approach"] == "BERT classifier"}
    assert len(sources) == 2


def test_missing_results_are_listed_and_left_out(tmp_path: Path) -> None:
    rows, missing = cmp.collect(_outputs(tmp_path, with_llm=False))
    assert len(missing) == 2
    assert all("llm_" in path for path in missing)
    assert not any(r["approach"] == "Prompted language model" for r in rows)
    text = cmp.to_markdown(rows, missing)
    assert "## Missing result files" in text


def test_markdown_has_one_table_per_setting(tmp_path: Path) -> None:
    rows, missing = cmp.collect(_outputs(tmp_path))
    text = cmp.to_markdown(rows, missing)
    for heading in ("## Given aspect", "## Extraction", "## End to end"):
        assert heading in text
    assert "| BERT extractor + BERT classifier | laptop | 0.8000 | 0.6000 | 0.6860 |" in text


def test_main_writes_csv_markdown_and_charts(tmp_path: Path) -> None:
    outputs = _outputs(tmp_path / "outputs")
    assert cmp.main(["--outputs-dir", str(outputs)]) == 0
    out = outputs / "comparison"
    with (out / "results.csv").open(encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == list(cmp.COLUMNS)
    assert (out / "results.md").exists()
    assert sorted(p.name for p in out.glob("*.png")) == [
        "end_to_end.png",
        "extraction.png",
        "given_aspect.png",
    ]


def test_main_fails_without_any_result(tmp_path: Path) -> None:
    assert cmp.main(["--outputs-dir", str(tmp_path)]) == 1
