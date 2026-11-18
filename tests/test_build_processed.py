from pathlib import Path

import pytest

from scripts.build_processed import main

XML = """<?xml version="1.0" encoding="UTF-8"?>
<sentences>
    <sentence id="1">
        <text>The battery is weak.</text>
        <aspectTerms>
            <aspectTerm term="battery" polarity="negative" from="4" to="11"/>
        </aspectTerms>
    </sentence>
</sentences>
"""


def test_main_builds_laptop_processed_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "laptops.xml"
    input_path.write_text(XML, encoding="utf-8")
    output_dir = tmp_path / "out"
    exit_code = main(
        [str(input_path), "--domain", "laptop", "--split", "train", "--output-dir", str(output_dir)]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "laptop_train.jsonl" in out
    assert "laptop_train_no_aspect.jsonl" in out
    assert (output_dir / "laptop_train.jsonl").is_file()


def test_main_builds_restaurant_processed_files(tmp_path: Path) -> None:
    input_path = tmp_path / "restaurants.xml"
    input_path.write_text(XML, encoding="utf-8")
    output_dir = tmp_path / "out"
    exit_code = main(
        [
            str(input_path),
            "--domain",
            "restaurant",
            "--split",
            "test",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    assert (output_dir / "restaurant_test.jsonl").is_file()


def test_main_rejects_unknown_domain(tmp_path: Path) -> None:
    input_path = tmp_path / "x.xml"
    input_path.write_text(XML, encoding="utf-8")
    with pytest.raises(SystemExit):
        main([str(input_path), "--domain", "phone", "--split", "train"])
