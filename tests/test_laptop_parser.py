from pathlib import Path

import pytest

pytest.importorskip("lxml")

from absa.data import parse_laptop_xml

NORMAL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sentences>
    <sentence id="1">
        <text>The battery is weak but the screen is excellent.</text>
        <aspectTerms>
            <aspectTerm term="battery" polarity="negative" from="4" to="11"/>
            <aspectTerm term="screen" polarity="positive" from="28" to="34"/>
        </aspectTerms>
    </sentence>
    <sentence id="2">
        <text>Great value for the price.</text>
    </sentence>
</sentences>
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_parses_sentence_with_two_aspect_terms(tmp_path: Path) -> None:
    sentences = parse_laptop_xml(_write(tmp_path, "sample.xml", NORMAL_XML))
    assert len(sentences) == 2
    first = sentences[0]
    assert first.sentence_id == "1"
    assert first.text == "The battery is weak but the screen is excellent."
    assert len(first.spans) == 2
    assert first.spans[0].term == "battery"
    assert first.spans[0].polarity == "negative"
    assert first.spans[1].term == "screen"


def test_sentence_without_aspect_terms_has_empty_spans(tmp_path: Path) -> None:
    sentences = parse_laptop_xml(_write(tmp_path, "sample.xml", NORMAL_XML))
    assert sentences[1].spans == ()


def test_offsets_round_trip_against_text(tmp_path: Path) -> None:
    sentences = parse_laptop_xml(_write(tmp_path, "sample.xml", NORMAL_XML))
    for span in sentences[0].spans:
        assert sentences[0].text[span.start : span.end] == span.term


def test_missing_sentence_id_rejected(tmp_path: Path) -> None:
    xml = """<sentences><sentence><text>No id.</text></sentence></sentences>"""
    with pytest.raises(ValueError, match="missing required attribute 'id'"):
        parse_laptop_xml(_write(tmp_path, "bad.xml", xml))


def test_missing_text_rejected(tmp_path: Path) -> None:
    xml = """<sentences><sentence id="1"></sentence></sentences>"""
    with pytest.raises(ValueError, match="missing <text>"):
        parse_laptop_xml(_write(tmp_path, "bad.xml", xml))


def test_missing_aspect_term_attribute_rejected(tmp_path: Path) -> None:
    xml = """<sentences>
        <sentence id="1">
            <text>The battery is weak.</text>
            <aspectTerms>
                <aspectTerm polarity="negative" from="4" to="11"/>
            </aspectTerms>
        </sentence>
    </sentences>"""
    with pytest.raises(ValueError, match="missing required attribute 'term'"):
        parse_laptop_xml(_write(tmp_path, "bad.xml", xml))


def test_non_integer_offset_rejected(tmp_path: Path) -> None:
    xml = """<sentences>
        <sentence id="1">
            <text>The battery is weak.</text>
            <aspectTerms>
                <aspectTerm term="battery" polarity="negative" from="x" to="11"/>
            </aspectTerms>
        </sentence>
    </sentences>"""
    with pytest.raises(ValueError, match="offsets must be integers"):
        parse_laptop_xml(_write(tmp_path, "bad.xml", xml))


def test_invalid_span_offsets_rejected(tmp_path: Path) -> None:
    xml = """<sentences>
        <sentence id="1">
            <text>The battery is weak.</text>
            <aspectTerms>
                <aspectTerm term="battery" polarity="negative" from="10" to="5"/>
            </aspectTerms>
        </sentence>
    </sentences>"""
    with pytest.raises(ValueError, match="start < end"):
        parse_laptop_xml(_write(tmp_path, "bad.xml", xml))


def test_mismatched_offsets_rejected(tmp_path: Path) -> None:
    xml = """<sentences>
        <sentence id="1">
            <text>The battery is weak.</text>
            <aspectTerms>
                <aspectTerm term="battery" polarity="negative" from="0" to="7"/>
            </aspectTerms>
        </sentence>
    </sentences>"""
    with pytest.raises(ValueError, match="does not match"):
        parse_laptop_xml(_write(tmp_path, "bad.xml", xml))


def test_runs_on_real_trial_file() -> None:
    path = Path("data/raw/laptops-trial.xml")
    if not path.is_file():
        pytest.skip("data/raw/laptops-trial.xml not present (not committed; see data/README.md)")
    sentences = parse_laptop_xml(path)
    assert len(sentences) == 100
    assert sum(1 for s in sentences if s.spans) == 40
