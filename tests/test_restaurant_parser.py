from pathlib import Path

import pytest

pytest.importorskip("lxml")

from absa.data import parse_restaurant_xml

NORMAL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sentences>
    <sentence id="1">
        <text>The food was great but the service was slow.</text>
        <aspectTerms>
            <aspectTerm term="food" polarity="positive" from="4" to="8"/>
            <aspectTerm term="service" polarity="negative" from="27" to="34"/>
        </aspectTerms>
        <aspectCategories>
            <aspectCategory category="food" polarity="positive"/>
            <aspectCategory category="service" polarity="negative"/>
        </aspectCategories>
    </sentence>
    <sentence id="2">
        <text>Go inside and you won't want to leave.</text>
        <aspectCategories>
            <aspectCategory category="anecdotes/miscellaneous" polarity="positive"/>
        </aspectCategories>
    </sentence>
</sentences>
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_parses_terms_and_categories_separately(tmp_path: Path) -> None:
    sentences, categories = parse_restaurant_xml(_write(tmp_path, "sample.xml", NORMAL_XML))
    assert len(sentences) == 2
    assert len(sentences[0].spans) == 2
    assert len(categories) == 3


def test_sentence_identifiers_are_preserved(tmp_path: Path) -> None:
    sentences, categories = parse_restaurant_xml(_write(tmp_path, "sample.xml", NORMAL_XML))
    assert [s.sentence_id for s in sentences] == ["1", "2"]
    assert {c.sentence_id for c in categories} == {"1", "2"}


def test_categories_are_not_used_as_terms(tmp_path: Path) -> None:
    sentences, categories = parse_restaurant_xml(_write(tmp_path, "sample.xml", NORMAL_XML))
    second_sentence_spans = sentences[1].spans
    assert second_sentence_spans == ()
    second_sentence_categories = [c for c in categories if c.sentence_id == "2"]
    assert len(second_sentence_categories) == 1
    assert second_sentence_categories[0].category == "anecdotes/miscellaneous"


def test_categories_have_no_offsets(tmp_path: Path) -> None:
    _, categories = parse_restaurant_xml(_write(tmp_path, "sample.xml", NORMAL_XML))
    assert not hasattr(categories[0], "start")
    assert not hasattr(categories[0], "end")


def test_sentence_with_terms_but_no_categories(tmp_path: Path) -> None:
    xml = """<sentences>
        <sentence id="1">
            <text>Great food.</text>
            <aspectTerms>
                <aspectTerm term="food" polarity="positive" from="6" to="10"/>
            </aspectTerms>
        </sentence>
    </sentences>"""
    sentences, categories = parse_restaurant_xml(_write(tmp_path, "sample.xml", xml))
    assert len(sentences[0].spans) == 1
    assert categories == []


def test_missing_category_attribute_rejected(tmp_path: Path) -> None:
    xml = """<sentences>
        <sentence id="1">
            <text>Great food.</text>
            <aspectCategories>
                <aspectCategory polarity="positive"/>
            </aspectCategories>
        </sentence>
    </sentences>"""
    with pytest.raises(ValueError, match="missing required attribute 'category'"):
        parse_restaurant_xml(_write(tmp_path, "bad.xml", xml))


def test_runs_on_real_trial_file() -> None:
    path = Path("data/raw/restaurants-trial.xml")
    if not path.is_file():
        pytest.skip(
            "data/raw/restaurants-trial.xml not present (not committed; see data/README.md)"
        )
    sentences, categories = parse_restaurant_xml(path)
    assert len(sentences) == 100
    assert sum(1 for s in sentences if s.spans) == 54
    assert len(categories) == 114
