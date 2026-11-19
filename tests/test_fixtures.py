"""Tests for shared pytest fixtures and registered markers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from absa.data.schema import validate_record


def test_synthetic_laptop_xml(synthetic_laptop_xml: str) -> None:
    root = ET.fromstring(synthetic_laptop_xml)  # noqa: S314
    sentences = root.findall("sentence")
    assert len(sentences) == 2
    first_terms = sentences[0].findall("aspectTerms/aspectTerm")
    assert len(first_terms) == 2
    assert first_terms[0].attrib["term"] == "battery life"
    assert first_terms[0].attrib["polarity"] == "positive"


def test_synthetic_restaurant_xml(synthetic_restaurant_xml: str) -> None:
    root = ET.fromstring(synthetic_restaurant_xml)  # noqa: S314
    sentence = root.find("sentence")
    assert sentence is not None
    terms = sentence.findall("aspectTerms/aspectTerm")
    assert len(terms) == 2
    cats = sentence.findall("aspectCategories/aspectCategory")
    assert len(cats) == 2
    assert cats[0].attrib["category"] == "food"


def test_sample_aspect_records(sample_aspect_records: list[dict[str, str]]) -> None:
    assert len(sample_aspect_records) == 2
    for record in sample_aspect_records:
        validated = validate_record(record)
        assert record["aspect"] in str(validated["review"])


@pytest.mark.slow
def test_slow_marker_registration() -> None:
    """Verify that slow marker is registered and does not raise unknown-marker error."""
    assert True


@pytest.mark.gpu
def test_gpu_marker_registration() -> None:
    """Verify that gpu marker is registered and does not raise unknown-marker error."""
    assert True
