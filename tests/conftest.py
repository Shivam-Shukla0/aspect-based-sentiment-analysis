"""Shared pytest fixtures and test configuration."""

from __future__ import annotations

import pytest

SAMPLE_LAPTOP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sentences>
  <sentence id="1">
    <text>The battery life of this machine is fantastic, though the keyboard feels cheap.</text>
    <aspectTerms>
      <aspectTerm term="battery life" polarity="positive" from="4" to="16"/>
      <aspectTerm term="keyboard" polarity="negative" from="58" to="66"/>
    </aspectTerms>
  </sentence>
  <sentence id="2">
    <text>Great performance for daily tasks.</text>
    <aspectTerms>
      <aspectTerm term="performance" polarity="positive" from="6" to="17"/>
    </aspectTerms>
  </sentence>
</sentences>
"""

SAMPLE_RESTAURANT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sentences>
  <sentence id="101">
    <text>The sushi was fresh and delicious, but service was slow.</text>
    <aspectTerms>
      <aspectTerm term="sushi" polarity="positive" from="4" to="9"/>
      <aspectTerm term="service" polarity="negative" from="39" to="46"/>
    </aspectTerms>
    <aspectCategories>
      <aspectCategory category="food" polarity="positive"/>
      <aspectCategory category="service" polarity="negative"/>
    </aspectCategories>
  </sentence>
</sentences>
"""


@pytest.fixture
def synthetic_laptop_xml() -> str:
    """Return a minimal synthetic SemEval-style XML string for laptop reviews."""
    return SAMPLE_LAPTOP_XML


@pytest.fixture
def synthetic_restaurant_xml() -> str:
    """Return a minimal synthetic SemEval-style XML string for restaurant reviews."""
    return SAMPLE_RESTAURANT_XML


@pytest.fixture
def sample_aspect_records() -> list[dict[str, str]]:
    """Return a list of valid aspect record dicts for testing."""
    return [
        {
            "review": "The battery life of this machine is fantastic.",
            "aspect": "battery life",
            "sentiment": "positive",
        },
        {
            "review": "The keyboard feels cheap.",
            "aspect": "keyboard",
            "sentiment": "negative",
        },
    ]
