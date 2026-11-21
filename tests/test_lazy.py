"""Tests for the shared lazy-attribute helper."""

import pytest

from absa._lazy import lazy_getattr


def test_resolves_a_registered_name_from_its_submodule() -> None:
    assert lazy_getattr("pkg", {"dumps": "json"}, "dumps")({"a": 1}) == '{"a": 1}'


def test_unknown_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'missing'"):
        lazy_getattr("pkg", {}, "missing")
