"""Regression tests for the lazy-import fix in ``absa.data`` (and ``absa._lazy``).

These simulate lxml being genuinely absent by making ``import lxml`` fail
inside a fresh subprocess (``sys.modules["lxml"] = None`` is the standard
trick: any subsequent import of that name raises ``ModuleNotFoundError``).
A subprocess is used, not an in-process mock, so an lxml copy already
cached by this test session's own imports cannot hide a regression.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from absa._lazy import lazy_getattr


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 (fixed, hardcoded script below; nothing user-controlled)
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


BLOCK_LXML = "import sys; sys.modules['lxml'] = None\n"


def test_lightweight_import_works_without_lxml() -> None:
    result = _run(
        BLOCK_LXML + "import absa.data\n"
        "record = absa.data.validate_record(\n"
        "    {'review': 'x', 'aspect': 'y', 'sentiment': 'positive'}\n"
        ")\n"
        "print(record)\n"
    )
    assert result.returncode == 0, result.stderr
    assert "'review': 'x'" in result.stdout


def test_importing_absa_data_does_not_import_xml_parser_without_lxml() -> None:
    result = _run(
        BLOCK_LXML + "import sys\nimport absa.data\nprint('absa.data.xml_parser' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_calling_xml_parser_without_lxml_raises_clear_error() -> None:
    result = _run(
        BLOCK_LXML + "import absa.data\nabsa.data.parse_laptop_xml('does-not-matter.xml')\n"
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr


def test_existing_public_imports_still_work_with_lxml_present() -> None:
    """``from absa.data import X`` must behave exactly as before this fix, lxml installed."""
    pytest.importorskip("lxml")
    from absa.data import ParsedSentence, parse_laptop_xml  # noqa: PLC0415 (needs the skip above)

    assert callable(parse_laptop_xml)
    assert isinstance(ParsedSentence, type)


def test_lazy_getattr_resolves_a_registered_name() -> None:
    pytest.importorskip("lxml")
    from absa.data.xml_parser import parse_laptop_xml  # noqa: PLC0415 (needs the skip above)

    registry = {"parse_laptop_xml": "absa.data.xml_parser"}
    resolved = lazy_getattr("absa.data", registry, "parse_laptop_xml")
    assert resolved is parse_laptop_xml


def test_lazy_getattr_rejects_an_unregistered_name() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        lazy_getattr("absa.data", {"parse_laptop_xml": "absa.data.xml_parser"}, "missing")
