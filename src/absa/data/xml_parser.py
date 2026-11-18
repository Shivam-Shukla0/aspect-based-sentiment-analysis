"""Shared SemEval-2014 Task 4 XML parsing, used by the laptop and restaurant parsers.

See ``docs/dataset_annotation.md`` for the annotation format this follows:
``from``/``to`` are zero-based, end-exclusive character offsets into
``<text>``, and a sentence with no ``<aspectTerms>`` element has no aspect
annotation at all (as opposed to an empty one).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from absa.data.schema import AspectSpan, ExtractionRecord


@dataclass(frozen=True, slots=True)
class ParsedSentence:
    """One parsed ``<sentence>`` element: its id, text and term-level spans."""

    sentence_id: str
    text: str
    spans: tuple[AspectSpan, ...]

    def __post_init__(self) -> None:
        """Validate ``text``/``spans`` consistency via the shared extraction contract."""
        ExtractionRecord(text=self.text, spans=self.spans)


def _parse_tree(path: Path) -> etree._Element:
    """Parse ``path`` with entity resolution and network access disabled."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    return etree.parse(str(path), parser).getroot()


def _require_attr(element: etree._Element, name: str, context: str) -> str:
    value = element.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{context}: missing required attribute {name!r}")
    return value


def _parse_spans(element: etree._Element, sentence_id: str) -> tuple[AspectSpan, ...]:
    terms_element = element.find("aspectTerms")
    if terms_element is None:
        return ()
    spans = []
    for term in terms_element.findall("aspectTerm"):
        context = f"<sentence id={sentence_id!r}> aspectTerm"
        term_text = _require_attr(term, "term", context)
        polarity = _require_attr(term, "polarity", context)
        from_text = _require_attr(term, "from", context)
        to_text = _require_attr(term, "to", context)
        try:
            start, end = int(from_text), int(to_text)
        except ValueError as exc:
            raise ValueError(
                f"{context}: offsets must be integers, got {from_text!r}/{to_text!r}"
            ) from exc
        try:
            spans.append(AspectSpan(term=term_text, start=start, end=end, polarity=polarity))
        except ValueError as exc:
            raise ValueError(f"{context}: {exc}") from exc
    return tuple(spans)


def _parse_sentence(element: etree._Element) -> ParsedSentence:
    sentence_id = _require_attr(element, "id", "<sentence>")
    text_element = element.find("text")
    if text_element is None or text_element.text is None:
        raise ValueError(f"<sentence id={sentence_id!r}>: missing <text>")
    spans = _parse_spans(element, sentence_id)
    try:
        return ParsedSentence(sentence_id=sentence_id, text=text_element.text, spans=spans)
    except ValueError as exc:
        raise ValueError(f"<sentence id={sentence_id!r}>: {exc}") from exc


def parse_laptop_xml(path: Path) -> list[ParsedSentence]:
    """Parse a laptop-domain SemEval-2014 XML file into one ``ParsedSentence`` per sentence.

    The laptop file has no ``<aspectCategories>`` element; only term-level
    annotations are parsed, which is all this domain provides.
    """
    return [_parse_sentence(element) for element in _parse_tree(path).findall("sentence")]
