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


@dataclass(frozen=True, slots=True)
class CategoryAnnotation:
    """One ``<aspectCategory>`` element: a coarse-grained, span-free category label.

    Restaurant-domain only. A category is not an aspect term: it has no
    offsets into the sentence text and must not be treated as a span.
    """

    sentence_id: str
    category: str
    polarity: str


def _parse_tree(path: Path) -> etree._Element:
    """Parse ``path`` with entity resolution and network access disabled."""
    # Hardening: no entity expansion, no network and no DTD loading, so an untrusted XML
    # file cannot trigger XXE or "billion laughs" style attacks.
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    return etree.parse(str(path), parser).getroot()


def _require_attr(element: etree._Element, name: str, context: str) -> str:
    value = element.get(name)
    # lxml returns None for an absent attribute; report which element/attribute was missing.
    if not isinstance(value, str):
        raise ValueError(f"{context}: missing required attribute {name!r}")
    return value


def _parse_spans(element: etree._Element, sentence_id: str) -> tuple[AspectSpan, ...]:
    terms_element = element.find("aspectTerms")
    # No <aspectTerms> element means "no annotation" (distinct from an empty element).
    if terms_element is None:
        return ()
    spans = []
    for term in terms_element.findall("aspectTerm"):
        context = f"<sentence id={sentence_id!r}> aspectTerm"
        term_text = _require_attr(term, "term", context)
        polarity = _require_attr(term, "polarity", context)
        from_text = _require_attr(term, "from", context)
        to_text = _require_attr(term, "to", context)
        # from/to are XML strings; convert and give a clear error instead of a bare int() failure.
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
    # ParsedSentence validates that every span offset really matches the sentence text.
    try:
        return ParsedSentence(sentence_id=sentence_id, text=text_element.text, spans=spans)
    except ValueError as exc:
        raise ValueError(f"<sentence id={sentence_id!r}>: {exc}") from exc


def _parse_categories(element: etree._Element, sentence_id: str) -> tuple[CategoryAnnotation, ...]:
    categories_element = element.find("aspectCategories")
    if categories_element is None:
        return ()
    context = f"<sentence id={sentence_id!r}> aspectCategory"
    return tuple(
        CategoryAnnotation(
            sentence_id=sentence_id,
            category=_require_attr(category, "category", context),
            polarity=_require_attr(category, "polarity", context),
        )
        for category in categories_element.findall("aspectCategory")
    )


def parse_laptop_xml(path: Path) -> list[ParsedSentence]:
    """Parse a laptop-domain SemEval-2014 XML file into one ``ParsedSentence`` per sentence.

    The laptop file has no ``<aspectCategories>`` element; only term-level
    annotations are parsed, which is all this domain provides.
    """
    return [_parse_sentence(element) for element in _parse_tree(path).findall("sentence")]


def parse_restaurant_xml(path: Path) -> tuple[list[ParsedSentence], list[CategoryAnnotation]]:
    """Parse a restaurant-domain SemEval-2014 XML file.

    Returns term-level sentences (the same shape as ``parse_laptop_xml``,
    sharing its parsing logic) and the coarse-grained category annotations
    separately, since a category is not an aspect term and must not be
    used as one.
    """
    # Parse the tree once and reuse the elements for both terms and categories.
    elements = _parse_tree(path).findall("sentence")
    sentences = [_parse_sentence(element) for element in elements]
    categories = [
        category
        for element in elements
        for category in _parse_categories(element, _require_attr(element, "id", "<sentence>"))
    ]
    return sentences, categories
