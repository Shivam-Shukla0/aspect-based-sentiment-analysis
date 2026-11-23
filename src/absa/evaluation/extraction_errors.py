"""Error analysis for extracted aspect spans: exact and overlap scores, boundary errors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from absa.evaluation.extraction import evaluate_extraction_spans

Span = tuple[int, int]


@dataclass(frozen=True)
class SpanError:
    """One extraction mistake: a boundary error, a missed span or a spurious span."""

    kind: str
    text: str
    gold: str
    predicted: str

    def to_dict(self) -> dict[str, str]:
        """Return the error as a dictionary."""
        return {
            "kind": self.kind,
            "text": self.text,
            "gold": self.gold,
            "predicted": self.predicted,
        }


def _classify_errors(text: str, gold: Sequence[Span], pred: Sequence[Span]) -> list[SpanError]:
    """Return the errors for one sentence; exact matches are not errors."""
    # Exact matches are correct and drop out; only the leftovers can be errors.
    gold_left = [span for span in gold if span not in pred]
    pred_left = [span for span in pred if span not in gold]
    errors: list[SpanError] = []
    # Copy with list(): pred_left is mutated inside the loop. A leftover prediction that
    # overlaps a leftover gold span is a "boundary" error (right place, wrong edges).
    for p_start, p_end in list(pred_left):
        overlapping = next((g for g in gold_left if max(g[0], p_start) < min(g[1], p_end)), None)
        if overlapping is not None:
            gold_left.remove(overlapping)
            pred_left.remove((p_start, p_end))
            errors.append(
                SpanError(
                    "boundary",
                    text,
                    text[overlapping[0] : overlapping[1]],
                    text[p_start:p_end],
                )
            )
    # Whatever is still unpaired: gold with no prediction = missed, prediction with no
    # gold = spurious.
    errors.extend(SpanError("missed", text, text[s:e], "") for s, e in gold_left)
    errors.extend(SpanError("spurious", text, "", text[s:e]) for s, e in pred_left)
    return errors


# Error types analysed: boundary (prediction overlaps the gold span but is not equal), missed
# (gold aspect not predicted) and spurious (predicted aspect not in gold). Counts as
# boundary/missed/spurious: extractor 70/89/49 on laptops and 73/51/85 on restaurants; LLM
# 95/182/404 and 83/269/238.
#
# End-to-end results and their limit: extractor + BERT classifier F1 0.591 (laptop), 0.726
# (restaurant). Extraction is the bottleneck: every missed or wrongly bounded span is lost
# before classification. The extractor's overlap F1 (0.891/0.941) is far above its exact F1,
# so many errors are boundary errors ("price" instead of "price tag").
def span_report(
    texts: Sequence[str],
    gold: Sequence[Sequence[Span]],
    predicted: Sequence[Sequence[Span]],
    max_examples: int = 60,
) -> dict[str, object]:
    """Score predicted spans: exact and overlap metrics, error counts and examples.

    Exact match is the headline figure; overlap match shows how much of the loss is only a
    boundary problem. Errors are boundary (overlapping but not equal), missed and spurious.
    """
    # Sort spans so greedy overlap matching is reproducible (see compute_span_counts).
    pred = [sorted(spans) for spans in predicted]
    exact = evaluate_extraction_spans(gold, pred, exact_match=True)
    overlap = evaluate_extraction_spans(gold, pred, exact_match=False)
    errors = [
        error
        for text, gold_spans, spans in zip(texts, gold, pred, strict=True)
        for error in _classify_errors(text, sorted(gold_spans), spans)
    ]
    counts = {
        kind: sum(1 for e in errors if e.kind == kind)
        for kind in ("boundary", "missed", "spurious")
    }
    return {
        "sentences": len(texts),
        "gold_spans": sum(len(g) for g in gold),
        "predicted_spans": sum(len(p) for p in pred),
        "exact": exact.to_dict(),
        "overlap": overlap.to_dict(),
        "error_counts": counts,
        # Cap the examples so the report stays small even for large evaluation sets.
        "error_examples": [e.to_dict() for e in errors[:max_examples]],
    }
