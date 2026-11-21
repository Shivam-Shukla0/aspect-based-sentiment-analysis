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
    gold_left = [span for span in gold if span not in pred]
    pred_left = [span for span in pred if span not in gold]
    errors: list[SpanError] = []
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
    errors.extend(SpanError("missed", text, text[s:e], "") for s, e in gold_left)
    errors.extend(SpanError("spurious", text, "", text[s:e]) for s, e in pred_left)
    return errors


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
        "error_examples": [e.to_dict() for e in errors[:max_examples]],
    }
