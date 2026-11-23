"""Benchmark prompted language-model extraction on the SemEval-2014 test file.

Usage: python -m scripts.benchmark_llm --domain laptop --model-name Qwen/Qwen2.5-7B-Instruct

Every test sentence is sent once with the versioned few-shot (or zero-shot) prompt. The raw
answers are saved to ``responses.jsonl``, so the scores can be recomputed without the model
(``--from-responses``). Answers that are not valid JSON, and predicted aspects that are not in
the review or carry an unknown label, are counted as invalid. The summary records the prompt
version, the model name, the decoding settings and the measured latency; no money is spent, so
cost is marked not applicable. Extraction is scored by exact span match and end to end by span
and label; gold ``conflict`` aspects are left out of the end-to-end gold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from absa.config import DATA_PROCESSED_DIR, OUTPUTS_DIR, processed_filename
from absa.data.validation import read_jsonl
from absa.evaluation import evaluate_end_to_end, span_report
from absa.extraction import Sentence, group_sentences
from absa.extraction.prompts import PROMPT_VERSION, format_few_shot_prompt, format_zero_shot_prompt
from absa.models.llm import parse_llm_json
from absa.models.llm_validation import ValidationReport, validate_aspect_predictions

# (start_char, end_char, sentiment), the same unit used by end-to-end evaluation.
Pair = tuple[int, int, str]


@dataclass(frozen=True)
class Answer:
    """The model's answer for one sentence and what was made of it."""

    sentence_id: str
    raw: str
    latency_seconds: float
    pairs: tuple[Pair, ...]
    parse_failed: bool
    invalid_aspects: int


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="benchmark_llm", description=__doc__)
    parser.add_argument("--domain", required=True, choices=("laptop", "restaurant"))
    parser.add_argument("--data-dir", type=Path, default=DATA_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt", choices=("few-shot", "zero-shot"), default="few-shot")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N sentences")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--from-responses", type=Path, default=None)
    return parser


def interpret(sentence: Sentence, raw: str) -> tuple[tuple[Pair, ...], bool, int]:
    """Parse and validate one answer; return the pairs, whether parsing failed, invalid count."""
    # Two-stage check: (1) is the reply valid JSON of the expected shape? (2) do the aspects
    # really occur in the sentence with a known label? Failures are counted, never hidden.
    try:
        parsed = parse_llm_json(raw)
    except ValueError:
        return (), True, 0
    report = ValidationReport()
    valid = validate_aspect_predictions(sentence.text, parsed, report)
    # Sorted by position so scoring does not depend on the order the model listed aspects.
    pairs = tuple(sorted((v.start, v.end, v.sentiment) for v in valid))
    return pairs, False, report.invalid_count


def collect_answers(
    sentences: Sequence[Sentence],
    generate_batch: Callable[[Sequence[str]], list[str]],
    prompt_for: Callable[[str], str],
    batch_size: int,
) -> list[Answer]:
    """Send every sentence to the model in batches and interpret the answers."""
    answers: list[Answer] = []
    for offset in range(0, len(sentences), batch_size):
        chunk = sentences[offset : offset + batch_size]
        # perf_counter is a monotonic high-resolution clock, right for measuring durations.
        started = time.perf_counter()
        raws = generate_batch([prompt_for(s.text) for s in chunk])
        # Batch time divided by batch size = the average per-sentence latency.
        latency = (time.perf_counter() - started) / len(chunk)
        for sentence, raw in zip(chunk, raws, strict=True):
            pairs, failed, invalid = interpret(sentence, raw)
            answers.append(Answer(sentence.sentence_id, raw, latency, pairs, failed, invalid))
    return answers


def score(sentences: Sequence[Sentence], answers: Sequence[Answer]) -> dict[str, object]:
    """Score extraction and end-to-end results and count invalid output."""
    # Same answers scored two ways: spans only (extraction) and span+label (end to end).
    spans = [[(start, end) for start, end, _ in a.pairs] for a in answers]
    report = span_report([s.text for s in sentences], [s.gold_spans for s in sentences], spans)
    # Gold "conflict" aspects are excluded from the end-to-end gold (see module docstring).
    gold_pairs = [[p for p in s.spans if p[2] != "conflict"] for s in sentences]
    end_to_end = evaluate_end_to_end(gold_pairs, [list(a.pairs) for a in answers])
    # Sorted so the median can be read from the middle element.
    latencies = sorted(a.latency_seconds for a in answers)
    return {
        "extraction_exact": report["exact"],
        "extraction_overlap": report["overlap"],
        "end_to_end": end_to_end.to_dict(),
        "error_counts": report["error_counts"],
        "error_examples": report["error_examples"],
        "unparsable_answers": sum(a.parse_failed for a in answers),
        "invalid_aspects": sum(a.invalid_aspects for a in answers),
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "median_latency_seconds": latencies[len(latencies) // 2] if latencies else 0.0,
    }


def _save(path: Path, sentences: Sequence[Sentence], answers: Sequence[Answer]) -> None:
    lines = [
        json.dumps(
            {
                "sentence_id": s.sentence_id,
                "review": s.text,
                "raw": a.raw,
                "latency_seconds": a.latency_seconds,
            }
        )
        for s, a in zip(sentences, answers, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _from_saved(sentences: Sequence[Sentence], path: Path) -> list[Answer]:
    # Index saved answers by (id, text) so replay works even if the order or the subset
    # (--limit) changed. Re-interpreting the raw text lets scoring code be fixed and re-run
    # without calling the model again.
    saved = {(r["sentence_id"], r["review"]): r for r in read_jsonl(path)}
    answers = []
    for sentence in sentences:
        record = saved[(sentence.sentence_id, sentence.text)]
        pairs, failed, invalid = interpret(sentence, str(record["raw"]))
        latency = float(str(record["latency_seconds"]))
        answers.append(
            Answer(sentence.sentence_id, str(record["raw"]), latency, pairs, failed, invalid)
        )
    return answers


def main(argv: Sequence[str] | None = None) -> int:
    """Run (or replay) the benchmark and write responses, summary and report."""
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or OUTPUTS_DIR / f"llm_{args.domain}"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.data_dir / processed_filename(args.domain, "test"))
    no_aspect_path = args.data_dir / f"{args.domain}_test_no_aspect.jsonl"
    sentences = group_sentences(
        records, read_jsonl(no_aspect_path) if no_aspect_path.exists() else []
    )
    if args.limit:
        sentences = sentences[: args.limit]

    # Two modes: replay saved answers (no GPU, no model) or generate fresh ones.
    if args.from_responses:
        answers = _from_saved(sentences, args.from_responses)
    else:
        # Imported only on this branch so replay mode works without torch installed.
        from absa.models.llm_local import LocalLLMClient  # noqa: PLC0415  (needs torch)

        client = LocalLLMClient(args.model_name, args.device, args.max_new_tokens)
        prompt_for = (
            format_few_shot_prompt if args.prompt == "few-shot" else format_zero_shot_prompt
        )
        answers = collect_answers(sentences, client.generate_batch, prompt_for, args.batch_size)
        _save(output_dir / "responses.jsonl", sentences, answers)

    result = score(sentences, answers)
    # Error details go to test_report.json; the compact numbers go to summary.json.
    report = {k: result[k] for k in ("error_counts", "error_examples")}
    (output_dir / "test_report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    summary = {
        "domain": args.domain,
        "approach": "prompted language model",
        "model": args.model_name,
        "prompt": args.prompt,
        "prompt_version": PROMPT_VERSION,
        "decoding": {"greedy": True, "max_new_tokens": args.max_new_tokens},
        "sentences": len(sentences),
        "cost": "not applicable (open-weights model run on a Colab GPU)",
        **{k: v for k, v in result.items() if k not in report},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print(
        f"{args.domain}: extraction F1 {summary['extraction_exact']['f1']:.4f}, "
        f"end-to-end F1 {summary['end_to_end']['f1']:.4f}, "
        f"unparsable {summary['unparsable_answers']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
