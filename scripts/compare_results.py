"""Collect the saved result files of every approach into comparison tables and charts.

Usage: python -m scripts.compare_results [--outputs-dir outputs] [--output-dir outputs/comparison]

It reads ``summary.json`` from ``lstm_<domain>``, ``bert_<domain>``, ``extractor_<domain>`` and
``llm_<domain>`` and writes ``results.csv``, ``results.md`` and one chart per setting. Every row
names the file its numbers come from, so each figure can be traced. The three settings are kept
apart because they answer different questions: sentiment of a supplied aspect (LSTM, BERT),
finding aspects (extractor, language model) and finding them with their sentiment end to end.
Approaches or domains without a result file are left out and listed as missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from absa.config import DOMAINS, OUTPUTS_DIR

COLUMNS = ("setting", "approach", "domain", "metric", "value", "source")
Row = dict[str, str]


def _load(path: Path) -> Mapping[str, object] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _prf(section: object) -> dict[str, float]:
    values = cast("Mapping[str, float]", section)
    return {
        "precision": float(values["precision"]),
        "recall": float(values["recall"]),
        "F1": float(values["f1"]),
    }


def _rows(
    setting: str, approach: str, domain: str, source: Path, values: dict[str, float]
) -> list[Row]:
    return [
        {
            "setting": setting,
            "approach": approach,
            "domain": domain,
            "metric": metric,
            "value": f"{value:.4f}",
            "source": source.as_posix(),
        }
        for metric, value in values.items()
    ]


def collect(outputs_dir: Path) -> tuple[list[Row], list[str]]:
    """Return the result rows and the list of result files that were not found."""
    rows: list[Row] = []
    missing: list[str] = []
    for domain in DOMAINS:
        for approach, folder in (("LSTM with attention", "lstm"), ("BERT classifier", "bert")):
            path = outputs_dir / f"{folder}_{domain}" / "summary.json"
            summary = _load(path)
            if summary is None:
                missing.append(path.as_posix())
                continue
            scores = {
                "accuracy": float(str(summary["test_accuracy"])),
                "macro F1": float(str(summary["test_macro_f1"])),
            }
            rows += _rows("given aspect", approach, domain, path, scores)
        for approach, folder, key in (
            ("BERT extractor", "extractor", "test_exact"),
            ("Prompted language model", "llm", "extraction_exact"),
        ):
            path = outputs_dir / f"{folder}_{domain}" / "summary.json"
            summary = _load(path)
            if summary is None:
                missing.append(path.as_posix())
                continue
            rows += _rows("extraction", approach, domain, path, _prf(summary[key]))
            if "end_to_end" in summary:
                label = "BERT extractor + BERT classifier" if folder == "extractor" else approach
                rows += _rows("end to end", label, domain, path, _prf(summary["end_to_end"]))
    return rows, missing


def to_markdown(rows: Sequence[Row], missing: Sequence[str]) -> str:
    """Render the rows as one Markdown table per setting, plus the list of missing files."""
    lines = ["# Results comparison", ""]
    for setting in ("given aspect", "extraction", "end to end"):
        selected = [r for r in rows if r["setting"] == setting]
        if not selected:
            continue
        metrics = list(dict.fromkeys(r["metric"] for r in selected))
        lines += [
            f"## {setting.capitalize()}",
            "",
            "| approach | domain | " + " | ".join(metrics) + " |",
        ]
        lines.append("| --- | --- |" + " --- |" * len(metrics))
        keys = list(dict.fromkeys((r["approach"], r["domain"]) for r in selected))
        for approach, domain in keys:
            cells = [
                next(
                    r["value"]
                    for r in selected
                    if (r["approach"], r["domain"], r["metric"]) == (approach, domain, m)
                )
                for m in metrics
            ]
            lines.append(f"| {approach} | {domain} | " + " | ".join(cells) + " |")
        lines.append("")
    if missing:
        lines += ["## Missing result files", ""] + [f"- `{path}`" for path in missing] + [""]
    return "\n".join(lines)


def write_charts(rows: Sequence[Row], output_dir: Path) -> list[Path]:
    """Write one grouped bar chart per setting (the F1 or macro F1 of every approach)."""
    import matplotlib  # noqa: PLC0415  (only needed for charts)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    written: list[Path] = []
    for setting, metric in (
        ("given aspect", "macro F1"),
        ("extraction", "F1"),
        ("end to end", "F1"),
    ):
        selected = [r for r in rows if r["setting"] == setting and r["metric"] == metric]
        if not selected:
            continue
        approaches = list(dict.fromkeys(r["approach"] for r in selected))
        figure, axis = plt.subplots(figsize=(7, 3.6))
        width = 0.8 / len(DOMAINS)
        for index, domain in enumerate(DOMAINS):
            values = [
                float(
                    next(
                        (
                            r["value"]
                            for r in selected
                            if (r["approach"], r["domain"]) == (a, domain)
                        ),
                        "0",
                    )
                )
                for a in approaches
            ]
            positions = [i + index * width for i in range(len(approaches))]
            axis.bar(positions, values, width, label=domain)
        axis.set_xticks(
            [i + width * (len(DOMAINS) - 1) / 2 for i in range(len(approaches))], approaches
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel(metric)
        axis.set_title(f"{setting.capitalize()}: {metric} on the test file")
        axis.legend()
        figure.autofmt_xdate(rotation=15)
        figure.tight_layout()
        path = output_dir / f"{setting.replace(' ', '_')}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Write the CSV, Markdown and chart files; fail if there is nothing to compare."""
    parser = argparse.ArgumentParser(prog="compare_results", description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.outputs_dir / "comparison"
    rows, missing = collect(args.outputs_dir)
    if not rows:
        print("no result files found", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "results.md").write_text(to_markdown(rows, missing) + "\n", encoding="utf-8")
    charts = write_charts(rows, output_dir)
    print(f"wrote {len(rows)} rows, {len(charts)} charts to {output_dir}; missing: {len(missing)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
