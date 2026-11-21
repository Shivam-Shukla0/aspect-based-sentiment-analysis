import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from absa.extraction import Sentence
from absa.models.llm import MockLLMClient
from scripts import benchmark_llm as bench

TEXT = "The battery life is great but the screen is dim."
SENTENCES = [
    Sentence("1", TEXT, ((4, 16, "positive"), (34, 40, "negative"))),
    Sentence("2", "Nothing to say."),
    Sentence("3", "Fast boot but loud fan.", ((0, 9, "positive"), (14, 22, "negative"))),
]
GOOD = (
    '[{"aspect": "battery life", "sentiment": "positive"},'
    ' {"aspect": "screen", "sentiment": "negative"}]'
)


def _generate(answers: list[str]) -> Callable[[Sequence[str]], list[str]]:
    iterator = iter(answers)
    return lambda prompts: [next(iterator) for _ in prompts]


def test_perfect_answers_score_one_and_empty_list_means_no_aspects() -> None:
    client = _generate([GOOD, "[]", '[{"aspect": "Fast boot", "sentiment": "positive"}]'])
    answers = bench.collect_answers(SENTENCES, client, lambda review: review, batch_size=2)
    result = bench.score(SENTENCES, answers)
    assert result["unparsable_answers"] == 0
    assert result["extraction_exact"]["tp"] == 3  # type: ignore[index]
    assert result["end_to_end"]["tp"] == 3  # type: ignore[index]
    assert result["error_counts"] == {"boundary": 0, "missed": 1, "spurious": 0}


def test_unparsable_and_invalid_answers_are_counted_not_hidden() -> None:
    bad_label = (
        '[{"aspect": "screen", "sentiment": "amazing"},'
        ' {"aspect": "ghost", "sentiment": "positive"}]'
    )
    client = _generate(["I cannot help with that.", bad_label, "[]"])
    answers = bench.collect_answers(SENTENCES, client, lambda review: review, batch_size=3)
    result = bench.score(SENTENCES, answers)
    assert result["unparsable_answers"] == 1
    assert result["invalid_aspects"] == 2
    assert result["end_to_end"]["tp"] == 0  # type: ignore[index]
    assert result["extraction_exact"]["fn"] == 4  # type: ignore[index]


def test_gold_conflict_is_left_out_of_the_end_to_end_gold() -> None:
    sentence = Sentence("9", TEXT, ((4, 16, "conflict"),))
    answers = bench.collect_answers(
        [sentence], _generate(['[{"aspect": "battery life", "sentiment": "positive"}]']), str, 1
    )
    result = bench.score([sentence], answers)
    assert result["extraction_exact"]["tp"] == 1  # type: ignore[index]
    assert result["end_to_end"]["fn"] == 0  # type: ignore[index]
    assert result["end_to_end"]["fp"] == 1  # type: ignore[index]


def test_mock_client_works_with_the_batch_interface() -> None:
    client = MockLLMClient(default_response=GOOD)
    answers = bench.collect_answers(
        SENTENCES[:1], lambda ps: [client.generate(p) for p in ps], str, 4
    )
    assert answers[0].pairs == ((4, 16, "positive"), (34, 40, "negative"))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_script_replays_saved_responses_and_writes_the_summary(tmp_path: Path) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    _write(
        data / "laptop_test.jsonl",
        [
            {
                "sentence_id": "1",
                "review": TEXT,
                "aspect": "battery life",
                "start": 4,
                "end": 16,
                "sentiment": "positive",
            },
            {
                "sentence_id": "1",
                "review": TEXT,
                "aspect": "screen",
                "start": 34,
                "end": 40,
                "sentiment": "negative",
            },
        ],
    )
    _write(
        data / "laptop_test_no_aspect.jsonl", [{"sentence_id": "2", "review": "Nothing to say."}]
    )
    saved = tmp_path / "responses.jsonl"
    _write(
        saved,
        [
            {"sentence_id": "1", "review": TEXT, "raw": GOOD, "latency_seconds": 0.5},
            {"sentence_id": "2", "review": "Nothing to say.", "raw": "[]", "latency_seconds": 0.1},
        ],
    )
    argv = ["--domain", "laptop", "--data-dir", str(data), "--output-dir", str(out)]
    assert bench.main([*argv, "--from-responses", str(saved)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["end_to_end"]["f1"] == pytest.approx(1.0)
    assert summary["prompt_version"] == "v1.0"
    assert summary["sentences"] == 2
    assert "not applicable" in summary["cost"]
    assert bench.main([*argv, "--from-responses", str(saved), "--limit", "1"]) == 0
    assert (out / "test_report.json").exists()


def test_save_then_replay_round_trips(tmp_path: Path) -> None:
    answers = bench.collect_answers(SENTENCES[:1], _generate([GOOD]), str, 1)
    path = tmp_path / "r.jsonl"
    bench._save(path, SENTENCES[:1], answers)
    again = bench._from_saved(SENTENCES[:1], path)
    assert again[0].pairs == answers[0].pairs
    assert again[0].raw == GOOD
