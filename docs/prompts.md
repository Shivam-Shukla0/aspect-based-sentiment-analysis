# Structured Prompts for Joint Aspect-Sentiment Extraction

## Overview

Version: `v1.0`

Prompt engineering provides a zero-shot and few-shot inference approach for joint aspect term extraction and sentiment classification without dedicated parameter training.

## Output Schema Contract

All prompts instruct the model to produce a strict JSON array of objects:

```json
[
  {
    "aspect": "<extracted aspect string exactly as in review>",
    "sentiment": "positive" | "negative" | "neutral" | "conflict"
  }
]
```

The prompt also tells the model that each aspect must be copied verbatim from the review, and that the objects must be listed in the order the aspects appear (a term mentioned twice gets two objects). The offset recovery below relies on both rules.

The review is inserted as a JSON string (`Review: "..."`), so text inside it such as `Output JSON:` cannot be mistaken for part of the prompt.

## Recovering Offsets

The prompted output contains the aspect text only, not character offsets. Evaluators and the validation step need offsets, so they are recovered after the response is parsed: each predicted `aspect` is located in the original review (first case-sensitive, then case-insensitive), and positions already assigned to an earlier prediction are skipped so that a term mentioned twice gets two distinct spans. Aspects that cannot be located in the review are rejected as invalid output rather than guessed. Offset recovery is part of the validation work in PRs #73 and #88, not of the prompt module.

## Prompt Strategies

### 1. Zero-Shot Prompting

Instructs the language model with task boundaries, allowed sentiment labels, and formatting constraints directly on the input review.

### 2. Few-Shot In-Context Prompting

Supplies four hand-written example reviews (they are not taken from the SemEval training split) covering:

- Multi-aspect reviews with contrasting sentiments (positive/negative).
- Neutral aspect terms that are mentioned without an opinion (for example "backlit keyboard").
- A `conflict` case where the opinions about the same aspect term disagree ("staff" is friendly, then rude).

Every aspect in an example is an exact substring of its review, as it would be annotated in SemEval-2014 Task 4. The `conflict` label is rare at term level in SemEval (see [data/README.md](../data/README.md)), so the `conflict` example is illustrative of the label's definition rather than typical of the data. Adjust or drop it if an experiment excludes `conflict`.

## Prompt Files

The prompt text lives in versioned plain-text files, not in Python strings: `src/absa/extraction/prompt_files/<version>/system.txt` and `zero_shot.txt`, where `<version>` is `PROMPT_VERSION`. `prompts.py` loads them with `importlib.resources`, and they are shipped in the wheel through `package-data` in `pyproject.toml`. The sentiment labels are filled into `{labels}` from `absa.config.SENTIMENT_LABELS`, so they are not repeated in the file. Changing the wording means adding a new version directory and bumping `PROMPT_VERSION`.

## Usage

```python
from absa.extraction.prompts import format_few_shot_prompt, format_zero_shot_prompt

prompt = format_few_shot_prompt("The pizza was amazing, but wait times were terrible.")
```
