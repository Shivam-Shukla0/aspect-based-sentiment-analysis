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

## Prompt Strategies

### 1. Zero-Shot Prompting

Instructs the language model with task boundaries, allowed sentiment labels, and formatting constraints directly on the input review.

### 2. Few-Shot In-Context Prompting

Supplies four diverse training examples covering:

- Multi-aspect reviews with contrasting sentiments (positive/negative).
- Neutral everyday operational statements.
- Conflict sentiment cases with mixed impressions on single targets.

## Usage

```python
from absa.extraction.prompts import format_few_shot_prompt, format_zero_shot_prompt

prompt = format_few_shot_prompt("The pizza was amazing, but wait times were terrible.")
```
