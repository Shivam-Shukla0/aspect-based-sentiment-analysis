# Data

Raw and processed data are **not committed** (see `.gitignore`). The SemEval-2014 licence terms should be checked before any redistribution.

## Source

SemEval-2014 Task 4: Aspect Based Sentiment Analysis, laptop and restaurant reviews.
Download from <https://alt.qcri.org/semeval2014/task4/> and place the XML files in `data/raw/`, keeping the original train and test files separate.

Citation: Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., Manandhar, S. (2014). _SemEval-2014 Task 4: Aspect Based Sentiment Analysis._ Proceedings of SemEval 2014.

## Splits

- The original train/test separation is preserved. Test data is used only for final evaluation.
- Any validation split is carved out of the training data with `absa.config.RANDOM_SEED` and must be documented in the experiment notebook.

## Task settings

Three settings share the schema below, and a model only supports some of them:

| Setting                     | Input                      | Output                                 | Contract                        |
| --------------------------- | -------------------------- | -------------------------------------- | ------------------------------- |
| Given-aspect classification | review + a supplied aspect | one sentiment                          | `absa.data.validate_record`     |
| Aspect extraction           | review only                | aspect terms with offsets and polarity | `absa.data.ExtractionRecord`    |
| End-to-end prediction       | review only                | aspects and their sentiments           | `absa.data.validate_prediction` |

The LSTM and BERT classifiers only do given-aspect classification; they need an extractor (`src/absa/extraction/`) upstream to run end-to-end. LLM prompting can do either, depending on how it is prompted.

## Processed record schema

One JSON object per (review, aspect) pair, for given-aspect classification:

```json
{
  "review": "The battery is weak but the screen is excellent.",
  "aspect": "battery",
  "sentiment": "negative"
}
```

`sentiment` is one of `positive`, `negative`, `neutral`, `conflict`. `conflict` is kept by default. An experiment that drops it must state so and pass a reduced `allowed_labels` to `absa.data.validate_record`.

## Extraction record schema

One JSON object per sentence, for aspect extraction:

```json
{
  "text": "The battery is weak but the screen is excellent.",
  "spans": [
    { "term": "battery", "start": 4, "end": 11, "polarity": "negative" },
    { "term": "screen", "start": 32, "end": 38, "polarity": "positive" }
  ]
}
```

`start` and `end` are character offsets into `text`, so `text[start:end] == term`; this matches the SemEval annotation convention. `absa.data.ExtractionRecord` and `absa.data.AspectSpan` enforce it.

## Inference output schema (interface contract, not a model output)

```json
{ "review": "...", "predictions": [{ "aspect": "battery", "sentiment": "negative" }] }
```

`absa.data.validate_prediction` checks this contract. An empty `predictions` list is valid: it means no aspect was extracted from the review.

## Processed file naming

Processed files live under `data/processed/` (git-ignored) as JSON Lines, one file per domain and split: `{domain}_{split}.jsonl`, for example `laptop_train.jsonl` or `restaurant_test.jsonl`. `absa.config.processed_filename(domain, split)` builds this name; `absa.config.DOMAINS` and `absa.config.SPLITS` list the allowed values. The official test split is never re-split or overwritten.
