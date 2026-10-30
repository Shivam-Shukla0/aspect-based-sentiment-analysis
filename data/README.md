# Data

Raw and processed data are **not committed** (see `.gitignore`). The SemEval-2014 licence terms should be checked before any redistribution.

## Source

SemEval-2014 Task 4: Aspect Based Sentiment Analysis, laptop and restaurant reviews.
Download from <https://alt.qcri.org/semeval2014/task4/> and place the XML files in `data/raw/`, keeping the original train and test files separate.

Citation: Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., Manandhar, S. (2014). *SemEval-2014 Task 4: Aspect Based Sentiment Analysis.* Proceedings of SemEval 2014.

## Splits

- The original train/test separation is preserved. Test data is used only for final evaluation.
- Any validation split is carved out of the training data with `absa.config.RANDOM_SEED` and must be documented in the experiment notebook.

## Processed record schema

One JSON object per (review, aspect) pair:

```json
{"review": "The battery is weak but the screen is excellent.", "aspect": "battery", "sentiment": "negative"}
```

`sentiment` is one of `positive`, `negative`, `neutral`, `conflict`. `conflict` is kept by default. An experiment that drops it must state so and pass a reduced `allowed_labels` to `absa.data.validate_record`.

## Inference output schema (interface contract, not a model output)

```json
{"review": "...", "predictions": [{"aspect": "battery", "sentiment": "negative"}]}
```

The LSTM and BERT classifiers take a supplied aspect. End-to-end evaluation also needs aspect extraction (`src/absa/extraction/`).
