# LSTM-with-attention: results and error analysis

## Scope and a real limitation

This evaluation runs on the laptop **trial file** (`data/raw/laptops-trial.xml`, 100 sentences, 49 aspect records), not the official SemEval-2014 test split — the full train/test files require a Metashare account that was not available in the environment this was produced in (see `data/README.md` and issue #13). There is no separate official test file to evaluate on yet, so the numbers below use the held-out validation split (11 records, `val_fraction=0.2, seed=42`) as a stand-in, clearly labelled as such.

Two consequences follow directly from this, honestly stated rather than worked around:

- The issue's acceptance criterion of inspecting **at least 30** incorrect predictions is not achievable on an 11-record evaluation set; only 3 misclassifications exist to inspect. All 3 are shown below.
- Macro F1 is weak (0.21) because this amount of training data (39 records) is not enough to learn anything beyond the majority class.

**This must be rerun on the real official test split once it is available**, using `python -m scripts.evaluate_lstm CHECKPOINT TEST_FILE`; the code itself does not depend on dataset size and needs no changes to do that. Everything else below is a real run, not invented numbers.

## Configuration

```python
LstmConfig(
    embedding_dim=50,
    hidden_size=32,
    num_layers=1,
    dropout=0.0,
    bidirectional=True,
    min_freq=1,
    batch_size=8,
    learning_rate=1e-3,
    epochs=15,
    val_fraction=0.2,
    seed=42,
)
```

Training/validation split: 38 train / 11 held-out, grouped by `sentence_id` (`absa.data.split_train_val`), checked leakage-free. Best epoch by validation macro F1: epoch 1 of 15 (macro F1 did not improve further; see the note on dataset size above).

## Metrics (held-out split, 11 records)

| Label    | Precision | Recall | F1   | Support |
| -------- | --------- | ------ | ---- | ------- |
| positive | 0.73      | 1.00   | 0.84 | 8       |
| negative | 0.00      | 0.00   | 0.00 | 2       |
| neutral  | 0.00      | 0.00   | 0.00 | 1       |
| conflict | 0.00      | 0.00   | 0.00 | 0       |

- Accuracy: 0.727
- Macro F1: 0.211
- Micro F1: 0.727

## Confusion matrix

Rows are true labels, columns are predicted labels, both in `[positive, negative, neutral, conflict]` order:

| true \ pred  | positive | negative | neutral | conflict |
| ------------ | -------- | -------- | ------- | -------- |
| **positive** | 8        | 0        | 0       | 0        |
| **negative** | 2        | 0        | 0       | 0        |
| **neutral**  | 1        | 0        | 0       | 0        |
| **conflict** | 0        | 0        | 0       | 0        |

## Error analysis

All 3 misclassifications (100% of the errors found) predicted **positive** regardless of the true label:

| #   | Review                                                                                                                                                       | Aspect            | True     | Predicted |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | -------- | --------- |
| 1   | "Call tech support, standard email the form and fax it back in to us."                                                                                       | tech support      | neutral  | positive  |
| 2   | "Games being the main issue."                                                                                                                                | Games             | negative | positive  |
| 3   | "When I called Toshiba, they would not do anything and even tried to charge me $35 for the phone call, even though they didn't offer any technical support." | technical support | negative | positive  |

**Category: majority-class collapse.** All three errors share one pattern — the model predicts the majority class (`positive`, 29/49 records in the raw trial data) instead of engaging with the aspect at all. This is consistent with the training config: 38 training records is far too little to learn an aspect-sensitive signal, so attention and the encoder fall back to the label prior. It is not evidence of a specific confusion between, say, `negative` and `neutral`, or of an attention/offset bug — the unit tests in `tests/test_aspect_attention.py` already check attention weighting, masking and gradient flow in isolation and pass. This is a dataset-size limitation, not a model-code defect.

## What to do once the full dataset is available

1. Rerun `scripts.build_processed` on the real train and test XML files.
2. Retrain with `scripts` wired to `run_training` (or re-run the same `LstmConfig`) on the full training set.
3. Re-run `scripts.evaluate_lstm` on the real, frozen test file and replace every number and example above.
4. With ~3000 sentences per domain, expect enough errors to categorize beyond "always predicts the majority class," which is the honest limit of what this trial-scale run can show.
