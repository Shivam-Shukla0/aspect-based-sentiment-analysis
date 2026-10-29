# Aspect-Based Sentiment Analysis of Product Reviews

Course project (4 members, 14 days). Given a product review, identify the aspects it discusses and the sentiment toward each one.

> **Status: scaffold only.** No model is implemented, trained or evaluated yet. Every component below is marked *planned* until it exists.

## Why aspect-based?

Document-level sentiment gives one label per review. Reviews often mix opinions:

```
Review: "The battery is weak but the screen is excellent."
Document level: mixed / ambiguous
Aspect level:   battery -> negative, screen -> positive
```

The JSON above is an illustration of the output contract, not a model prediction.

## Objectives

1. Build a reproducible pipeline on SemEval-2014 Task 4 (laptop and restaurant reviews).
2. Compare three approaches under one shared evaluation protocol.
3. Report honest results with limitations.

## Approaches

| # | Approach | Module | Status |
|---|----------|--------|--------|
| 1 | LSTM with aspect attention | `src/absa/models/lstm_attention.py` | planned |
| 2 | BERT sentence-pair classification | `src/absa/models/bert_classifier.py` | planned |
| 3 | LLM prompting for aspect and sentiment extraction | `src/absa/extraction/` | planned |

Approaches 1 and 2 classify sentiment for a **supplied** aspect. End-to-end evaluation also needs aspect extraction (approach 3 or another extractor), so the two settings are reported separately.

## Architecture (planned)

```
raw SemEval XML -> data (parse, split, validate) -> processed records
processed records -> models / extraction -> predictions
predictions -> evaluation (shared protocol) -> reports
review text -> inference -> {"review", "predictions": [{"aspect", "sentiment"}]}
```

## Dataset

SemEval-2014 Task 4, <https://alt.qcri.org/semeval2014/task4/>. Data is not committed. See [data/README.md](data/README.md) for download, citation, splits, record schema and the handling of the `conflict` label. The original train/test separation is preserved.

## Repository structure

```
src/absa/        config, data schema, models, extraction, evaluation, inference
notebooks/       one planning notebook per experiment (no results yet)
tests/           offline pytest suite using synthetic records
data/            raw/ and processed/ (git-ignored), data README
outputs/         generated artifacts (git-ignored)
reports/         course report material
```

A web demo under `app/` is a possible later addition and does not exist yet.

## Environment setup (Kali Linux, CPU)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

Installing the CPU build of `torch` first means `requirements.txt` will not replace it. Dependencies are unpinned; record the versions you used (`pip freeze`) alongside any reported result.

## Google Colab (GPU)

Colab ships a CUDA-enabled `torch`; do not reinstall it.

```python
!git clone https://github.com/Shivam-Shukla0/aspect-based-sentiment-analysis.git
%cd aspect-based-sentiment-analysis
!pip install -q numpy pandas scikit-learn lxml transformers accelerate sentencepiece matplotlib pytest
!pip install -q -e . --no-deps
```

The repository is private, so cloning requires authentication; use a Colab secret, never a token pasted into a notebook. Upload the dataset to the Colab session or Drive; do not commit it.

## Data preparation

Download the SemEval-2014 files into `data/raw/` as described in [data/README.md](data/README.md). Parsing and processing code is planned and not yet implemented.

## Tests

```bash
pytest
```

Tests run offline and use small synthetic records.

## Team workflow

See [CONTRIBUTING.md](CONTRIBUTING.md): issue first, feature branch, pull request with review, tests passing.

## Implementation status

- Done: project layout, config, record schema validation, tests.
- Planned: data parsing and splits, LSTM-attention, BERT classifier, LLM extraction, evaluation, inference, demo app.

## Limitations and reproducibility

- No results exist yet; none should be quoted until produced by code in this repository.
- Seeds are fixed via `absa.config.RANDOM_SEED`, but GPU training can still be non-deterministic.
- No licence has been chosen. The team must decide before public reuse; until then all rights are reserved by default.
