# Aspect-Based Sentiment Analysis of Product Reviews

Given a product review, identify the aspects it discusses and the sentiment expressed towards each one. The project compares two supervised approaches on SemEval-2014 Task 4 (laptop and restaurant reviews) under a single, leak-free evaluation protocol: an LSTM with aspect-driven attention, and a BERT sentence-pair classifier fine-tuned on a GPU. A third approach, language-model prompting, is implemented up to prompt construction, parsing and validation, but is not benchmarked (see [Scope and limitations](#scope-and-limitations)).

```
Review:         "The battery is weak but the screen is excellent."
Document level: mixed
Aspect level:   battery -> negative, screen -> positive
```

## Contents

- [Approaches](#approaches)
- [Quick start on Google Colab](#quick-start-on-google-colab)
- [Local setup](#local-setup)
- [Data](#data)
- [Usage](#usage)
- [Evaluation protocol](#evaluation-protocol)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Quality gates and testing](#quality-gates-and-testing)
- [Scope and limitations](#scope-and-limitations)
- [Contributing](#contributing)

## Approaches

| #   | Approach                          | Code                                                          | Status                                       |
| --- | --------------------------------- | ------------------------------------------------------------- | -------------------------------------------- |
| 1   | LSTM with aspect attention        | `src/absa/models/lstm_*.py`, `scripts/train_lstm.py`          | implemented, trained and evaluated           |
| 2   | BERT sentence-pair classification | `src/absa/models/bert_classifier.py`, `scripts/train_bert.py` | implemented, trained and evaluated           |
| 3   | Language-model prompting          | `src/absa/extraction/prompts.py`, `src/absa/models/llm*.py`   | prompts, parser, validation; not benchmarked |

Approaches 1 and 2 solve **given-aspect classification**: the aspect term is supplied and the model predicts its polarity. A complete system also needs aspect extraction, which is reported as a separate setting; BIO-tagged training data is prepared (`absa.extraction.bio`), an extractor is not trained.

## Quick start on Google Colab

The training code uses a CUDA GPU when one is available, so the notebooks are the recommended way to run it. Select a GPU runtime (_Runtime → Change runtime type_) and open a notebook:

| Notebook                                                                                                                                                           | Purpose                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| [01_dataset_exploration](https://colab.research.google.com/github/Shivam-Shukla0/aspect-based-sentiment-analysis/blob/main/notebooks/01_dataset_exploration.ipynb) | label balance, aspect frequency, review lengths |
| [02_lstm_attention](https://colab.research.google.com/github/Shivam-Shukla0/aspect-based-sentiment-analysis/blob/main/notebooks/02_lstm_attention.ipynb)           | tune and evaluate the LSTM                      |
| [03_bert_classification](https://colab.research.google.com/github/Shivam-Shukla0/aspect-based-sentiment-analysis/blob/main/notebooks/03_bert_classification.ipynb) | fine-tune and evaluate BERT, confusion matrices |
| [04_llm_prompting](https://colab.research.google.com/github/Shivam-Shukla0/aspect-based-sentiment-analysis/blob/main/notebooks/04_llm_prompting.ipynb)             | prompts, parsing and validation (no benchmark)  |

Each notebook clones the repository, installs the requirements, checks that a GPU is visible and runs the same scripts described below. The committed notebooks keep the outputs of a run on a Colab GPU, so the results can be read without re-running them.

## Local setup

Python 3.10 or newer. Use a virtual environment, and install the PyTorch build that matches your hardware first; `requirements.txt` leaves `torch` unpinned so it never replaces that choice.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch                      # or the CUDA/CPU build for your platform, see pytorch.org
pip install -r requirements.txt
pip install -e . --no-deps
python -m scripts.verify_environment   # add --require-gpu to fail without a CUDA device
```

`verify_environment` prints the Python, package and device versions and exits non-zero if a core package is missing. `--record DIR` writes the same information, the platform and the git commit to `DIR/environment.txt`; keep that file next to every result you report.

Without a GPU the code runs on the CPU, which is adequate for the test suite and small experiments but slow for BERT fine-tuning.

## Data

SemEval-2014 Task 4 is not committed to the repository. The official files are distributed through the ILSP Metashare repository and require registration and acceptance of its licence; follow [data/README.md](data/README.md) for the exact files, the record schema, the checksum table and the handling of the `conflict` label. Place the four XML files in `data/raw/` (they are git-ignored), then:

```bash
python -m scripts.check_dataset                     # verify files against the recorded checksums
python -m scripts.build_processed data/raw/Laptop_Train_v2.xml       --domain laptop     --split train
python -m scripts.build_processed data/raw/Laptops_Test_Gold.xml     --domain laptop     --split test
python -m scripts.build_processed data/raw/Restaurants_Train_v2.xml  --domain restaurant --split train
python -m scripts.build_processed data/raw/Restaurants_Test_Gold.xml --domain restaurant --split test
python -m scripts.validate_processed data/processed/laptop_train.jsonl
```

Each run writes one aspect-level JSON Lines file per domain and split, plus a `*_no_aspect.jsonl` file with the sentences that carry no aspect annotation.

## Usage

```bash
# LSTM: tune on the validation split, then score the test file once
python -m scripts.train_lstm --domain laptop --hidden 128 --lr 0.001 0.003 --dropout 0.0 0.3 0.5 --weighting off --epochs 40

# BERT: learning-rate search, best epoch kept, test file scored once
python -m scripts.train_bert --domain laptop
```

Both scripts write `summary.json` (every tried setting with its validation macro F1, the chosen setting, the test metrics) and `test_report.json` (per-class precision, recall and F1, a confusion matrix and every misclassified example) to `outputs/<model>_<domain>/`. Outputs are git-ignored.

The validation CLI checks aspect records from a file or stdin:

```bash
echo '{"review": "The battery is weak.", "aspect": "battery", "sentiment": "negative"}' | python -m absa validate
```

## Evaluation protocol

- **Official splits.** The original train and test files are never mixed. A validation split (20%) is carved out of the training data only.
- **No leakage.** Records are grouped by normalised review text before splitting, so two aspects of one sentence, or two sentences with identical text, never fall on both sides of a split; `absa.data.leakage.assert_no_leakage` checks the result.
- **The test set is read once.** Every choice (learning rate, dropout, the epoch to keep) uses validation macro F1 only. The test file is scored after the setting is fixed.
- **Labels.** The rare `conflict` label is dropped for training and scoring (the usual SemEval protocol); metrics are accuracy and macro F1 over the three polarity classes, with per-class results and a confusion matrix.
- **Baselines.** Always predicting the majority class (`positive`) scores about 0.53 accuracy on laptops and 0.65 on restaurants. A model should be judged against these figures, not against chance.
- **Reproducibility.** Every random number generator is seeded through `absa.reproducibility.set_seed` (default seed 42). GPU kernels can still introduce small run-to-run differences.

## Results

The slides are in [reports/slides.pptx](reports/slides.pptx) and the reproduction and demo script in [docs/demo-guide.md](docs/demo-guide.md). Results, per-class tables, the error analysis and their limitations are in the technical report, [reports/report.tex](reports/report.tex), written for the _Natural Language Engineering_ journal class (see [reports/README.md](reports/README.md) for how to build it). Figures are not repeated here so that there is one authoritative source. The report states plainly that results come from one run per setting and from a copy of the dataset that did not come through the official Metashare route, and they should be re-run on the official files before being quoted elsewhere.

## Repository layout

```
src/absa/            installable package
  config.py            paths, labels, seed
  data/                aspect-record schema, parsers, normalisation, validation, splits, EDA, leakage guard
  extraction/          BIO tagging and structured prompts
  models/              LSTM with aspect attention, BERT classifier, language-model client, output validation
  evaluation/          shared precision/recall/F1, span and end-to-end evaluators
  inference/           common predictor interface and registry
scripts/             command line entry points and CI tooling
app/                 Flask backend and demo page (see app/README.md)
notebooks/           Colab notebooks (with the outputs of a GPU run)
tests/               offline test suite on synthetic records
docs/                architecture, annotation format, model notes
reports/             technical report (LaTeX)
data/  outputs/      git-ignored working directories (data, checkpoints, metrics)
```

Package dependencies point one way (`config` → `data` → `models`/`extraction` → `inference`, with `evaluation` independent of models); the rules are in [docs/architecture.md](docs/architecture.md) and enforced by `tests/test_architecture.py`.

## Quality gates and testing

```bash
pip install -r requirements-dev.txt
pytest                      # offline, warnings are errors, 95% coverage floor
ruff check . && ruff format --check . && mypy
```

CI runs on every pull request: lint and format, strict type checking, tests on Python 3.10 to 3.13, a separate job with the full ML dependencies and a coverage floor, a dependency audit, packaging and Docker builds, repository hygiene (no datasets, checkpoints, secrets or notebook outputs), and commit and pull-request format checks. The test suite is deterministic and needs no network or GPU; GPU-specific tests skip themselves on machines without CUDA.

## Scope and limitations

- Given-aspect classification only; no trained aspect extractor and therefore no end-to-end figure.
- Language-model prompting is not benchmarked: it needs a model endpoint and key that are not part of the repository.
- The LSTM starts from random word vectors and is a baseline; BERT is the stronger model.
- Single seed per setting; differences between near-identical settings are within what another seed could change.
- The demo back end (`python -m app.server`) serves the trained classifiers for aspect terms the user supplies; it is a single-process demo server, not a production deployment.
- No licence has been chosen. Until the team adds one, all rights are reserved by default.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): an issue first, a short-lived branch (`<type>/<issue-number>-<name>`), conventional commit subjects of at most 72 characters, and a pull request that passes the checks.
