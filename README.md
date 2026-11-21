# Aspect-Based Sentiment Analysis of Product Reviews

Course project (4 members, 14 days). Given a product review, identify the aspects it discusses and the sentiment toward each one.

> **Status: scaffold only.** No model is implemented, trained or evaluated yet. Every component below is marked _planned_ until it exists.

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

| #   | Approach                                          | Module                               | Status  |
| --- | ------------------------------------------------- | ------------------------------------ | ------- |
| 1   | LSTM with aspect attention                        | `src/absa/models/lstm_attention.py`  | planned |
| 2   | BERT sentence-pair classification                 | `src/absa/models/bert_classifier.py` | planned |
| 3   | LLM prompting for aspect and sentiment extraction | `src/absa/extraction/`               | planned |

Approaches 1 and 2 classify sentiment for a **supplied** aspect. End-to-end evaluation also needs aspect extraction (approach 3 or another extractor), so the two settings are reported separately.

## Architecture (planned)

```
raw SemEval XML -> data (parse, split, validate) -> processed records
processed records -> models / extraction -> predictions
predictions -> evaluation (shared protocol) -> reports
review text -> inference -> {"review", "predictions": [{"aspect", "sentiment"}]}
```

Package responsibilities, the rules for which package may import which, and where scripts and the web app live are in [docs/architecture.md](docs/architecture.md). The import rules are checked by `tests/test_architecture.py`.

## Dataset

SemEval-2014 Task 4, <https://alt.qcri.org/semeval2014/task4/>. Data is not committed. See [data/README.md](data/README.md) for download, citation, splits, record schema and the handling of the `conflict` label. The original train/test separation is preserved.

## Repository structure

```
src/absa/        config, reproducibility, data, models, extraction, evaluation, inference
scripts/         experiment entry points (planned) and CI tooling
notebooks/       one planning notebook per experiment (no results yet)
tests/           offline pytest suite using synthetic records
docs/            architecture and dependency rules
data/            raw/ and processed/ (git-ignored), data README
outputs/         generated artifacts (git-ignored)
reports/         course report material
app/             Flask backend and frontend (planned, does not exist yet)
```

See [docs/architecture.md](docs/architecture.md) for what each package does and may import.

## Environment setup (Kali Linux, CPU)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
python -m scripts.verify_environment
```

The last command prints the Python, package and device versions, then `environment ok`, and exits 0. It exits 1 and names the problem if Python is older than 3.10 or a core package is missing or broken. On Kali the device line reads `CUDA not available, running on CPU`.

- Always use a virtual environment: recent Kali releases mark the system Python as externally managed, so a global `pip install` is refused.
- Install the CPU build of `torch` first. The default Linux wheel pulls in several gigabytes of CUDA libraries, and `requirements.txt` does not replace a `torch` that is already installed.
- Dependencies are unpinned. Do not pin a CUDA build in `requirements.txt`: it would break either Kali (no GPU) or Colab (its own CUDA version).
- The check needs no network access.

## Google Colab (GPU)

Colab ships a CUDA-enabled `torch`; do not reinstall or upgrade it.

1. Choose a GPU runtime: _Runtime_, _Change runtime type_, then a GPU such as T4.
2. Store a read-only GitHub token as a Colab secret named `GITHUB_TOKEN` (the key icon in the left bar) and allow this notebook to read it. The repository is private, so cloning needs it.
3. Run this in the first cell:

```python
import os

from google.colab import userdata

os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
!git clone https://oauth2:$GITHUB_TOKEN@github.com/Shivam-Shukla0/aspect-based-sentiment-analysis.git
%cd aspect-based-sentiment-analysis
!git remote set-url origin https://github.com/Shivam-Shukla0/aspect-based-sentiment-analysis.git
!pip install -q -r requirements.txt
!pip install -q -e . --no-deps
!python -m scripts.verify_environment --require-gpu
```

The `set-url` line removes the token from `.git/config`. The last line prints the GPU name and CUDA version, and exits 1 if the runtime has no GPU. `requirements.txt` leaves `torch` unpinned, so pip keeps the version Colab already has.

Known pitfalls:

- **Sessions reset.** The VM disk is wiped when the session ends or idles out, and GPU availability is not guaranteed on the free tier. Repeat the cell above in every new session, and copy results to Drive when a run finishes.
- **Drive.** Mount it with `from google.colab import drive; drive.mount("/content/drive")`, then copy, for example, `!cp -r outputs/<experiment> /content/drive/MyDrive/absa/`. Keep checkpoints and datasets there, never in Git.
- **Secrets.** Read tokens and API keys with `userdata.get`. Never paste them into a cell or let them appear in a cell output, and clear notebook outputs before committing.
- **Restart prompts.** If pip replaces a package that is already imported, Colab asks for a restart. Restart the runtime and run the cell again.

## Recording the environment for a result

Every result in a report, table or pull request must be traceable to the environment that produced it. In the session that produced a result, run:

```bash
python -m scripts.verify_environment --record outputs/<experiment>/
```

This writes `outputs/<experiment>/environment.txt` with the Python and platform versions, the device, the git commit, and every installed package as `name==version`, sorted by name (the project itself is left out). Convention:

- Keep one `outputs/<experiment>/` directory per run, with the metrics, the checkpoint and `environment.txt` together. On Colab, copy the whole directory to Drive.
- Quote a result together with its seed (see `absa.reproducibility.set_seed`), its git commit and the Python, `torch`, `transformers` and device lines from the record.
- `outputs/` is git-ignored, so copy those lines into `reports/` or the pull request when you report the result.
- The file is a record, not a lock file. The `torch` pin carries its build suffix (for example `+cpu`), which is not on PyPI, so rebuild by installing the same `torch` from the matching index.

## Data preparation

Download the SemEval-2014 files into `data/raw/` as described in [data/README.md](data/README.md). Parsing and processing code is planned and not yet implemented.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests run offline and use small synthetic records. `pytest` also enforces a coverage floor. To run only the fast tests, use `pytest -m "not slow and not gpu"`. See [Test conventions](CONTRIBUTING.md#test-conventions) for the rules and shared fixtures. The full CI check set can be run locally (see [CONTRIBUTING.md](CONTRIBUTING.md#running-ci-locally)).

## Docker

```bash
docker build -t absa .
echo '{"review": "The battery is weak.", "aspect": "battery", "sentiment": "negative"}' | docker run --rm -i absa validate
```

The image only contains the lightweight package and its record-validation CLI (`python -m absa validate`). It does not contain any model.

## Team workflow

See [CONTRIBUTING.md](CONTRIBUTING.md): issue first, feature branch, pull request with review, tests passing.

## Implementation status

- Done: project layout, config, record schema validation, seeding helper, leakage guard, architecture document and import-rule tests.
- Planned: data parsing and splits, LSTM-attention, BERT classifier, LLM extraction, evaluation, inference, demo app.

## Limitations and reproducibility

- No results exist yet; none should be quoted until produced by code in this repository.
- Seeds are fixed via `absa.config.RANDOM_SEED`, but GPU training can still be non-deterministic.
- No licence has been chosen. The team must decide before public reuse; until then all rights are reserved by default.
