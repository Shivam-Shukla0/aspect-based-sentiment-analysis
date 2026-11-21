# Architecture

This document describes how the code in this repository is organised and which parts may depend on which. The dependency rules are checked by `tests/test_architecture.py`, so this page and the code cannot drift apart without CI failing.

**Status.** Only the pieces marked _implemented_ below exist today: project constants, the aspect-record schema and validation, the train/test leakage guard, the seeding helper and the validation CLI. Everything marked _planned_ is a documented module boundary with no implementation yet, and no result has been produced.

## Data flow

```
data/raw/                  SemEval-2014 Task 4 XML (git-ignored)
  -> absa.data             parse, normalise, validate, split
data/processed/            aspect-level records (git-ignored)
  -> absa.models           sentiment for a supplied aspect (LSTM with aspect attention, BERT sentence pair)
  -> absa.extraction       aspect terms found in a review (BERT token classification, LLM prompting)
predictions                written to outputs/ (git-ignored)
  -> absa.evaluation       shared metrics on the original SemEval test split
reports/                   course report material

review text
  -> absa.inference        combines models and extraction
  -> {"review": ..., "predictions": [{"aspect": ..., "sentiment": ...}]}
  -> app/                  Flask backend and frontend (planned)
```

The JSON shape is the output contract, not a model prediction. Classification with a supplied aspect and end-to-end extraction are different tasks and are evaluated separately.

## Packages

All packages live under `src/` and are installed as `absa`. The last column lists the only other `absa` parts a package may import, apart from itself.

| Package                | Responsibility                                                                                                 | Status                                                     | May import                                                 |
| ---------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| `absa.config`          | Shared constants: paths, domains, sentiment labels, default seed. No logic.                                    | implemented                                                | none                                                       |
| `absa._lazy`           | Lazy attribute helper that keeps optional heavy dependencies (torch, scikit-learn, lxml) out of plain imports. | implemented                                                | none                                                       |
| `absa.reproducibility` | Seeding for Python, NumPy and PyTorch.                                                                         | implemented                                                | `config`                                                   |
| `absa.data`            | Aspect-record schema and validation, leakage guard. Parsing and splits will join it.                           | schema and leakage implemented, parsing and splits planned | `config`, `reproducibility`, `_lazy`                       |
| `absa.models`          | Sentiment classifiers that take a review and a supplied aspect.                                                | planned                                                    | `config`, `data`, `evaluation`, `reproducibility`, `_lazy` |
| `absa.extraction`      | Aspect extraction from a review.                                                                               | planned                                                    | `config`, `data`, `reproducibility`, `_lazy`               |
| `absa.evaluation`      | Metrics and the evaluation protocol. Works on predictions, never on models.                                    | planned                                                    | `config`, `data`                                           |
| `absa.inference`       | One prediction interface that combines models and extraction.                                                  | planned                                                    | `config`, `data`, `models`, `extraction`                   |

`absa.__main__` is the command line entry point (`python -m absa`). Like `app/`, `scripts/` and the notebooks, it is allowed to compose several packages.

## Dependency rules

```
config, reproducibility, _lazy  leaf modules, importable from anywhere below
        ^
      data
        ^
  models    extraction         never import each other
        ^
inference   evaluation         never import each other
        ^
absa.__main__, scripts/, app/, notebooks/
```

1. Imports point downwards only. The rules form a directed acyclic graph, which the tests verify, so circular imports cannot appear.
2. Siblings do not import each other. `models` and `extraction` are combined in `inference`; `evaluation` receives predictions as plain records, so metrics do not depend on how a prediction was produced. The one exception is `models`, which may use `evaluation` for validation metrics during training; `evaluation` never imports `models`. A script combines models and evaluation when a benchmark needs both.
3. Only entry points compose packages: `absa.__main__`, `scripts/`, `app/` and notebooks.
4. Imports inside functions or under `TYPE_CHECKING` count as dependencies.
5. Changing a rule means editing the table above and `tests/test_architecture.py` in the same pull request. Both are shared contracts, so every workstream should review such a change.

## Import conventions

- Use absolute imports from `absa`, for example `from absa.data import validate_record`. Relative imports are not used, and the tests reject them.
- Between packages, import what a package exports from its `__init__.py` rather than reaching into its submodules.
- Take paths, labels, domains and the default seed from `absa.config`. Do not redefine them.
- Notebooks hold exploration, plots and commentary. Any function that is reused, tested or needed to reproduce a result belongs in `src/absa`, and the notebook imports it.
- Document a planned module in its docstring or in this file instead of adding an empty implementation.

## Code outside the package

| Location     | Holds                                                                                        | May import                           |
| ------------ | -------------------------------------------------------------------------------------------- | ------------------------------------ |
| `src/absa/`  | The installable library: everything with logic that is tested.                               | per the table above                  |
| `scripts/`   | Command line entry points for experiments (training, evaluation, benchmarks) and CI tooling. | any `absa` package                   |
| `app/`       | Flask backend and static frontend. Planned; the directory does not exist yet.                | `inference`, `data`, `config`        |
| `notebooks/` | Exploration and Colab runs. No results yet.                                                  | public names from any `absa` package |
| `tests/`     | Offline pytest suite using synthetic records.                                                | anything                             |
| `data/`      | `raw/` and `processed/` are git-ignored. [data/README.md](../data/README.md) has the schema. | not code                             |
| `outputs/`   | Generated artifacts such as checkpoints and metrics, git-ignored.                            | not code                             |
| `reports/`   | Course report material.                                                                      | not code                             |

Decisions recorded here:

- **Experiment entry points live in `scripts/`.** They stay thin: parse arguments, call `absa`, write to `outputs/`. They run from the repository root, for example `python -m scripts.<name>`, and are not part of the wheel or the Docker image.
- **The web application lives in `app/`**, outside the package, so the library and the Docker image stay free of web dependencies. It talks to the models only through `absa.inference`. Flask is added to the requirements when the app is added.
- **Large files never enter Git.** Checkpoints and datasets stay in `outputs/` and `data/`, or on Drive when training on Colab.

## Packaging

- `pyproject.toml` uses the `src` layout: `[tool.setuptools.packages.find]` has `where = ["src"]`, so only `absa` is installed. `scripts/`, `app/` and `tests/` are not.
- Runtime dependencies are listed once, in `requirements.txt`. `requirements-dev.txt` adds the tools CI runs. The package metadata does not repeat them.
- The supported Python versions are 3.10 to 3.13.
- See [CONTRIBUTING.md](../CONTRIBUTING.md#running-ci-locally) for the commands that reproduce CI locally.

## How the rules are enforced

`tests/test_architecture.py` checks that:

- every package or module under `src/absa` has a rule, so a new package cannot be added without deciding what it may import;
- the rules are acyclic and refer only to existing packages;
- no source file imports anything its package is not allowed to, using the syntax tree, including imports inside functions and relative imports;
- importing each package in a fresh interpreter loads no other `absa` package than the ones it may import;
- the table in this document lists exactly the same rules as the test;
- once it exists, `app/` imports only what the table allows.

The checks cannot see dynamic imports such as `importlib.import_module(name)`. Importing a package in a fresh interpreter catches those that run at import time, and code review covers the rest.

## Adding a package

1. Create `src/absa/<name>/__init__.py` with a docstring that states its responsibility.
2. Add its row to the table above and its entry to `ALLOWED` in `tests/test_architecture.py`.
3. Run `pytest tests/test_architecture.py`. It fails until the table and the test agree.
