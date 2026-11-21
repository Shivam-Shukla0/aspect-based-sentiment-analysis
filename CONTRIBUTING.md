# Contributing

## Workflow

1. Open or pick a GitHub issue before starting work.
2. Self-assign the issue (comment on the issue, add yourself to assignees). Maintain one active issue per person per workstream to prevent bottlenecks.
3. Branch from the latest `main`: `feature/<issue-number>-<short-name>` (also `fix/...`, `docs/...`, `test/...`, `chore/...`, `ci/...`).
4. Commit with a conventional prefix and the issue number, e.g. `feat(scope): subject (#<issue>)`. Prefixes: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `ci`. Commit subject lines must stay under 72 characters.
5. Push the branch and open a pull request that says `Closes #<issue>`.
6. Do not push directly to `main`.

## Self-Assignment and Workstreams

- Pick issues marked as unassigned in the backlog or discussion.
- Self-assign on GitHub before creating your branch so teammates know work has started.
- Keep to one active issue per contributor at a time within any given workstream (Foundation, Data, LSTM, BERT, Extraction, LLM, Evaluation, Integration).
- If blocked, comment on the issue and request input from the workstream lead.

## Labels and Issue Taxonomy

Issues use three distinct label groups:

- **Type labels** indicate the nature of the task:
  - `documentation`: Guides, architecture specs, README updates, presentations.
  - `enhancement`: Feature implementation, core logic, pipelines.
  - `bug`: Bug fixes and regression repairs.
  - `testing`: Unit tests, fixtures, end-to-end assertions.
  - `infrastructure`: Tooling, environment configurations, CI scripts.
  - `research`: Experiments, EDA, hyperparameter optimization, benchmarking.
- **Priority labels** reflect scheduling urgency:
  - `priority:high`: Core architecture and blocking path requirements.
  - `priority:medium`: Important functionality and secondary evaluations.
  - `priority:low`: Polish, nice-to-have visualisations, optional tooling.
- **Component labels** delineate functional modules:
  - `component:foundation`: Repository structure, environment, quality gates.
  - `component:data`: Data loading, XML parsers, validation, splits.
  - `component:lstm`: LSTM architectures, tokenizers, word embeddings, training.
  - `component:bert`: BERT tokenization, sentence-pair classification, fine-tuning.
  - `component:extraction`: Aspect extraction, sequence labelling (BIO tagging).
  - `component:llm`: Prompt engineering, LLM inference adapters, JSON parsing.
  - `component:evaluation`: Precision, recall, F1 metrics, joint evaluation.
  - `component:integration`: Model registry, Flask backend, end-to-end interface.
  - `component:frontend`: Interactive web UI and visualizations.
  - `component:documentation`: Final technical reports, demos, and presentations.

## Pull requests

- Describe what changed, how it was validated (commands and real results) and what is not done.
- Review requirements: at least two teammates review before merge, including a code owner. There is no fixed rotation; the author asks the owner of the area touched (see Shared contracts) and one other teammate.
- Reviewers verify code correctness, test execution results, type hints, and that no results are claimed that were not produced.
- Use the pull request template. The title must be `type(scope): subject`, the branch `<type>/<issue-number>-<name>`, every commit subject `type(scope): subject`, and the body needs `Closes #<issue>`.
- The `Explanation` section must say, in your own words, how the change works, what you checked yourself, and what could break. Do not copy the summary.
- Tick "I have read every line of this diff and can explain it" only if it is true. Reviewers may ask you to walk through any line.
- Approvals are dismissed automatically when new commits are pushed; all conversations must be marked resolved before merge.
- The author who pushed the latest commit cannot approve their own pull request.
- Keep PRs under 2000 changed lines. Split larger work.
- Open issues with an issue form; title prefix and required sections are validated.

## Checks

CI must be green before merge:

- Python: ruff lint (including docstrings, complexity, security, no TODO/FIXME), ruff format, vulture (dead code), mypy strict, pytest on Python 3.10 to 3.13 with warnings as errors and a 95% coverage floor.
- Docs and config: prettier for Markdown, YAML and JSON, codespell, yamllint, actionlint, zizmor, shellcheck.
- Packaging: wheel build, `twine check`, and a smoke test of the wheel in a clean virtualenv.
- Docker: hadolint, image build, container smoke test, non-root user check.
- ML environment: a separate job installs CPU PyTorch and `requirements.txt`, runs `pip check`, runs the NumPy/PyTorch reproducibility tests, and fails on GPL/AGPL dependencies. The whole pipeline also runs weekly to catch upstream breakage.
- Data and experiments: use `absa.reproducibility.set_seed` in every experiment and `absa.data.leakage.assert_no_leakage` before evaluating, so train and test reviews never overlap.
- No AI-agent attribution anywhere: commit messages, commit authors and committers, PR titles and bodies, issue text, review and issue comments, and tracked files are scanned for attribution trailers, footers and signatures, and every push to `main` is checked too. Any hit fails CI. The only bot allowed as a commit author is `dependabot[bot]`.
- Supply chain: `pip-audit`, and GitHub Actions pinned to commit SHAs.
- Repository hygiene: secrets, large files, datasets or checkpoints, notebook outputs, line endings, broken links.

### Running CI locally

```bash
pip install -r requirements-dev.txt
pip install -e . --no-deps
ruff check . && ruff format --check . && vulture && mypy && pytest
codespell && yamllint --strict . && actionlint && zizmor --offline --min-severity low .github
shellcheck scripts/*.sh && python -m scripts.checks repo
npx --yes prettier@3.6.2 --check "**/*.{md,yml,yaml,json}"
docker build -t absa:ci . && scripts/smoke_test.sh docker run --rm -i absa:ci
```

## Google Colab and Notebook Workflow

- Notebooks in `notebooks/` are reserved for EDA, experiment demonstrations, and training visualizations.
- Always clear execution outputs (`Cell -> All Output -> Clear`) before committing notebooks, or verify outputs contain zero sensitive information, tokens, or raw dataset payloads.
- Model weights, GloVe embeddings, and checkpoints generated during Colab runs must never be committed to Git. Store checkpoints in Google Drive or dedicated artifact storage.
- Downloaded SemEval XML files must remain in local uncommitted paths or Google Drive.

## Do not commit

- Secrets, tokens, API keys, or `.env` files.
- Datasets, model checkpoints, large outputs, virtual environments, or notebook checkpoints.
- Notebook outputs containing keys or large data; clear them first.

## Shared contracts

Changes to the record schema (`src/absa/data/schema.py`, `data/README.md`) or the evaluation protocol (`src/absa/evaluation/`) affect everyone. Contract ownership assignments:

- **Data Schema Owner**: Rajeev Yadav (`Rajeevydvv`)
- **Evaluation Protocol Owner**: Shivam Shukla (`Shivam-Shukla0`)

Any change to these modules requires explicit review and approval from the designated owner. The package dependency rules (`docs/architecture.md` and `tests/test_architecture.py`) are shared as well: change both together, and ask every workstream for review.

## License

No licence has been chosen yet, so all rights are reserved by default. The team has not made a licensing decision, and there is no `LICENSE` file. Do not assume any open-source terms until one is added.
