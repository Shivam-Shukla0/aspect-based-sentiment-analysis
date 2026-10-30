# Contributing

## Workflow

1. Open or pick a GitHub issue before starting work.
2. Branch from the latest `main`: `feature/<issue-number>-<short-name>` (also `fix/...`, `docs/...`).
3. Commit with a conventional prefix and the issue number, e.g. `feat: add BERT pair encoder (#7)`. Prefixes: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`.
4. Push the branch and open a pull request that says `Closes #<issue>`.
5. Do not push directly to `main`.

## Pull requests

- Describe what changed, how it was validated (commands and real results) and what is not done.
- At least one teammate reviews before merge. Reviewers check correctness, tests, and that no results are claimed that were not produced.
- Use the pull request template. The title must be `type(scope): subject`, the branch `<type>/<issue-number>-<name>`, every commit subject `type(scope): subject`, and the body needs `Closes #<issue>` and all checklist items ticked.
- Open issues with an issue form; title prefix and required sections are validated.
- CI must be green: ruff lint and format, mypy (strict), codespell, yamllint, actionlint, pytest on Python 3.10 to 3.13 with a coverage floor, and repository hygiene checks (secrets, large files, notebook outputs, broken links). Install tools with `pip install -r requirements-dev.txt`.

## Do not commit

- Secrets, tokens, API keys, or `.env` files.
- Datasets, model checkpoints, large outputs, virtual environments, or notebook checkpoints.
- Notebook outputs containing keys or large data; clear them first.

## Shared contracts

Changes to the record schema (`src/absa/data/schema.py`, `data/README.md`) or the evaluation protocol (`src/absa/evaluation/`) affect everyone. Each has one owner, to be named by the team here: data schema — _TBD_; evaluation protocol — _TBD_. Changes need that owner's review.
