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
- Use the pull request template. The title must be `type(scope): subject`, the branch `<type>/<issue-number>-<name>`, every commit subject `type(scope): subject`, and the body needs `Closes #<issue>`.
- The `Explanation` section must say, in your own words, how the change works, what you checked yourself, and what could break. Do not copy the summary.
- Tick "I have read every line of this diff and can explain it" only if it is true. Reviewers may ask you to walk through any line.
- Keep PRs under 2000 changed lines. Split larger work.
- Open issues with an issue form; title prefix and required sections are validated.
- Every PR needs 2 approvals including a code owner, approvals are dismissed when new commits are pushed, the last pusher cannot approve their own push, and all review conversations must be resolved.

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

## Experiments and results

- Seed every experiment with `absa.reproducibility.set_seed`.
- Record the environment next to every result you report: `python -m scripts.verify_environment --record outputs/<experiment>/`. Quote the seed, git commit and key package versions with the result. See the README section "Recording the environment for a result".

## Do not commit

- Secrets, tokens, API keys, or `.env` files.
- Datasets, model checkpoints, large outputs, virtual environments, or notebook checkpoints.
- Notebook outputs containing keys or large data; clear them first.

## Shared contracts

Changes to the record schema (`src/absa/data/schema.py`, `data/README.md`) or the evaluation protocol (`src/absa/evaluation/`) affect everyone. Each has one owner, to be named by the team here: data schema — _TBD_; evaluation protocol — _TBD_. Changes need that owner's review.
