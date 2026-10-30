import json
from pathlib import Path

import pytest

from scripts import checks

PR_BODY = """Closes #12

## Summary
Adds the BERT pair encoder.

## Related issue
Closes #12

## Changes
- New encoder module and tests.

## Validation
pytest: 30 passed.

## Not included
Training loop.

## Checklist
- [x] Tests pass
- [x] No secrets
"""


def test_valid_pr_passes() -> None:
    assert (
        checks.check_pr("feat(models): add pair encoder", PR_BODY, "feature/12-pair-encoder") == []
    )


@pytest.mark.parametrize(
    "title",
    ["Add encoder", "feat: ", "feat:no space", "Feat: upper", "feat: " + "x" * 80],
)
def test_bad_titles_fail(title: str) -> None:
    assert checks.check_pr(title, PR_BODY, "feature/12-x")


@pytest.mark.parametrize(
    "branch", ["main", "feature/encoder", "Feature/1-x", "feature/1-X", "wip/1-x"]
)
def test_bad_branches_fail(branch: str) -> None:
    errors = checks.check_pr("feat: add thing", PR_BODY, branch)
    assert any("branch name" in e for e in errors)


def test_pr_without_closing_keyword_fails() -> None:
    body = PR_BODY.replace("Closes #12", "See #12")
    errors = checks.check_pr("feat: add thing", body, "feature/1-x")
    assert any("Related issue" in e and "Closes #" in e for e in errors)


def test_pr_missing_and_placeholder_sections_fail() -> None:
    body = PR_BODY.replace("## Validation\npytest: 30 passed.", "## Validation\nTBD")
    body = body.replace("## Changes", "## Other")
    errors = checks.check_pr("feat: add thing", body, "feature/1-x")
    assert any("missing the '## Changes'" in e for e in errors)
    assert any("'Validation' is empty" in e for e in errors)


def test_unchecked_checklist_fails() -> None:
    body = PR_BODY + "- [ ] Docs updated\n"
    assert any("unchecked" in e for e in checks.check_pr("feat: add thing", body, "feature/1-x"))


def test_html_comment_does_not_count_as_content() -> None:
    body = PR_BODY.replace("Adds the BERT pair encoder.", "<!-- describe your change here -->")
    errors = checks.check_pr("feat: add thing", body, "feature/1-x")
    assert any("'Summary' is empty" in e for e in errors)


def test_commit_checks() -> None:
    assert checks.check_commits(["feat: add thing (#3)\n\nBody text."]) == []
    assert checks.check_commits(["Added stuff"])
    trailer = "feat: add thing\n\nCo-Authored-By: Some Bot <noreply@example.com>"
    assert any("trailer" in e for e in checks.check_commits([trailer]))
    human = "feat: add thing\n\nCo-authored-by: A Person <1+a@users.noreply.github.com>"
    assert checks.check_commits([human]) == []


def test_generated_footer_rejected() -> None:
    assert checks.check_traces("body\n\nGenerated with SomeTool", "x")
    assert checks.check_traces("plain body", "x") == []


ISSUE_BODY = """### Objective

Set up the evaluation protocol for all approaches.

### Scope

Metrics, splits and reporting format.

### Acceptance criteria

Protocol is documented and tested.
"""


def test_valid_issue_passes() -> None:
    assert checks.check_issue("[Task]: define evaluation protocol", ISSUE_BODY) == []


def test_issue_title_prefix_required() -> None:
    assert checks.check_issue("define evaluation protocol", ISSUE_BODY)
    assert checks.check_issue("[Idea]: define evaluation protocol", ISSUE_BODY)


def test_issue_missing_or_empty_sections_fail() -> None:
    body = ISSUE_BODY.replace("Protocol is documented and tested.", "_No response_")
    body = body.replace("### Scope", "### Extra")
    errors = checks.check_issue("[Task]: define evaluation protocol", body)
    assert any("missing the 'Scope'" in e for e in errors)
    assert any("'Acceptance criteria' is empty" in e for e in errors)


def test_bug_issue_requires_bug_sections() -> None:
    errors = checks.check_issue("[Bug]: crash on empty review", ISSUE_BODY)
    assert any("Steps to reproduce" in e for e in errors)


def test_find_secrets_detects_known_shapes() -> None:
    assert checks.find_secrets("token=" + "ghp_" + "a" * 36)
    assert checks.find_secrets("AKIA" + "A" * 16)
    assert checks.find_secrets("-----BEGIN " + "RSA PRIVATE KEY-----")
    assert checks.find_secrets("nothing to see here") == []


def test_text_hygiene() -> None:
    assert checks.check_text_hygiene("f.md", "ok\n") == []
    assert checks.check_text_hygiene("f.md", "") == []
    assert any("final newline" in e for e in checks.check_text_hygiene("f", "x"))
    assert any("trailing whitespace" in e for e in checks.check_text_hygiene("f", "x \n"))
    assert any("CR" in e for e in checks.check_text_hygiene("f", "x\r\n"))
    assert any("extra blank" in e for e in checks.check_text_hygiene("f", "x\n\n"))


def _notebook(outputs: list[object] | None = None, execution_count: int | None = None) -> str:
    cell: dict[str, object] = {
        "cell_type": "code",
        "metadata": {},
        "source": ["1 + 1"],
        "outputs": outputs or [],
        "execution_count": execution_count,
        "id": "a",
    }
    return (
        json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, indent=1)
        + "\n"
    )


def test_notebook_checks() -> None:
    assert checks.check_notebook("n.ipynb", _notebook()) == []
    assert any(
        "outputs" in e for e in checks.check_notebook("n.ipynb", _notebook(execution_count=3))
    )
    assert any("invalid notebook" in e for e in checks.check_notebook("n.ipynb", "{}"))


def test_links(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("x\n")
    text = "[ok](a.md) [web](https://example.com) [anchor](#top) [bad](missing.md)"
    errors = checks.check_links(tmp_path, "docs/b.md", text)
    assert errors == ["docs/b.md: broken relative link 'missing.md'"]


def _repo(tmp_path: Path, extra: dict[str, str]) -> list[str]:
    files = dict.fromkeys(checks.REQUIRED_FILES, "x\n")
    files["pyproject.toml"] = '[project]\nname = "x"\n'
    files.update(extra)
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return checks.check_files(tmp_path, sorted(files))


def test_clean_repo_passes(tmp_path: Path) -> None:
    assert _repo(tmp_path, {"data/raw/.gitkeep": "", "a.json": "{}\n"}) == []


def test_repo_problems_are_reported(tmp_path: Path) -> None:
    errors = _repo(
        tmp_path,
        {
            "data/raw/train.xml": "x\n",
            "model.pt": "x\n",
            "bad.json": "{\n",
            "bad.toml": "a = \n",
            "key.txt": "ghp_" + "b" * 36 + "\n",
            "big.txt": "a" * (checks.MAX_FILE_BYTES + 1) + "\n",
        },
    )
    text = "\n".join(errors)
    for expected in (
        "train.xml",
        "model.pt",
        "invalid JSON",
        "invalid TOML",
        "possible secret",
        "larger than",
    ):
        assert expected in text


def test_missing_required_file_reported(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("x\n")
    errors = checks.check_files(tmp_path, ["README.md"])
    assert "required file missing: CONTRIBUTING.md" in errors


def test_binary_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "img.png").write_bytes(b"\xff\xfe\x00")
    errors = checks.check_files(tmp_path, ["img.png"])
    assert not any("img.png" in e for e in errors)
