"""Validators run by CI for pull requests, issues and repository hygiene.

Usage: python -m scripts.checks {pr,commits,issue,repo}
Inputs for ``pr``, ``commits`` and ``issue`` come from environment variables.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import tomllib

TYPES = "feat|fix|docs|test|chore|refactor|ci|build|perf|style"
SUBJECT_RE = re.compile(rf"^({TYPES})(\([a-z0-9-]+\))?: [a-z0-9].+$")
MAX_SUBJECT = 72
BRANCH_RE = re.compile(r"^(feature|fix|docs|chore|test|refactor|ci)/[0-9]+-[a-z0-9]+(-[a-z0-9]+)*$")
CLOSES_RE = re.compile(r"^\s*(closes|fixes|resolves) #[0-9]+\s*$", re.IGNORECASE | re.MULTILINE)
UNCHECKED_RE = re.compile(r"^\s*[-*] \[ \]", re.MULTILINE)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
BOT_TRAILER_RE = re.compile(
    r"^co-authored-by:.*noreply@(?!users\.noreply\.github\.com)", re.IGNORECASE | re.MULTILINE
)
GENERATED_RE = re.compile(r"^.*generated (with|by) .*$", re.IGNORECASE | re.MULTILINE)

PR_SECTIONS = ("Summary", "Related issue", "Changes", "Validation", "Not included", "Checklist")
ISSUE_TITLE_RE = re.compile(r"^\[(Bug|Feature|Task)\]: \S.{4,}$")
ISSUE_SECTIONS = {
    "Bug": ("Summary", "Steps to reproduce", "Expected behavior", "Actual behavior", "Environment"),
    "Feature": ("Problem", "Proposed solution", "Acceptance criteria"),
    "Task": ("Objective", "Scope", "Acceptance criteria"),
}
MIN_SECTION_CHARS = 10
EMPTY_ANSWERS = {"_no response_", "n/a", "none", "tbd", "todo"}

REQUIRED_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    ".gitignore",
    ".editorconfig",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
)
MAX_FILE_BYTES = 1024 * 1024
FORBIDDEN_PATH_RE = re.compile(
    r"(^|/)(\.env(\..*)?|.*\.(pt|pth|ckpt|safetensors|bin|pem|key)|data/raw/(?!\.gitkeep).+"
    r"|data/processed/(?!\.gitkeep).+|outputs/(?!README\.md).+)$"
)
SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{50,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"sk-[A-Za-z0-9_-]{32,}"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
)
LINK_RE = re.compile(r"\]\(([^)#\s:]+)(?:#[^)]*)?\)")


def sections(body: str, marker: str) -> dict[str, str]:
    """Map each ``marker`` heading in ``body`` to its stripped text."""
    body = COMMENT_RE.sub("", body)
    parts = re.split(rf"^{re.escape(marker)} +(.+?) *$", body, flags=re.MULTILINE)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def _is_empty(text: str) -> bool:
    return len(text) < MIN_SECTION_CHARS or text.lower() in EMPTY_ANSWERS


def check_subject(subject: str, what: str) -> list[str]:
    errors = []
    if not SUBJECT_RE.match(subject):
        errors.append(
            f"{what} must look like 'type(scope): subject' with type in {TYPES}: {subject!r}"
        )
    if len(subject) > MAX_SUBJECT:
        errors.append(f"{what} is longer than {MAX_SUBJECT} characters")
    return errors


def check_traces(text: str, what: str) -> list[str]:
    if BOT_TRAILER_RE.search(text) or GENERATED_RE.search(text):
        return [f"{what} contains a bot co-author trailer or a 'generated with/by' footer"]
    return []


def check_pr(title: str, body: str, branch: str) -> list[str]:
    errors = check_subject(title, "PR title")
    if not BRANCH_RE.match(branch):
        errors.append(f"branch name {branch!r} must match <type>/<issue-number>-<kebab-name>")
    found = sections(body, "##")
    for name in PR_SECTIONS:
        if name not in found:
            errors.append(f"PR body is missing the '## {name}' section")
        elif name == "Related issue":
            if not CLOSES_RE.search(found[name]):
                errors.append("'Related issue' needs a line 'Closes #<issue>' (or Fixes/Resolves)")
        elif _is_empty(found[name]):
            errors.append(f"PR section '{name}' is empty or a placeholder")
    if UNCHECKED_RE.search(COMMENT_RE.sub("", body)):
        errors.append("PR checklist has unchecked items")
    return errors + check_traces(f"{title}\n{body}", "PR title/body")


def check_commits(messages: list[str]) -> list[str]:
    errors = []
    for message in messages:
        subject = message.strip().splitlines()[0] if message.strip() else ""
        errors += check_subject(subject, f"commit subject {subject!r}")
        errors += check_traces(message, f"commit {subject!r}")
    return errors


def check_issue(title: str, body: str) -> list[str]:
    match = ISSUE_TITLE_RE.match(title)
    if not match:
        return [f"issue title must look like '[Bug|Feature|Task]: summary', got {title!r}"]
    found = sections(body, "###")
    errors = []
    for name in ISSUE_SECTIONS[match.group(1)]:
        if name not in found:
            errors.append(f"issue is missing the '{name}' section")
        elif _is_empty(found[name]):
            errors.append(f"issue section '{name}' is empty or a placeholder")
    return errors


def find_secrets(text: str) -> list[str]:
    return [p.pattern for p in SECRET_PATTERNS if p.search(text)]


def check_text_hygiene(name: str, text: str) -> list[str]:
    errors = []
    if "\r" in text:
        errors.append(f"{name}: contains CR characters (use LF line endings)")
    if text and not text.endswith("\n"):
        errors.append(f"{name}: missing final newline")
    if text.endswith("\n\n"):
        errors.append(f"{name}: extra blank lines at end of file")
    for number, line in enumerate(text.split("\n"), start=1):
        if line != line.rstrip():
            errors.append(f"{name}:{number}: trailing whitespace")
    return errors


def check_notebook(name: str, text: str) -> list[str]:
    import nbformat  # noqa: PLC0415  (only needed for notebooks)

    try:
        notebook = nbformat.reads(text, as_version=4)  # type: ignore[no-untyped-call]
        nbformat.validate(notebook)
    except Exception as exc:
        return [f"{name}: invalid notebook: {exc}"]
    errors = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and (cell.outputs or cell.execution_count is not None):
            errors.append(f"{name}: cell {index} has stored outputs; clear them before committing")
    return errors


def check_links(root: Path, name: str, text: str) -> list[str]:
    base = (root / name).parent
    return [
        f"{name}: broken relative link {target!r}"
        for target in LINK_RE.findall(text)
        if not (base / target).exists()
    ]


def check_by_type(root: Path, name: str, text: str) -> list[str]:
    if name.endswith(".ipynb"):
        return check_notebook(name, text)
    if name.endswith(".md"):
        return check_links(root, name, text)
    try:
        if name.endswith(".json"):
            json.loads(text)
        elif name.endswith(".toml"):
            tomllib.loads(text)
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        return [f"{name}: invalid {name.rsplit('.', 1)[-1].upper()}: {exc}"]
    return []


def check_files(root: Path, files: list[str]) -> list[str]:
    errors = [f"required file missing: {name}" for name in REQUIRED_FILES if name not in files]
    for name in files:
        path = root / name
        if FORBIDDEN_PATH_RE.search(name):
            errors.append(f"{name}: dataset, checkpoint, secret or output file must not be tracked")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"{name}: larger than {MAX_FILE_BYTES} bytes")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        errors += [f"{name}: possible secret ({p})" for p in find_secrets(text)]
        errors += check_text_hygiene(name, text)
        errors += check_by_type(root, name, text)
    return errors


def _report(errors: list[str]) -> int:  # pragma: no cover
    for error in errors:
        print(f"::error::{error}")
    print(f"{len(errors)} problem(s) found" if errors else "all checks passed")
    return 1 if errors else 0


def main(argv: list[str]) -> int:  # pragma: no cover
    env = os.environ
    command = argv[1] if len(argv) > 1 else ""
    if command == "pr":
        errors = check_pr(env["PR_TITLE"], env.get("PR_BODY", ""), env["PR_BRANCH"])
    elif command == "commits":
        log = subprocess.run(
            ["git", "log", "--format=%B%x00", f"{env['BASE_SHA']}..{env['HEAD_SHA']}"],
            check=True, capture_output=True, text=True,
        ).stdout  # fmt: skip
        errors = check_commits([m for m in log.split("\x00") if m.strip()])
    elif command == "issue":
        errors = check_issue(env["ISSUE_TITLE"], env.get("ISSUE_BODY", ""))
    elif command == "repo":
        listing = subprocess.run(
            ["git", "ls-files", "-z"], check=True, capture_output=True, text=True
        ).stdout
        errors = check_files(Path.cwd(), [f for f in listing.split("\x00") if f])
    else:
        print(__doc__)
        return 2
    return _report(errors)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
