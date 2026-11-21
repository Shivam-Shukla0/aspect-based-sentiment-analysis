"""Enforce the package dependency rules documented in docs/architecture.md."""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "absa"
APP = ROOT / "app"
DOC = ROOT / "docs" / "architecture.md"

# Entry points compose packages, so they are exempt from the rules below.
ENTRY_MODULES = frozenset({"__init__", "__main__"})

# Package or module under src/absa -> the other absa packages it may import.
# Keep this in sync with the table in docs/architecture.md (a test compares them).
ALLOWED: dict[str, frozenset[str]] = {
    "config": frozenset(),
    "_lazy": frozenset(),
    "reproducibility": frozenset({"config"}),
    "data": frozenset({"config", "reproducibility", "_lazy"}),
    "models": frozenset({"config", "data", "evaluation", "extraction", "reproducibility", "_lazy"}),
    "extraction": frozenset({"config", "data", "reproducibility", "_lazy"}),
    "evaluation": frozenset({"config", "data"}),
    "inference": frozenset({"_lazy", "config", "data", "models", "extraction"}),
}
APP_ALLOWED = frozenset({"config", "data", "inference"})

# Imported in a fresh interpreter to list which absa packages a package pulls in.
PROBE = (
    "import importlib, json, sys\n"
    "importlib.import_module('absa.' + sys.argv[1])\n"
    "print(json.dumps(sorted({m.split('.')[1] for m in sys.modules if m.startswith('absa.')})))"
)


def discover_units() -> set[str]:
    """Return the names of all packages and modules directly under ``src/absa``."""
    packages = {p.name for p in PACKAGE.iterdir() if (p / "__init__.py").exists()}
    modules = {p.stem for p in PACKAGE.glob("*.py")} - ENTRY_MODULES
    return packages | modules


def unit_files(unit: str) -> list[Path]:
    """Return every Python source file that belongs to ``unit``."""
    package = PACKAGE / unit
    return sorted(package.rglob("*.py")) if package.is_dir() else [PACKAGE / f"{unit}.py"]


def _from_modules(node: ast.ImportFrom) -> list[str]:
    module = node.module or ""
    if module == "absa":
        return [f"absa.{alias.name}" for alias in node.names if not alias.name.startswith("_")]
    return [module]


def imported_units(tree: ast.AST) -> set[str]:
    """Return the first-level ``absa`` packages and modules that ``tree`` imports."""
    units: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = _from_modules(node)
        else:
            continue
        units |= {module.split(".")[1] for module in modules if module.startswith("absa.")}
    return units


def relative_import_lines(tree: ast.AST) -> list[int]:
    """Return the line numbers of relative imports in ``tree``."""
    return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level > 0]


def source_violations(source: str, name: str, own: str, allowed: frozenset[str]) -> list[str]:
    """Return the rule violations in ``source``, a file called ``name`` that belongs to ``own``."""
    tree = ast.parse(source)
    errors = [f"{name}:{line}: use absolute imports" for line in relative_import_lines(tree)]
    errors.extend(
        f"{name}: imports absa.{unit}, which {own} may not depend on"
        for unit in sorted(imported_units(tree) - allowed - {own})
    )
    return errors


def violations(path: Path, own: str, allowed: frozenset[str]) -> list[str]:
    """Return the rule violations in the file ``path``, which belongs to ``own``."""
    name = path.relative_to(ROOT).as_posix()
    return source_violations(path.read_text(encoding="utf-8"), name, own, allowed)


def documented_rules() -> dict[str, frozenset[str]]:
    """Read the package table in docs/architecture.md into the same shape as ``ALLOWED``."""
    rules: dict[str, frozenset[str]] = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        match = re.fullmatch(r"`absa\.(\w+)`", cells[0])
        if line.startswith("|") and match:
            rules[match.group(1)] = frozenset(re.findall(r"`(\w+)`", cells[-1]))
    return rules


def test_every_package_has_a_rule() -> None:
    assert discover_units() == set(ALLOWED), (
        "add the new package to ALLOWED in tests/test_architecture.py "
        "and to the table in docs/architecture.md"
    )


def test_rules_only_name_existing_packages() -> None:
    known = set(ALLOWED)
    for unit, allowed in ALLOWED.items():
        assert allowed <= known, f"{unit} lists unknown packages: {sorted(allowed - known)}"
        assert unit not in allowed, f"{unit} lists itself"


def test_rules_have_no_cycles() -> None:
    ordered: set[str] = set()
    remaining = dict(ALLOWED)
    while remaining:
        ready = [unit for unit, deps in remaining.items() if deps <= ordered]
        assert ready, f"circular dependency between {sorted(remaining)}"
        ordered.update(ready)
        for unit in ready:
            del remaining[unit]


@pytest.mark.parametrize("unit", sorted(ALLOWED))
def test_sources_respect_the_dependency_rules(unit: str) -> None:
    found = [error for path in unit_files(unit) for error in violations(path, unit, ALLOWED[unit])]
    assert found == []


@pytest.mark.parametrize("unit", sorted(ALLOWED))
def test_package_imports_in_isolation(unit: str) -> None:
    result = subprocess.run(  # noqa: S603  (fixed interpreter and probe, unit is a known name)
        [sys.executable, "-c", PROBE, unit],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    loaded = set(json.loads(result.stdout))
    assert loaded <= ALLOWED[unit] | {unit}, f"importing absa.{unit} also loaded {sorted(loaded)}"


def test_app_only_uses_the_inference_interface() -> None:
    found = [
        error
        for path in sorted(APP.rglob("*.py"))
        for error in violations(path, "app", APP_ALLOWED)
    ]
    assert found == []


def test_document_lists_the_same_rules() -> None:
    assert documented_rules() == ALLOWED


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from absa.models import lstm_attention", ["data may not"]),
        ("from absa._lazy import lazy_getattr\nfrom absa.config import RANDOM_SEED", []),
        ("import absa.extraction.llm", ["data may not"]),
        ("from absa import inference", ["data may not"]),
        ("def f():\n    from absa.inference import predict", ["data may not"]),
        ("from .schema import validate_record", ["use absolute imports"]),
        ("from absa.config import RANDOM_SEED\nimport absa.data.schema", []),
        ("from absa import __version__\nimport json", []),
    ],
)
def test_checker_detects_forbidden_imports(source: str, expected: list[str]) -> None:
    errors = source_violations(source, "probe.py", "data", ALLOWED["data"])
    assert len(errors) == len(expected)
    assert all(fragment in error for fragment, error in zip(expected, errors, strict=True))
