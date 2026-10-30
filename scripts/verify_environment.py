"""Check that this Python environment can run the project and record its exact versions.

Usage, from the repository root:

    python -m scripts.verify_environment [--require-gpu] [--record DIR]

The exit status is 0 when Python and every core package are present (and a CUDA GPU is
visible, if ``--require-gpu`` is given), otherwise 1. Nothing here needs network access.
``--record DIR`` also writes ``DIR/environment.txt`` with the Python and platform versions,
the device, the git commit and every installed package, to keep beside a result.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

MIN_PYTHON = (3, 10)
# Distribution names, kept equal to requirements.txt (a test checks this).
CORE_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lxml",
    "torch",
    "transformers",
    "accelerate",
    "sentencepiece",
    "matplotlib",
)
PROJECT = "absa"
RECORD_NAME = "environment.txt"


class Device(NamedTuple):
    """What PyTorch can see: whether it imports, whether CUDA works, and a description."""

    importable: bool
    cuda: bool
    summary: str


class Report(NamedTuple):
    """Human readable lines plus the problems that make the environment unusable."""

    lines: list[str]
    errors: list[str]


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Return the installed version of each distribution, or None if it is not installed."""
    return {name: _version(name) for name in names}


def load_torch(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> ModuleType | None:
    """Import PyTorch, returning None if it is missing or its native libraries fail to load."""
    try:
        return importer("torch")
    except (ImportError, OSError):
        return None


def describe_device(torch: ModuleType | None) -> Device:
    """Summarise the compute device that ``torch`` reports."""
    if torch is None:
        return Device(importable=False, cuda=False, summary="unknown (torch cannot be imported)")
    if not torch.cuda.is_available():
        return Device(importable=True, cuda=False, summary="CUDA not available, running on CPU")
    count = torch.cuda.device_count()
    summary = (
        f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}, "
        f"{count} device{'' if count == 1 else 's'})"
    )
    return Device(importable=True, cuda=True, summary=summary)


def _package_problems(versions: Mapping[str, str | None], device: Device) -> list[str]:
    missing = [name for name in CORE_PACKAGES if not versions.get(name)]
    errors = [f"missing core package(s): {', '.join(missing)}"] if missing else []
    if versions.get("torch") and not device.importable:
        errors.append("torch is installed but cannot be imported")
    return errors


def _absent(name: str) -> str:
    return "not installed (run: pip install -e . --no-deps)" if name == PROJECT else "MISSING"


def build_report(
    python: tuple[int, int, int],
    versions: Mapping[str, str | None],
    device: Device,
    *,
    require_gpu: bool = False,
) -> Report:
    """Build the report lines and the list of problems for the given environment facts."""
    lines = [f"{'python':<15} {'.'.join(map(str, python))}"]
    lines += [f"{name:<15} {version or _absent(name)}" for name, version in versions.items()]
    lines.append(f"{'device':<15} {device.summary}")
    errors = _package_problems(versions, device)
    if python < MIN_PYTHON:
        errors.append(f"Python {'.'.join(map(str, MIN_PYTHON))} or newer is required")
    if require_gpu and not device.cuda:
        errors.append("a CUDA GPU is required (--require-gpu) but none is available")
    return Report(lines, errors)


def freeze() -> list[str]:
    """Return ``name==version`` for every installed distribution except this project."""
    pins: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name and name.lower() != PROJECT:
            pins[name.lower()] = f"{name}=={dist.version}"
    return [pins[key] for key in sorted(pins)]


def git_commit(run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> str:
    """Return the current git commit, or ``unknown`` outside a git checkout."""
    try:
        result = run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def write_record(directory: Path, lines: Sequence[str]) -> Path:
    """Write the environment record into ``directory`` and return the file path."""
    directory.mkdir(parents=True, exist_ok=True)
    header = [f"# {line}" for line in lines]
    header += [f"# {'platform':<15} {platform.platform()}", f"# {'git commit':<15} {git_commit()}"]
    path = directory / RECORD_NAME
    path.write_text("\n".join([*header, *freeze()]) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks, print the report and return the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--require-gpu", action="store_true", help="fail if no CUDA GPU is visible")
    parser.add_argument("--record", type=Path, metavar="DIR", help=f"write DIR/{RECORD_NAME}")
    args = parser.parse_args(argv)

    versions = package_versions((*CORE_PACKAGES, PROJECT))
    report = build_report(
        sys.version_info[:3],
        versions,
        describe_device(load_torch()),
        require_gpu=args.require_gpu,
    )
    print("\n".join(report.lines))
    if args.record:
        print(f"recorded {write_record(args.record, report.lines)}")
    for error in report.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print("environment ok" if not report.errors else f"{len(report.errors)} problem(s) found")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
