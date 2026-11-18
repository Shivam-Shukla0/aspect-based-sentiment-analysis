"""Verify files in ``data/raw/`` against the checksums recorded in ``data/README.md``.

Usage: python -m scripts.check_dataset
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path

from absa.config import DATA_RAW_DIR

# Keep in sync with the "Checksums" table in data/README.md. An entry here
# with no matching file in data/raw/ is reported as missing, not skipped;
# a file in data/raw/ with no entry here is not checked at all.
MANIFEST: Mapping[str, str] = {
    "laptops-trial.xml": "0971a30e6add454d344ac011385fb8407db4193db048c8cd8e7a82908ac5172d",
    "restaurants-trial.xml": "48d758803cd5183e9b8ee8a04520889a4b7ff371a2ff66e399ba193620e5bc41",
}


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(raw_dir: Path = DATA_RAW_DIR, manifest: Mapping[str, str] = MANIFEST) -> list[str]:
    """Return one error message per missing or modified file listed in ``manifest``."""
    errors = []
    for name, expected in manifest.items():
        path = raw_dir / name
        if not path.is_file():
            errors.append(f"{name}: missing from {raw_dir}")
            continue
        actual = sha256_of(path)
        if actual != expected:
            errors.append(f"{name}: checksum mismatch (expected {expected}, got {actual})")
    return errors


def main() -> int:  # pragma: no cover
    """Print any checksum problems found in ``data/raw/`` and return the exit code."""
    errors = check()
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print(f"ok: {len(MANIFEST)} file(s) verified")
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
