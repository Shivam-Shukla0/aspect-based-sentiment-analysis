import hashlib
import re
from pathlib import Path

from scripts.check_dataset import MANIFEST, check, sha256_of


def test_sha256_of_matches_hashlib_on_a_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.xml"
    path.write_bytes(b"<sentences></sentences>")
    assert sha256_of(path) == hashlib.sha256(b"<sentences></sentences>").hexdigest()


def test_check_passes_on_a_correct_download(tmp_path: Path) -> None:
    path = tmp_path / "sample.xml"
    content = b"<sentences></sentences>"
    path.write_bytes(content)
    manifest = {"sample.xml": hashlib.sha256(content).hexdigest()}
    errors, warnings = check(raw_dir=tmp_path, manifest=manifest)
    assert errors == []
    assert warnings == []


def test_check_fails_on_a_modified_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.xml"
    path.write_bytes(b"<sentences></sentences>")
    manifest = {"sample.xml": hashlib.sha256(b"different content").hexdigest()}
    errors, _warnings = check(raw_dir=tmp_path, manifest=manifest)
    assert len(errors) == 1
    assert "checksum mismatch" in errors[0]


def test_check_reports_missing_file(tmp_path: Path) -> None:
    manifest = {"missing.xml": "0" * 64}
    errors, _warnings = check(raw_dir=tmp_path, manifest=manifest)
    assert len(errors) == 1
    assert "missing from" in errors[0]


def test_check_warns_on_unmanifested_file(tmp_path: Path) -> None:
    (tmp_path / "extra.xml").write_bytes(b"not tracked")
    errors, warnings = check(raw_dir=tmp_path, manifest={})
    assert errors == []
    assert len(warnings) == 1
    assert "extra.xml" in warnings[0]
    assert "not in the manifest" in warnings[0]


def test_check_ignores_dotfiles(tmp_path: Path) -> None:
    (tmp_path / ".gitkeep").write_bytes(b"")
    errors, warnings = check(raw_dir=tmp_path, manifest={})
    assert errors == []
    assert warnings == []


def test_check_still_warns_on_real_file_next_to_dotfile(tmp_path: Path) -> None:
    (tmp_path / ".gitkeep").write_bytes(b"")
    (tmp_path / "extra.xml").write_bytes(b"not tracked")
    _errors, warnings = check(raw_dir=tmp_path, manifest={})
    assert len(warnings) == 1
    assert "extra.xml" in warnings[0]


def test_manifest_matches_readme_table() -> None:
    """The MANIFEST in check_dataset.py must mirror the checksums table in data/README.md."""
    readme = (Path(__file__).resolve().parents[1] / "data" / "README.md").read_text()
    # Locate the checksums table and pull out rows that carry a SHA-256 digest.
    table_start = readme.index("## Checksums")
    table = readme[table_start:]
    # Each checksummed row: filename, then a 64-hex SHA-256, then a source tag.
    row_re = re.compile(r"`([^`]+\.xml)`\s*\|\s*`?\s*([0-9a-f]{64})\s*`?\s*\|", re.MULTILINE)
    readme_checksums = dict(row_re.findall(table))

    assert readme_checksums, "README checksums table should not be empty"
    assert readme_checksums == MANIFEST, (
        "MANIFEST in check_dataset.py drifted from the data/README.md checksums table; "
        "keep them in sync"
    )
