import hashlib
from pathlib import Path

from scripts.check_dataset import check, sha256_of


def test_sha256_of_matches_hashlib_on_a_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.xml"
    path.write_bytes(b"<sentences></sentences>")
    assert sha256_of(path) == hashlib.sha256(b"<sentences></sentences>").hexdigest()


def test_check_passes_on_a_correct_download(tmp_path: Path) -> None:
    path = tmp_path / "sample.xml"
    content = b"<sentences></sentences>"
    path.write_bytes(content)
    manifest = {"sample.xml": hashlib.sha256(content).hexdigest()}
    assert check(raw_dir=tmp_path, manifest=manifest) == []


def test_check_fails_on_a_modified_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.xml"
    path.write_bytes(b"<sentences></sentences>")
    manifest = {"sample.xml": hashlib.sha256(b"different content").hexdigest()}
    errors = check(raw_dir=tmp_path, manifest=manifest)
    assert len(errors) == 1
    assert "checksum mismatch" in errors[0]


def test_check_reports_missing_file(tmp_path: Path) -> None:
    manifest = {"missing.xml": "0" * 64}
    errors = check(raw_dir=tmp_path, manifest=manifest)
    assert len(errors) == 1
    assert "missing from" in errors[0]


def test_check_ignores_files_not_in_manifest(tmp_path: Path) -> None:
    (tmp_path / "extra.xml").write_bytes(b"not tracked")
    assert check(raw_dir=tmp_path, manifest={}) == []
