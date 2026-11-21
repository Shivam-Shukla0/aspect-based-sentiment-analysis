import re
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts import verify_environment as env

ROOT = Path(__file__).resolve().parents[1]
CPU = env.Device(importable=True, cuda=False, summary="CUDA not available, running on CPU")
GPU = env.Device(importable=True, cuda=True, summary="Tesla T4 (CUDA 12.4, 1 device)")
ALL_PRESENT = dict.fromkeys(env.CORE_PACKAGES, "1.0")
PYTHON = (3, 12, 3)


def fake_torch(*, cuda: bool) -> ModuleType:
    module = ModuleType("torch")
    module.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: cuda,
        device_count=lambda: 2,
        get_device_name=lambda index: f"Fake GPU {index}",
    )
    module.version = SimpleNamespace(cuda="12.4")  # type: ignore[attr-defined]
    return module


def test_core_packages_match_requirements() -> None:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    names = {line.strip() for line in text.splitlines() if line.strip() and line[0] != "#"}
    assert set(env.CORE_PACKAGES) == names - {"pytest"}


def test_report_for_a_correct_cpu_environment() -> None:
    report = env.build_report(PYTHON, ALL_PRESENT, CPU)
    assert report.errors == []
    assert report.lines[0].startswith("python")
    assert "3.12.3" in report.lines[0]
    assert any(line.startswith("torch") for line in report.lines)
    assert report.lines[-1].startswith("device")
    assert "CPU" in report.lines[-1]


def test_missing_core_package_is_an_error() -> None:
    versions = {**ALL_PRESENT, "transformers": None}
    report = env.build_report(PYTHON, versions, CPU)
    assert any("transformers" in error for error in report.errors)
    assert any(line.startswith("transformers") and "MISSING" in line for line in report.lines)


def test_project_not_installed_is_reported_but_not_an_error() -> None:
    report = env.build_report(PYTHON, {**ALL_PRESENT, "absa": None}, CPU)
    assert report.errors == []
    assert any(line.startswith("absa") and "pip install -e" in line for line in report.lines)


def test_old_python_is_an_error() -> None:
    report = env.build_report((3, 9, 18), ALL_PRESENT, CPU)
    assert any("Python 3.10" in error for error in report.errors)


def test_torch_installed_but_not_importable_is_an_error() -> None:
    broken = env.Device(importable=False, cuda=False, summary="unknown")
    report = env.build_report(PYTHON, ALL_PRESENT, broken)
    assert any("cannot be imported" in error for error in report.errors)


def test_require_gpu() -> None:
    assert any(
        "CUDA GPU" in e for e in env.build_report(PYTHON, ALL_PRESENT, CPU, require_gpu=True).errors
    )
    assert env.build_report(PYTHON, ALL_PRESENT, GPU, require_gpu=True).errors == []


def test_package_versions_reads_metadata() -> None:
    versions = env.package_versions(["pytest", "definitely-not-installed-xyz"])
    assert re.match(r"\d", versions["pytest"] or "")
    assert versions["definitely-not-installed-xyz"] is None


def test_describe_device() -> None:
    assert not env.describe_device(None).importable
    assert env.describe_device(fake_torch(cuda=False)) == CPU
    gpu = env.describe_device(fake_torch(cuda=True))
    assert gpu.cuda
    assert gpu.summary == "Fake GPU 0 (CUDA 12.4, 2 devices)"


def test_load_torch() -> None:
    module = ModuleType("torch")
    assert env.load_torch(lambda _name: module) is module

    def broken(_name: str) -> ModuleType:
        raise OSError("libcudart.so: cannot open shared object file")

    assert env.load_torch(broken) is None
    assert env.load_torch(lambda _name: (_ for _ in ()).throw(ImportError("no torch"))) is None


def test_freeze_is_sorted_pins_without_the_project() -> None:
    pins = env.freeze()
    assert any(pin.lower().startswith("pytest==") for pin in pins)
    assert not any(pin.lower().startswith("absa==") for pin in pins)
    assert pins == sorted(pins, key=lambda pin: pin.split("==")[0].lower())


def test_git_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", env.git_commit())

    def no_git(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git")

    def not_a_repo(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(128, "git")

    def empty(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout="\n")

    assert env.git_commit(no_git) == "unknown"
    assert env.git_commit(not_a_repo) == "unknown"
    assert env.git_commit(empty) == "unknown"


def test_write_record_creates_nested_directory(tmp_path: Path) -> None:
    path = env.write_record(tmp_path / "runs" / "lstm_1", ["python 3.12.3", "device CPU"])
    assert path == tmp_path / "runs" / "lstm_1" / "environment.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == ["# python 3.12.3", "# device CPU"]
    assert any(line.startswith("# platform") for line in lines)
    assert any(line.startswith("# git commit") for line in lines)
    assert any(line.lower().startswith("pytest==") for line in lines)


@pytest.fixture
def healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "package_versions", lambda names: dict.fromkeys(names, "1.0"))
    monkeypatch.setattr(env, "load_torch", lambda: fake_torch(cuda=False))


@pytest.mark.usefixtures("healthy")
def test_main_succeeds_on_a_correct_environment(capsys: pytest.CaptureFixture[str]) -> None:
    assert env.main([]) == 0
    assert "environment ok" in capsys.readouterr().out


@pytest.mark.usefixtures("healthy")
def test_main_require_gpu_fails_on_cpu(capsys: pytest.CaptureFixture[str]) -> None:
    assert env.main(["--require-gpu"]) == 1
    assert "ERROR: a CUDA GPU is required" in capsys.readouterr().err


@pytest.mark.usefixtures("healthy")
def test_main_records_the_environment(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert env.main(["--record", str(tmp_path / "run")]) == 0
    assert (tmp_path / "run" / "environment.txt").is_file()
    assert "recorded" in capsys.readouterr().out


def test_main_fails_when_a_core_package_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    versions = {**ALL_PRESENT, "lxml": None}
    monkeypatch.setattr(env, "package_versions", lambda _names: versions)
    monkeypatch.setattr(env, "load_torch", lambda: fake_torch(cuda=False))
    assert env.main([]) == 1
    assert "lxml" in capsys.readouterr().err
