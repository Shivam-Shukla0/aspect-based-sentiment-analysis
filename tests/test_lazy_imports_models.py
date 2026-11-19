"""Regression tests for the lazy-import fix in ``absa.models`` (torch).

Mirrors ``tests/test_lazy_imports.py``: ``import torch`` is made to fail
inside a fresh subprocess, so a torch copy already cached by this test
session cannot hide a regression.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from absa._lazy import lazy_getattr

BLOCK_TORCH = "import sys; sys.modules['torch'] = None\n"


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 (fixed, hardcoded script below; nothing user-controlled)
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_lightweight_import_works_without_torch() -> None:
    result = _run(
        BLOCK_TORCH + "import absa.models\n"
        "vocab = absa.models.Vocabulary.build(['hello world'])\n"
        "print(len(vocab))\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().isdigit()


def test_importing_absa_models_does_not_import_lstm_dataset_without_torch() -> None:
    result = _run(
        BLOCK_TORCH + "import sys\n"
        "import absa.models\n"
        "print('absa.models.lstm_dataset' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_calling_lstm_dataset_without_torch_raises_clear_error() -> None:
    result = _run(BLOCK_TORCH + "import absa.models\nabsa.models.build_embedding(10, 8, 0)\n")
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr


def test_existing_public_imports_still_work_with_torch_present() -> None:
    """``from absa.models import X`` must behave exactly as before this fix, torch installed."""
    pytest.importorskip("torch")
    from absa.models import (  # noqa: PLC0415 (needs the skip above)
        AspectDataset,
        build_embedding,
    )

    assert callable(build_embedding)
    assert isinstance(AspectDataset, type)


def test_lazy_getattr_rejects_an_unregistered_name_in_models_registry() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        lazy_getattr("absa.models", {"build_embedding": "absa.models.lstm_dataset"}, "missing")
