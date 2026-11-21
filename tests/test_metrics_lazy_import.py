"""Importing the evaluation package must not need scikit-learn."""

import subprocess
import sys

PROBE = (
    "import sys\n"
    "sys.modules['sklearn'] = None\n"
    "import absa.evaluation, absa.evaluation.metrics\n"
    "print('imported')"
)


def test_evaluation_imports_without_scikit_learn() -> None:
    result = subprocess.run(  # noqa: S603  (fixed interpreter and probe)
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"
