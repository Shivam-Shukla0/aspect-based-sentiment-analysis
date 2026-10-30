"""Seeding helper so experiments are repeatable across Python, NumPy and PyTorch."""

from __future__ import annotations

import importlib
import importlib.util
import os
import random

from absa.config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed every random number generator that is installed.

    NumPy and PyTorch are optional here so the lightweight CI jobs can import this module.
    PyTorch is also asked for deterministic algorithms; operations without a deterministic
    implementation only warn. Full GPU determinism can still depend on hardware and drivers.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if importlib.util.find_spec("numpy") is not None:
        importlib.import_module("numpy").random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        torch.manual_seed(seed)  # also seeds CUDA devices
        torch.use_deterministic_algorithms(mode=True, warn_only=True)
