import random

import pytest

from absa.reproducibility import set_seed


def test_python_random_is_repeatable() -> None:
    set_seed(7)
    first = [random.random() for _ in range(5)]  # noqa: S311
    set_seed(7)
    assert [random.random() for _ in range(5)] == first  # noqa: S311


def test_different_seeds_differ() -> None:
    set_seed(1)
    a = random.random()  # noqa: S311
    set_seed(2)
    assert random.random() != a  # noqa: S311


def test_numpy_is_repeatable() -> None:
    np = pytest.importorskip("numpy")
    set_seed(7)
    first = np.random.rand(5)
    set_seed(7)
    assert (np.random.rand(5) == first).all()


def test_torch_is_repeatable() -> None:
    torch = pytest.importorskip("torch")
    set_seed(7)
    first = torch.randn(5)
    set_seed(7)
    assert torch.equal(torch.randn(5), first)
    assert torch.are_deterministic_algorithms_enabled()
