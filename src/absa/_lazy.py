"""Shared PEP 562 lazy-attribute helper for optional heavy-dependency submodules.

Used by package ``__init__`` modules that need to expose names from a
submodule requiring an optional dependency (lxml, torch, scikit-learn)
without importing that submodule - and therefore without requiring the
dependency - just because the package itself was imported.
"""

from __future__ import annotations

import importlib
from typing import Any


def lazy_getattr(package_name: str, lazy_names: dict[str, str], name: str) -> Any:
    """Return ``name`` by importing the submodule ``lazy_names[name]`` on first use.

    Intended as a package's module-level ``__getattr__`` (PEP 562):
    ``lazy_names`` maps each lazily-exported attribute name to the dotted
    submodule that defines it. Raises ``AttributeError`` for any other
    name, as ``__getattr__`` must. A genuinely missing optional dependency
    still surfaces as a normal ``ImportError`` the moment that name is
    actually used, not when the package itself is imported.
    """
    submodule_name = lazy_names.get(name)
    if submodule_name is None:
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")
    return getattr(importlib.import_module(submodule_name), name)
