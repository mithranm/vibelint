"""Compatibility layer exposing built-in report formatters.

Historically, formatters lived in a dedicated ``formatters`` module. The
current codebase centralizes them in :mod:`vibelint.reporting`, but several
callers – including parts of the test-suite – still import from
``vibelint.formatters``. This module re-exports the public formatter API so
those imports keep working.

vibelint/src/vibelint/formatters.py
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_reporting = import_module("vibelint.reporting")

# Explicitly expose the mapping of built-in formatter names to classes so that
# existing imports (``from vibelint.formatters import BUILTIN_FORMATTERS``)
# continue to work. Additional formatter utilities are proxied lazily via
# ``__getattr__`` to avoid duplicating exports and triggering namespace
# collision warnings.
BUILTIN_FORMATTERS = _reporting.BUILTIN_FORMATTERS

__all__ = ["BUILTIN_FORMATTERS"]


def __getattr__(name: str) -> Any:
    """Proxy attribute lookups to :mod:`vibelint.reporting`.

    This keeps backward compatibility for code that imported individual
    formatter classes without introducing duplicate definitions in the
    namespace tree.
    """

    return getattr(_reporting, name)
