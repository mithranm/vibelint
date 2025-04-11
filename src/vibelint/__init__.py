"""
Vibelint package initialization module.

vibelint/__init__.py
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("vibelint")
except PackageNotFoundError:
    # Package is not installed
    try:
        from ._version import version as __version__  # type: ignore
    except ImportError:
        __version__ = "unknown"
