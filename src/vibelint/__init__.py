"""
vibelint package initialization module.

src/vibelint/__init__.py
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("vibelint")
except PackageNotFoundError:
    __version__ = "unknown"
