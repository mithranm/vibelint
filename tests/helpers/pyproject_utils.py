"""
Test utilities for manipulating pyproject.toml files.

Extracted from cli_utils.py to follow single responsibility principle.
"""
import sys
from pathlib import Path
from typing import Any

import pytest

# Conditional TOML library import
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

try:
    import tomli_w
except ImportError:
    tomli_w = None


def modify_pyproject(file_path: Path, section: str, updates: dict[str, Any]) -> None:
    """Modify a pyproject.toml file with new configuration."""
    if tomllib is None or tomli_w is None:
        pytest.skip("TOML libraries not available")

    if not file_path.exists():
        config = {}
    else:
        with open(file_path, "rb") as f:
            config = tomllib.load(f)

    # Navigate to or create the section
    current = config
    parts = section.split(".")
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    # Update the final section
    if parts[-1] not in current:
        current[parts[-1]] = {}
    current[parts[-1]].update(updates)

    # Write back
    with open(file_path, "wb") as f:
        tomli_w.dump(config, f)