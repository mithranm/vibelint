"""
Configuration handling for vibelint.

vibelint/config.py
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import copy

# Import tomllib for Python 3.11+, fallback to tomli for earlier versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


DEFAULT_CONFIG = {
    "package_root": "",
    "allowed_shebangs": ["#!/usr/bin/env python3"],
    "docstring_regex": r"^[A-Z].+\.$",
    "include_globs": ["**/*.py"],
    "exclude_globs": [
        "**/tests/**",
        "**/migrations/**",
        "**/site-packages/**",
        "**/dist-packages/**",
    ],
    "large_dir_threshold": 500,
}


def find_pyproject_toml(directory: Path) -> Optional[Path]:
    """
    Find the pyproject.toml file by traversing up from the given directory.

    vibelint/config.py
    """
    current = directory.absolute()
    while current != current.parent:
        pyproject_path = current / "pyproject.toml"
        if pyproject_path.exists():
            return pyproject_path
        current = current.parent
    return None


def load_user_config() -> Dict[str, Any]:
    """
    Load user configuration from ~/.config/vibelint/config.toml if it exists.

    vibelint/config.py
    """
    config_path = Path.home() / ".config" / "vibelint" / "config.toml"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "rb") as f:
            user_config = tomllib.load(f)
        return user_config.get("tool", {}).get("vibelint", {})
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def load_project_config(directory: Path) -> Dict[str, Any]:
    """
    Load project configuration from pyproject.toml if it exists.

    vibelint/config.py
    """
    pyproject_path = find_pyproject_toml(directory)
    if not pyproject_path:
        return {}

    try:
        with open(pyproject_path, "rb") as f:
            project_config = tomllib.load(f)
        return project_config.get("tool", {}).get("vibelint", {})
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def load_config(directory: Path) -> Dict[str, Any]:
    """
    Load configuration by merging default, user, and project configurations.

    vibelint/config.py
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    user_config = load_user_config()
    project_config = load_project_config(directory)

    # Update with user config first, then project config (project has higher precedence)
    config.update(user_config)
    config.update(project_config)

    return config
