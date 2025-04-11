"""
Configuration loading for vibelint.

src/vibelint/config.py
"""

import sys
import copy
from pathlib import Path
from typing import Dict, Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


DEFAULT_CONFIG = {
    "include_globs": ["**/*.py"],
    "exclude_globs": ["**/tests/**", "**/migrations/**", "**/site-packages/**", "**/dist-packages/**"],
    "peek_globs": [],
    "allowed_shebangs": ["#!/usr/bin/env python3"],
    "large_dir_threshold": 500,
    "package_root": "",
}


def find_pyproject_toml(directory: Path) -> Optional[Path]:
    """
    Search upward for a pyproject.toml file.

    src/vibelint/config.py
    """
    current = directory
    while current != current.parent:
        candidate = current / "pyproject.toml"
        if candidate.exists():
            return candidate
        current = current.parent
    return None


def load_toml_config(config_path: Path, section: Optional[str] = None) -> Dict[str, Any]:
    """
    Load TOML config from file, optionally restricting to a section.

    src/vibelint/config.py
    """
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        if section:
            parts = section.split(".")
            for part in parts:
                if part in data:
                    data = data[part]
                else:
                    return {}
        return data
    except Exception:
        return {}


def load_project_config(directory: Path) -> Dict[str, Any]:
    """
    Load config from pyproject.toml under [tool.vibelint], if present.

    src/vibelint/config.py
    """
    pyproj = find_pyproject_toml(directory)
    if not pyproj:
        return {}
    return load_toml_config(pyproj, "tool.vibelint")


def load_config(directory: Path) -> Dict[str, Any]:
    """
    Merge default config with project config.

    src/vibelint/config.py
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    proj = load_project_config(directory)
    config.update(proj)
    return config
