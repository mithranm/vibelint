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
    """
    current = directory.absolute()
    while current != current.parent:
        pyproject_path = current / "pyproject.toml"
        if pyproject_path.exists():
            return pyproject_path
        current = current.parent
    return None


def load_toml_config(config_path: Path, section: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a TOML file.
    
    Args:
        config_path: Path to the TOML config file
        section: Optional section to read (e.g. "tool.vibelint")
        
    Returns:
        Dictionary with configuration values
    """
    try:
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
            
        if section:
            # Navigate nested sections (e.g. "tool.vibelint")
            parts = section.split('.')
            for part in parts:
                if part in config_data:
                    config_data = config_data[part]
                else:
                    return {}
        
        return config_data
    except (tomllib.TOMLDecodeError, OSError) as e:
        print(f"Warning: Error loading {config_path}: {str(e)}", file=sys.stderr)
        return {}


def load_user_config() -> Dict[str, Any]:
    """
    Load user configuration from ~/.config/vibelint/config.toml if it exists.
    """
    config_path = Path.home() / ".config" / "vibelint" / "config.toml"
    if not config_path.exists():
        return {}

    return load_toml_config(config_path)


def load_project_config(directory: Path) -> Dict[str, Any]:
    """
    Load project configuration from pyproject.toml under [tool.vibelint].
    """
    pyproject_path = find_pyproject_toml(directory)
    if not pyproject_path:
        return {}
        
    return load_toml_config(pyproject_path, "tool.vibelint")


def load_config(directory: Path) -> Dict[str, Any]:
    """
    Load configuration by merging default, user, and project configurations.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    user_config = load_user_config()
    project_config = load_project_config(directory)

    # Update with user config first, then project config (project has higher precedence)
    config.update(user_config)
    config.update(project_config)

    return config
