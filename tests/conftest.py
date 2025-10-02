"""Pytest configuration and fixtures for vibelint tests."""

import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from vibelint.config import Config


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_file(temp_dir: Path) -> Path:
    """Create a sample Python file for testing."""
    file_path = temp_dir / "sample.py"
    file_path.write_text(
        '''"""Sample module for testing."""

def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
'''
    )
    return file_path


@pytest.fixture
def sample_config(temp_dir: Path) -> Config:
    """Create a sample Config object for testing."""
    return Config(
        project_root=temp_dir,
        config_dict={
            "include_globs": ["**/*.py"],
            "exclude_globs": ["**/__pycache__/**", "**/.*"],
            "rules": {},
            "plugins": {"enabled": ["vibelint.core"]},
        },
    )


@pytest.fixture
def pyproject_toml(temp_dir: Path) -> Path:
    """Create a sample pyproject.toml file."""
    config_path = temp_dir / "pyproject.toml"
    config_path.write_text(
        """[tool.vibelint]
include_globs = ["**/*.py"]
exclude_globs = ["**/__pycache__/**"]

[tool.vibelint.rules]
EMOJI-IN-STRING = "WARN"
EXPORTS-MISSING-ALL = "INFO"
"""
    )
    return config_path
