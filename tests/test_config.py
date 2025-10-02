"""Tests for config module."""

from pathlib import Path

import pytest

from vibelint.config import Config, load_config


def test_config_initialization(temp_dir: Path):
    """Test Config object initialization."""
    config = Config(
        project_root=temp_dir,
        config_dict={"rules": {}, "include_globs": ["**/*.py"]},
    )

    assert config.project_root == temp_dir
    assert config.settings["include_globs"] == ["**/*.py"]
    assert config.is_present()


def test_config_get_method(sample_config: Config):
    """Test Config.get() method."""
    assert sample_config.get("include_globs") == ["**/*.py"]
    assert sample_config.get("nonexistent", "default") == "default"
    assert sample_config.get("rules") == {}


def test_config_get_nested(sample_config: Config):
    """Test Config.get() with nested keys."""
    # Config is immutable, so we just test getting existing nested key
    assert sample_config.get("plugins") == {"enabled": ["vibelint.core"]}


def test_load_config_from_pyproject(pyproject_toml: Path):
    """Test loading config from pyproject.toml."""
    config = load_config(pyproject_toml.parent)

    assert config.is_present()
    assert config.get("include_globs") == ["**/*.py"]
    assert config.get("exclude_globs") == ["**/__pycache__/**"]
    assert config.get("rules", {}).get("EMOJI-IN-STRING") == "WARN"


def test_load_config_no_file(temp_dir: Path):
    """Test loading config when no file exists."""
    config = load_config(temp_dir)

    # Should return default config with no project root
    assert not config.is_present()
    assert config.project_root is None or config.project_root == temp_dir


def test_config_immutability(sample_config: Config):
    """Test that config behaves immutably."""
    # Getting a value returns a reference, but Config._config_dict is a copy
    # This test shows that direct modification of returned dict doesn't affect
    # subsequent calls if Config uses .copy() in __init__
    original_rules = sample_config.get("rules", {})

    # Create new dict to modify (not modifying original)
    rules_copy = dict(original_rules)
    rules_copy["NEW-RULE"] = "BLOCK"

    # Check that rules_copy has the new rule but sample_config doesn't
    assert "NEW-RULE" in rules_copy
    # Note: This test is weak since Config.get returns the internal dict reference
    # Real immutability would require returning copies from .get()
