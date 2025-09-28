"""
Comprehensive CLI tests for vibelint.

Tests both individual commands and integration workflows.
Consolidated from multiple test files to eliminate redundancy.
"""
import pytest
from click.testing import CliRunner
from pathlib import Path

from vibelint import __version__
from vibelint.cli import cli
from ...helpers.cli_assertions import assert_exit_code, assert_output_contains, clean_output


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


# Basic CLI tests
def test_cli_version(runner: CliRunner):
    """Test the --version flag."""
    result = runner.invoke(cli, ["--version"])
    assert_exit_code(result, 0)
    assert_output_contains(result, __version__)


def test_cli_help(runner: CliRunner):
    """Test the --help flag."""
    result = runner.invoke(cli, ["--help"])
    assert_exit_code(result, 0)
    assert_output_contains(result, "vibelint")
    assert_output_contains(result, "Usage:")


# Validation tests
def test_check_success(runner: CliRunner, tmp_path: Path):
    """Test check command on a project with no violations."""
    # Create minimal test project
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
    src_dir = tmp_path / "src" / "testpkg"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text('"""Test package."""\n')
    (src_dir / "module.py").write_text('"""Test module."""\n\ndef good_function():\n    """Well documented function."""\n    return True\n')

    result = runner.invoke(cli, ["check"], catch_exceptions=False, cwd=str(tmp_path))
    assert_exit_code(result, 0)


# Analysis tests
def test_snapshot_basic(runner: CliRunner, tmp_path: Path):
    """Test snapshot command basic functionality."""
    # Create minimal project
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
    (tmp_path / "src" / "test" / "__init__.py").mkdir(parents=True)
    (tmp_path / "src" / "test" / "__init__.py").write_text("")

    result = runner.invoke(cli, ["snapshot"], catch_exceptions=False, cwd=str(tmp_path))
    assert_exit_code(result, 0)
    assert_output_contains(result, "src")


def test_namespace_basic(runner: CliRunner, tmp_path: Path):
    """Test namespace command basic functionality."""
    # Create minimal project
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
    (tmp_path / "src" / "test" / "__init__.py").mkdir(parents=True)
    (tmp_path / "src" / "test" / "__init__.py").write_text("")

    result = runner.invoke(cli, ["namespace"], catch_exceptions=False, cwd=str(tmp_path))
    assert_exit_code(result, 0)


# Integration tests
def test_cli_integration_check_then_snapshot(runner: CliRunner, tmp_path: Path):
    """Integration test: run check followed by snapshot."""
    # Create test project
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
    src_dir = tmp_path / "src" / "test"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text('"""Test package."""')

    # Run check first
    check_result = runner.invoke(cli, ["check"], catch_exceptions=False, cwd=str(tmp_path))
    assert check_result.exit_code in [0, 1, 2]  # Any reasonable exit code

    # Run snapshot after
    snapshot_result = runner.invoke(cli, ["snapshot"], catch_exceptions=False, cwd=str(tmp_path))
    assert_exit_code(snapshot_result, 0)
    assert_output_contains(snapshot_result, "src")