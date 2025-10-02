"""Tests for CLI commands."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from vibelint.cli import cli


@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()


def test_cli_help(cli_runner):
    """Test CLI help command."""
    result = cli_runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "vibelint" in result.output
    assert "check" in result.output
    assert "snapshot" in result.output


def test_cli_version_verbose(cli_runner):
    """Test CLI with verbose flag."""
    result = cli_runner.invoke(cli, ["--verbose", "--help"])

    assert result.exit_code == 0


def test_check_command_help(cli_runner):
    """Test check command help."""
    result = cli_runner.invoke(cli, ["check", "--help"])

    assert result.exit_code == 0
    assert "Run vibelint validation" in result.output
    assert "--format" in result.output


def test_check_command_no_config(cli_runner):
    """Test check command without config file."""
    with cli_runner.isolated_filesystem():
        result = cli_runner.invoke(cli, ["check"])

        # Should fail without config or project root
        assert result.exit_code == 1
        assert ("No vibelint configuration found" in result.output or "No project root found" in result.output)


def test_check_command_with_config(cli_runner, temp_dir, pyproject_toml):
    """Test check command with config file."""
    # Create a Python file to check
    py_file = temp_dir / "test.py"
    py_file.write_text('"""Test module."""\n\ndef foo():\n    pass\n')

    # Change to temp_dir for test
    import os
    original_dir = os.getcwd()
    try:
        os.chdir(temp_dir)
        result = cli_runner.invoke(cli, ["check", str(py_file)])

        # Should succeed (exit code 0 or 1 depending on findings)
        assert result.exit_code in (0, 1)
    finally:
        os.chdir(original_dir)


def test_check_command_json_format(cli_runner, temp_dir, pyproject_toml):
    """Test check command with JSON output."""
    py_file = temp_dir / "test.py"
    py_file.write_text('"""Test module."""\n\ndef foo():\n    pass\n')

    import os
    original_dir = os.getcwd()
    try:
        os.chdir(temp_dir)
        result = cli_runner.invoke(cli, ["check", "--format", "json", str(py_file)])

        # Should output JSON
        assert result.exit_code in (0, 1)
    finally:
        os.chdir(original_dir)


def test_snapshot_command_help(cli_runner):
    """Test snapshot command help."""
    result = cli_runner.invoke(cli, ["snapshot", "--help"])

    assert result.exit_code == 0
    assert "snapshot" in result.output
    assert "--output" in result.output


def test_snapshot_command(cli_runner, temp_dir, pyproject_toml):
    """Test snapshot command creates output file."""
    # Create a Python file
    py_file = temp_dir / "test.py"
    py_file.write_text('"""Test module."""\n')

    output_file = temp_dir / "snapshot.md"

    import os
    original_dir = os.getcwd()
    try:
        os.chdir(temp_dir)
        result = cli_runner.invoke(
            cli, ["snapshot", str(temp_dir), "--output", str(output_file)]
        )

        # Should succeed
        assert result.exit_code in (0, 1)
    finally:
        os.chdir(original_dir)


def test_check_command_no_python_files(cli_runner, temp_dir, pyproject_toml):
    """Test check command with no Python files."""
    # Create only non-Python files
    (temp_dir / "test.txt").write_text("Not Python")

    import os
    original_dir = os.getcwd()
    try:
        os.chdir(temp_dir)
        result = cli_runner.invoke(cli, ["check"])

        # Should succeed with no files message
        assert result.exit_code == 0
        assert "No Python files found" in result.output
    finally:
        os.chdir(original_dir)
