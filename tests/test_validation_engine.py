"""Tests for validation engine."""

from pathlib import Path

import pytest

from vibelint.config import Config
from vibelint.validation_engine import PluginValidationRunner
from vibelint.validators import Finding, Severity


def test_validation_runner_initialization(sample_config: Config, temp_dir: Path):
    """Test PluginValidationRunner initialization."""
    runner = PluginValidationRunner(sample_config, temp_dir)

    assert runner.config == sample_config
    assert runner.project_root == temp_dir
    assert isinstance(runner.findings, list)
    assert len(runner.findings) == 0


def test_run_validation_on_valid_file(
    sample_config: Config, temp_dir: Path, sample_python_file: Path
):
    """Test running validation on a valid Python file."""
    runner = PluginValidationRunner(sample_config, temp_dir)

    findings = runner.run_validation([sample_python_file])

    assert isinstance(findings, list)
    # All findings should be Finding objects
    assert all(isinstance(f, Finding) for f in findings)


def test_run_validation_skips_non_python(sample_config: Config, temp_dir: Path):
    """Test that validation skips non-Python files."""
    # Create a non-Python file
    txt_file = temp_dir / "test.txt"
    txt_file.write_text("Not Python code")

    runner = PluginValidationRunner(sample_config, temp_dir)
    findings = runner.run_validation([txt_file])

    # Should have no findings since file is skipped
    assert len(findings) == 0


def test_run_validation_handles_missing_file(sample_config: Config, temp_dir: Path):
    """Test that validation handles missing files gracefully."""
    missing_file = temp_dir / "nonexistent.py"

    runner = PluginValidationRunner(sample_config, temp_dir)
    findings = runner.run_validation([missing_file])

    # Should not crash, just skip the file
    assert isinstance(findings, list)


def test_get_summary(sample_config: Config, temp_dir: Path):
    """Test getting summary of findings."""
    runner = PluginValidationRunner(sample_config, temp_dir)

    # Manually add some findings
    runner.findings = [
        Finding(
            rule_id="TEST-1",
            message="Test",
            file_path=Path("test.py"),
            severity=Severity.WARN,
        ),
        Finding(
            rule_id="TEST-2",
            message="Test",
            file_path=Path("test.py"),
            severity=Severity.WARN,
        ),
        Finding(
            rule_id="TEST-3",
            message="Test",
            file_path=Path("test.py"),
            severity=Severity.BLOCK,
        ),
    ]

    summary = runner.get_summary()

    assert summary["WARN"] == 2
    assert summary["BLOCK"] == 1


def test_has_blocking_issues(sample_config: Config, temp_dir: Path):
    """Test checking for blocking issues."""
    runner = PluginValidationRunner(sample_config, temp_dir)

    # No findings initially
    assert not runner.has_blocking_issues()

    # Add a blocking finding
    runner.findings = [
        Finding(
            rule_id="TEST",
            message="Test",
            file_path=Path("test.py"),
            severity=Severity.BLOCK,
        )
    ]

    assert runner.has_blocking_issues()


def test_get_exit_code(sample_config: Config, temp_dir: Path):
    """Test getting appropriate exit code."""
    runner = PluginValidationRunner(sample_config, temp_dir)

    # No findings = success
    assert runner.get_exit_code() == 0

    # Warnings = success
    runner.findings = [
        Finding(
            rule_id="TEST",
            message="Test",
            file_path=Path("test.py"),
            severity=Severity.WARN,
        )
    ]
    assert runner.get_exit_code() == 0

    # Blocking = failure
    runner.findings = [
        Finding(
            rule_id="TEST",
            message="Test",
            file_path=Path("test.py"),
            severity=Severity.BLOCK,
        )
    ]
    assert runner.get_exit_code() == 1


def test_format_output_human(sample_config: Config, temp_dir: Path):
    """Test formatting output in human format."""
    runner = PluginValidationRunner(sample_config, temp_dir)

    runner.findings = [
        Finding(
            rule_id="TEST",
            message="Test finding",
            file_path=Path("test.py"),
            line=10,
            severity=Severity.WARN,
        )
    ]

    output = runner.format_output("human")

    assert isinstance(output, str)
    assert len(output) > 0


def test_format_output_json(sample_config: Config, temp_dir: Path):
    """Test formatting output in JSON format."""
    import json

    runner = PluginValidationRunner(sample_config, temp_dir)

    runner.findings = [
        Finding(
            rule_id="TEST",
            message="Test finding",
            file_path=Path("test.py"),
            line=10,
            severity=Severity.WARN,
        )
    ]

    output = runner.format_output("json")

    # Should be valid JSON
    result = json.loads(output)
    assert isinstance(result, dict)
