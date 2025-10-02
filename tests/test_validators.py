"""Tests for validator system."""

from pathlib import Path

import pytest

from vibelint.validators import (
    BaseValidator,
    Finding,
    Severity,
    get_all_validators,
    get_validator,
)


def test_get_all_validators():
    """Test getting all registered validators."""
    validators = get_all_validators()

    assert isinstance(validators, dict)
    assert len(validators) > 0

    # Check that some core validators are registered
    validator_ids = list(validators.keys())
    assert any("EMOJI" in vid for vid in validator_ids)
    assert any("EXPORTS" in vid for vid in validator_ids)


def test_get_validator():
    """Test getting a specific validator."""
    validators = get_all_validators()

    # Get first validator
    first_id = list(validators.keys())[0]
    validator_class = get_validator(first_id)

    assert validator_class is not None
    assert hasattr(validator_class, "rule_id")
    assert hasattr(validator_class, "default_severity")


def test_get_nonexistent_validator():
    """Test getting a validator that doesn't exist."""
    validator = get_validator("NONEXISTENT-RULE")
    assert validator is None


def test_finding_creation():
    """Test Finding dataclass."""
    finding = Finding(
        rule_id="TEST-RULE",
        message="Test message",
        file_path=Path("test.py"),
        line=10,
        column=5,
        severity=Severity.WARN,
        context="test context",
        suggestion="Fix this",
    )

    assert finding.rule_id == "TEST-RULE"
    assert finding.message == "Test message"
    assert finding.line == 10
    assert finding.severity == Severity.WARN


def test_finding_to_dict():
    """Test Finding.to_dict() method."""
    finding = Finding(
        rule_id="TEST-RULE",
        message="Test message",
        file_path=Path("test.py"),
        line=10,
        severity=Severity.WARN,
    )

    result = finding.to_dict()

    assert result["rule"] == "TEST-RULE"
    assert result["msg"] == "Test message"
    assert result["line"] == 10
    assert result["level"] == "WARN"
    assert "path" in result


def test_severity_ordering():
    """Test Severity enum ordering."""
    assert Severity.OFF < Severity.INFO
    assert Severity.INFO < Severity.WARN
    assert Severity.WARN < Severity.BLOCK


def test_base_validator_initialization(sample_config):
    """Test BaseValidator initialization."""

    class TestValidator(BaseValidator):
        rule_id = "TEST-RULE"
        default_severity = Severity.WARN

        def validate(self, file_path: Path, content: str, config=None):
            yield self.create_finding(
                message="Test finding",
                file_path=file_path,
                line=1,
            )

    validator = TestValidator(config=sample_config)

    assert validator.rule_id == "TEST-RULE"
    assert validator.severity == Severity.WARN
    assert validator.config == sample_config


def test_base_validator_custom_severity(sample_config):
    """Test BaseValidator with custom severity."""

    class TestValidator(BaseValidator):
        rule_id = "TEST-RULE"
        default_severity = Severity.WARN

        def validate(self, file_path: Path, content: str, config=None):
            return []

    validator = TestValidator(severity=Severity.BLOCK, config=sample_config)

    assert validator.severity == Severity.BLOCK


def test_validator_create_finding(sample_config):
    """Test BaseValidator.create_finding() method."""

    class TestValidator(BaseValidator):
        rule_id = "TEST-RULE"
        default_severity = Severity.INFO

        def validate(self, file_path: Path, content: str, config=None):
            return []

    validator = TestValidator(config=sample_config)
    finding = validator.create_finding(
        message="Test message",
        file_path=Path("test.py"),
        line=5,
        column=10,
    )

    assert finding.rule_id == "TEST-RULE"
    assert finding.severity == Severity.INFO
    assert finding.line == 5
    assert finding.column == 10
