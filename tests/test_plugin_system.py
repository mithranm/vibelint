"""
Tests for the vibelint plugin system.
"""

import json
from pathlib import Path
from typing import Iterator

import pytest

from vibelint.plugin_system import BaseValidator, BaseFormatter, Finding, Severity, plugin_manager
from vibelint.rules import RuleEngine
from vibelint.formatters import HumanFormatter, JsonFormatter


class TestValidator(BaseValidator):
    """Test validator for plugin system tests."""

    rule_id = "TEST001"
    name = "Test Validator"
    description = "A test validator"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str) -> Iterator[Finding]:
        if "test_issue" in content:
            yield self.create_finding(
                message="Found test issue",
                file_path=file_path,
                line=1,
                suggestion="Remove test_issue",
            )


class TestFormatter(BaseFormatter):
    """Test formatter for plugin system tests."""

    name = "test"
    description = "Test formatter"

    def format_results(self, findings, summary):
        return f"Test format: {len(findings)} findings"


def test_finding_creation():
    """Test Finding dataclass functionality."""
    finding = Finding(
        rule_id="VBL001",
        message="Test message",
        file_path=Path("test.py"),
        line=10,
        severity=Severity.WARN,
    )

    assert finding.rule_id == "VBL001"
    assert finding.message == "Test message"
    assert finding.line == 10
    assert finding.severity == Severity.WARN

    # Test to_dict conversion
    data = finding.to_dict()
    assert data["rule"] == "VBL001"
    assert data["level"] == "WARN"
    assert data["path"] == "test.py"
    assert data["line"] == 10


def test_base_validator():
    """Test BaseValidator functionality."""
    validator = TestValidator()

    assert validator.rule_id == "TEST001"
    assert validator.severity == Severity.WARN

    # Test validation
    findings = list(validator.validate(Path("test.py"), "test_issue here"))
    assert len(findings) == 1
    assert findings[0].rule_id == "TEST001"
    assert findings[0].message == "Found test issue"

    # Test no findings
    findings = list(validator.validate(Path("test.py"), "clean code"))
    assert len(findings) == 0


def test_severity_override():
    """Test severity override in validators."""
    validator = TestValidator(severity=Severity.BLOCK)
    assert validator.severity == Severity.BLOCK

    findings = list(validator.validate(Path("test.py"), "test_issue here"))
    assert findings[0].severity == Severity.BLOCK


def test_plugin_manager_loads_formatters():
    """Test that plugin manager can load and retrieve formatters."""
    manager = plugin_manager
    
    # The plugin manager loads formatters from entry points  
    manager.load_plugins()
    
    # Test getting all formatters
    formatters = manager.get_all_formatters()
    assert isinstance(formatters, dict)
    assert len(formatters) > 0
    
    # Test getting specific formatter
    formatter_class = manager.get_formatter("human")
    assert formatter_class is not None


def test_rule_engine():
    """Test RuleEngine functionality."""
    config = {"rules": {"TEST001": "BLOCK", "TEST002": "OFF"}}

    engine = RuleEngine(config)

    # Test rule enabling/disabling
    assert engine.is_rule_enabled("TEST001")
    assert not engine.is_rule_enabled("TEST002")
    assert engine.is_rule_enabled("TEST003")  # Default enabled

    # Test severity override
    assert engine.get_rule_severity("TEST001") == Severity.BLOCK
    assert engine.get_rule_severity("TEST003", Severity.INFO) == Severity.INFO


def test_human_formatter():
    """Test HumanFormatter output."""
    formatter = HumanFormatter()

    findings = [
        Finding(
            rule_id="VBL001",
            message="Test error",
            file_path=Path("test.py"),
            line=10,
            severity=Severity.BLOCK,
        ),
        Finding(
            rule_id="VBL002",
            message="Test warning",
            file_path=Path("other.py"),
            line=5,
            severity=Severity.WARN,
        ),
    ]

    summary = {"BLOCK": 1, "WARN": 1, "INFO": 0}
    output = formatter.format_results(findings, summary)

    assert "BLOCK:" in output
    assert "WARN:" in output
    assert "VBL001" in output
    assert "VBL002" in output
    assert "test.py:10" in output


def test_json_formatter():
    """Test JsonFormatter output."""
    formatter = JsonFormatter()

    findings = [
        Finding(
            rule_id="VBL001",
            message="Test issue",
            file_path=Path("test.py"),
            line=10,
            severity=Severity.WARN,
        )
    ]

    summary = {"WARN": 1}
    output = formatter.format_results(findings, summary)

    # Parse JSON to verify structure
    data = json.loads(output)
    assert "summary" in data
    assert "findings" in data
    assert data["summary"]["WARN"] == 1
    assert len(data["findings"]) == 1
    assert data["findings"][0]["rule"] == "VBL001"
    assert data["findings"][0]["level"] == "WARN"


def test_severity_comparison():
    """Test Severity enum comparison."""
    assert Severity.OFF < Severity.INFO
    assert Severity.INFO < Severity.WARN
    assert Severity.WARN < Severity.BLOCK

    # Test sorting
    severities = [Severity.BLOCK, Severity.OFF, Severity.WARN, Severity.INFO]
    sorted_severities = sorted(severities)
    expected = [Severity.OFF, Severity.INFO, Severity.WARN, Severity.BLOCK]
    assert sorted_severities == expected
