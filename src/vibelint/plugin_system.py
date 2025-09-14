"""
Plugin system for vibelint validators and formatters.

This module provides the core interfaces and discovery mechanisms for
extending vibelint with custom validators and output formatters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Dict, List, Optional, Any
import importlib.metadata

__all__ = ["Severity", "Finding", "BaseValidator", "BaseFormatter", "PluginManager", "plugin_manager"]


class Severity(Enum):
    """Severity levels for validation findings."""
    OFF = "OFF"
    INFO = "INFO"
    WARN = "WARN"
    BLOCK = "BLOCK"

    def __lt__(self, other):
        """Enable sorting by severity."""
        order = {"OFF": 0, "INFO": 1, "WARN": 2, "BLOCK": 3}
        return order[self.value] < order[other.value]


@dataclass
class Finding:
    """A validation finding from a validator."""
    rule_id: str
    message: str
    file_path: Path
    line: int = 0
    column: int = 0
    severity: Severity = Severity.WARN
    context: str = ""
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary for JSON output."""
        return {
            "rule": self.rule_id,
            "level": self.severity.value,
            "path": str(self.file_path),
            "line": self.line,
            "column": self.column,
            "msg": self.message,
            "context": self.context,
            "suggestion": self.suggestion
        }


class BaseValidator(ABC):
    """Abstract base class for all vibelint validators."""

    rule_id: str = ""
    name: str = ""
    description: str = ""
    default_severity: Severity = Severity.WARN

    def __init__(self, severity: Optional[Severity] = None):
        """Initialize validator with optional severity override."""
        self.severity = severity or self.default_severity
        if not self.rule_id:
            raise ValueError(f"Validator {self.__class__.__name__} must define rule_id")

    @abstractmethod
    def validate(self, file_path: Path, content: str) -> Iterator[Finding]:
        """
        Validate a file and yield findings.

        Args:
            file_path: Path to the file being validated
            content: File content as string

        Yields:
            Finding objects for any issues found
        """
        pass

    def create_finding(
        self,
        message: str,
        file_path: Path,
        line: int = 0,
        column: int = 0,
        context: str = "",
        suggestion: Optional[str] = None,
        severity: Optional[Severity] = None
    ) -> Finding:
        """Helper method to create a finding with this validator's rule_id."""
        return Finding(
            rule_id=self.rule_id,
            message=message,
            file_path=file_path,
            line=line,
            column=column,
            severity=severity or self.severity,
            context=context,
            suggestion=suggestion
        )


class BaseFormatter(ABC):
    """Abstract base class for output formatters."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def format_results(self, findings: List[Finding], summary: Dict[str, int]) -> str:
        """
        Format validation results for output.

        Args:
            findings: List of all findings from validation
            summary: Summary counts by severity level

        Returns:
            Formatted output string
        """
        pass


class PluginManager:
    """Manages discovery and loading of validator and formatter plugins."""

    def __init__(self):
        self._validators: Dict[str, type[BaseValidator]] = {}
        self._formatters: Dict[str, type[BaseFormatter]] = {}
        self._loaded = False

    def load_plugins(self):
        """Load all available plugins via entry points."""
        if self._loaded:
            return

        # Load validators
        for entry_point in importlib.metadata.entry_points(group="vibelint.validators"):
            try:
                validator_class = entry_point.load()
                if issubclass(validator_class, BaseValidator):
                    self._validators[validator_class.rule_id] = validator_class
            except Exception:
                # Skip invalid plugins silently
                pass

        # Load formatters
        for entry_point in importlib.metadata.entry_points(group="vibelint.formatters"):
            try:
                formatter_class = entry_point.load()
                if issubclass(formatter_class, BaseFormatter):
                    self._formatters[formatter_class.name] = formatter_class
            except Exception:
                # Skip invalid plugins silently
                pass

        self._loaded = True

    def get_validator(self, rule_id: str) -> Optional[type[BaseValidator]]:
        """Get validator class by rule ID."""
        self.load_plugins()
        return self._validators.get(rule_id)

    def get_all_validators(self) -> Dict[str, type[BaseValidator]]:
        """Get all available validator classes."""
        self.load_plugins()
        return self._validators.copy()

    def get_formatter(self, name: str) -> Optional[type[BaseFormatter]]:
        """Get formatter class by name."""
        self.load_plugins()
        return self._formatters.get(name)

    def get_all_formatters(self) -> Dict[str, type[BaseFormatter]]:
        """Get all available formatter classes."""
        self.load_plugins()
        return self._formatters.copy()


# Global plugin manager instance
plugin_manager = PluginManager()
