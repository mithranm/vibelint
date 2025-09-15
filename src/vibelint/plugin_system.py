"""
Core types and validation system for vibelint.

Simplified from the original over-engineered plugin system to focus on
essential functionality without unnecessary abstractions.

vibelint/src/vibelint/plugin_system.py
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "Severity",
    "Finding",
    "Validator",
    "Formatter",
    "BaseValidator",
    "BaseFormatter",
    "get_all_validators",
    "get_all_formatters",
    "get_validator",
    "get_formatter",
]


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
            "suggestion": self.suggestion,
        }


class Validator(Protocol):
    """Protocol for validator classes - simpler than abstract base class."""

    rule_id: str
    default_severity: Severity

    def __init__(
        self, severity: Optional[Severity] = None, config: Optional[Dict] = None
    ) -> None: ...

    def validate(self, file_path: Path, content: str, config: Any) -> Iterator[Finding]:
        """Validate a file and yield findings."""
        ...

    def create_finding(
        self,
        message: str,
        file_path: Path,
        line: int = 0,
        column: int = 0,
        context: str = "",
        suggestion: Optional[str] = None,
    ) -> Finding:
        """Create a Finding object with this validator's rule_id and severity."""
        ...


class Formatter(Protocol):
    """Protocol for formatter classes - simpler than abstract base class."""

    name: str

    def format_results(
        self, findings: List[Finding], summary: Dict[str, int], config: Optional[Any] = None
    ) -> str:
        """Format validation results for output."""
        ...


# Simple registry - no complex plugin discovery needed
_VALIDATORS: Dict[str, type] = {}
_FORMATTERS: Dict[str, type] = {}


def register_validator(validator_class: type) -> None:
    """Register a validator class."""
    _VALIDATORS[validator_class.rule_id] = validator_class


def register_formatter(formatter_class: type) -> None:
    """Register a formatter class."""
    _FORMATTERS[formatter_class.name] = formatter_class


def get_validator(rule_id: str) -> Optional[type]:
    """Get validator class by rule ID."""
    return _VALIDATORS.get(rule_id)


def get_all_validators() -> Dict[str, type]:
    """Get all registered validator classes."""
    # Lazy load validators from entry points on first access
    if not _VALIDATORS:
        _load_builtin_validators()
    return _VALIDATORS.copy()


def get_formatter(name: str) -> Optional[type]:
    """Get formatter class by name."""
    return _FORMATTERS.get(name)


def get_all_formatters() -> Dict[str, type]:
    """Get all registered formatter classes."""
    # Lazy load formatters from entry points on first access
    if not _FORMATTERS:
        _load_builtin_formatters()
    return _FORMATTERS.copy()


def _load_builtin_validators() -> None:
    """Load built-in validators from entry points."""
    import importlib.metadata

    for entry_point in importlib.metadata.entry_points(group="vibelint.validators"):
        try:
            validator_class = entry_point.load()
            if hasattr(validator_class, "rule_id"):
                _VALIDATORS[validator_class.rule_id] = validator_class
        except (ImportError, AttributeError, TypeError) as e:
            logger.warning(f"Failed to load validator '{entry_point.name}' from entry point {entry_point.value}: {e}")
            pass


def _load_builtin_formatters() -> None:
    """Load built-in formatters from entry points."""
    import importlib.metadata

    for entry_point in importlib.metadata.entry_points(group="vibelint.formatters"):
        try:
            formatter_class = entry_point.load()
            if hasattr(formatter_class, "name"):
                _FORMATTERS[formatter_class.name] = formatter_class
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"Failed to load formatter from entry point {entry_point.name}: {e}")
            pass


# Concrete base classes
class BaseValidator:
    """Base class for validators."""

    rule_id: str = ""
    default_severity: Severity = Severity.WARN

    def __init__(self, severity: Optional[Severity] = None, config: Optional[Dict] = None) -> None:
        self.severity = severity or self.default_severity
        self.config = config or {}

    def validate(self, file_path: Path, content: str, config: Any) -> Iterator[Finding]:
        """Validate a file and yield findings."""
        raise NotImplementedError

    def create_finding(
        self,
        message: str,
        file_path: Path,
        line: int = 0,
        column: int = 0,
        context: str = "",
        suggestion: Optional[str] = None,
    ) -> Finding:
        """Create a Finding object with this validator's rule_id and severity."""
        return Finding(
            rule_id=self.rule_id,
            message=message,
            file_path=file_path,
            line=line,
            column=column,
            severity=self.severity,
            context=context,
            suggestion=suggestion,
        )


class BaseFormatter(ABC):
    """Base class for formatters."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def format_results(
        self, findings: List[Finding], summary: Dict[str, int], config: Optional[Any] = None
    ) -> str:
        """Format validation results for output."""
        pass


# Legacy global manager for backward compatibility
class _LegacyPluginManager:
    """Legacy compatibility wrapper."""

    def load_plugins(self) -> None:
        """Load plugins - delegated to new system."""
        get_all_validators()
        get_all_formatters()

    def get_validator(self, rule_id: str) -> Optional[Any]:
        """Get validator by rule ID."""
        return get_validator(rule_id)

    def get_all_validators(self) -> Dict[str, type]:
        """Get all validators."""
        return get_all_validators()

    def get_formatter(self, name: str) -> Optional[Any]:
        """Get formatter by name."""
        return get_formatter(name)

    def get_all_formatters(self) -> Dict[str, type]:
        """Get all formatters."""
        return get_all_formatters()


plugin_manager = _LegacyPluginManager()
