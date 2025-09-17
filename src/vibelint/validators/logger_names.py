"""
Logger name validator using BaseValidator plugin system.

Detects hardcoded logger names that should use __name__ instead
for proper module hierarchy and maintainability.

vibelint/src/vibelint/validators/logger_names.py
"""

import ast
from pathlib import Path
from typing import Iterator

from ..plugin_system import BaseValidator, Finding, Severity

__all__ = ["LoggerNameValidator"]


class LoggerNameValidator(BaseValidator):
    """Validator for detecting hardcoded logger names."""

    rule_id = "LOGGER-NAME"
    name = "Logger Name Checker"
    description = "Detects get_logger() calls with hardcoded strings instead of __name__"
    default_severity = Severity.BLOCK

    def __init__(self, severity=None, config=None):
        super().__init__(severity, config)

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Validate a single file for hardcoded logger names."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        class LoggerNameVisitor(ast.NodeVisitor):
            """Visitor class that traverses an AST to collect all logger names used in the code."""

            def __init__(self, validator):
                """Initialize logger name visitor."""
                self.validator = validator
                self.logger_names = []
                self.findings = []

            def visit_Call(self, node: ast.Call):
                # Check for get_logger() calls
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "get_logger"
                    and len(node.args) == 1
                ):

                    arg = node.args[0]

                    # If argument is a string literal (not __name__)
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        self.findings.append(
                            Finding(
                                rule_id=self.validator.rule_id,
                                severity=self.validator.severity,
                                message=f"Use __name__ instead of hardcoded string '{arg.value}' for logger name",
                                file_path=file_path,
                                line=node.lineno,
                                suggestion="get_logger(__name__)",
                            )
                        )

                # Also check for attribute access like logging.getLogger()
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in ["getLogger", "get_logger"]
                    and len(node.args) == 1
                ):

                    arg = node.args[0]

                    # If argument is a string literal (not __name__)
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        # Skip if this is in logging infrastructure getting specific loggers
                        if "logging.py" in str(file_path) and "kaia." in arg.value:
                            pass  # This is logging infrastructure, skip
                        else:
                            self.findings.append(
                                Finding(
                                    rule_id=self.validator.rule_id,
                                    severity=Severity.WARN,  # Less severe for standard logging
                                    message=f"Consider using __name__ instead of hardcoded string '{arg.value}' for logger name",
                                    file_path=file_path,
                                    line=node.lineno,
                                    suggestion="Use __name__ for consistent logger hierarchy",
                                )
                            )

                self.generic_visit(node)

        visitor = LoggerNameVisitor(self)
        visitor.visit(tree)
        yield from visitor.findings
