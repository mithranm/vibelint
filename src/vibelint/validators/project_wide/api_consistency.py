"""
API consistency validator for vibelint.

Detects inconsistent API usage patterns, missing required parameters,
and architectural violations that lead to runtime failures.

vibelint/src/vibelint/validators/api_consistency.py
"""

import ast
import logging
from pathlib import Path
from typing import Iterator

from ...plugin_system import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)

__all__ = ["APIConsistencyValidator"]


def _get_function_name(node: ast.Call) -> str:
    """Extract function name from call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


class APIConsistencyValidator(BaseValidator):
    """Validator for API consistency and usage patterns."""

    rule_id = "API-CONSISTENCY"
    name = "API Consistency Checker"
    description = "Detects inconsistent API usage, missing parameters, and architectural violations"
    default_severity = Severity.WARN

    def __init__(self, severity=None, config=None):
        super().__init__(severity, config)
        # Known API signatures and their requirements
        self.known_apis = {
            "load_config": {
                "required_args": ["start_path"],
                "module": "config",
                "common_mistakes": [
                    "Called without required start_path parameter",
                    "Often needs Path('.') or similar as argument",
                ],
            },
            "create_llm_manager": {
                "required_args": ["config"],
                "module": "llm_manager",
                "common_mistakes": ["Requires config dict with llm section"],
            },
        }

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for API consistency issues."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Check function calls for API misuse
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                yield from self._check_function_call(node, file_path)

            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                yield from self._check_import_usage(node, file_path, tree)

    def _check_function_call(self, node: ast.Call, file_path: Path) -> Iterator[Finding]:
        """Check individual function calls for API consistency."""
        func_name = _get_function_name(node)

        if func_name in self.known_apis:
            api_info = self.known_apis[func_name]

            # Check required arguments
            provided_args = len(node.args)
            required_args = len(api_info["required_args"])

            if provided_args < required_args:
                missing_args = api_info["required_args"][provided_args:]
                yield self.create_finding(
                    message=f"API misuse: {func_name}() missing required arguments: {', '.join(missing_args)}",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion=f"Add required arguments: {func_name}({', '.join(api_info['required_args'])})",
                )

    def _check_import_usage(
        self, node: ast.AST, file_path: Path, tree: ast.AST
    ) -> Iterator[Finding]:
        """Check for inconsistent import and usage patterns."""
        if isinstance(node, ast.ImportFrom):
            if node.module == "config" and any(
                alias.name == "load_config" for alias in (node.names or [])
            ):
                # Check if load_config is used correctly in this file
                for call_node in ast.walk(tree):
                    if (
                        isinstance(call_node, ast.Call)
                        and isinstance(call_node.func, ast.Name)
                        and call_node.func.id == "load_config"
                    ):

                        if not call_node.args:
                            yield self.create_finding(
                                message="Configuration anti-pattern: load_config() called without start_path",
                                file_path=file_path,
                                line=call_node.lineno,
                                suggestion="Use load_config(Path('.')) or pass explicit path for config discovery",
                            )


class ConfigurationPatternValidator(BaseValidator):
    """Validator for configuration pattern consistency."""

    rule_id = "CONFIG-PATTERN"
    name = "Configuration Pattern Checker"
    description = "Ensures consistent configuration loading and usage patterns"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for configuration pattern issues."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Track how config is loaded and used
        config_loading_patterns = []
        config_usage_patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = _get_function_name(node)

                if func_name == "load_config":
                    config_loading_patterns.append(node.lineno)

                elif "config" in str(node).lower():
                    config_usage_patterns.append(node.lineno)

        # Check for multiple config loading approaches in same file
        if len(config_loading_patterns) > 1:
            yield self.create_finding(
                message="Configuration inconsistency: Multiple config loading patterns detected",
                file_path=file_path,
                line=config_loading_patterns[0],
                suggestion="Consolidate to single source of truth for configuration",
            )

        # Check for config dict creation vs proper loading
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict) and any(
                isinstance(key, ast.Constant) and key.value == "llm" for key in node.keys if key
            ):

                yield self.create_finding(
                    message="Configuration anti-pattern: Manual config dict creation detected",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Use load_config() for single source of truth instead of manual dict creation",
                )
