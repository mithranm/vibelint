"""
Fallback pattern analyzer for detecting silent failure patterns.

Identifies problematic fallback patterns that mask errors and cause silent failures:
1. Bare except blocks that return default values
2. Function calls with broad exception catching that return None/empty
3. Dictionary.get() chains that hide missing configuration
4. Try/except blocks that swallow important errors
5. Default parameter fallbacks that mask real issues

vibelint/validators/fallback_analyzer.py
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ...plugin_system import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)

__all__ = ["FallbackAnalyzer"]


class FallbackAnalyzer(BaseValidator, ast.NodeVisitor):
    """
    Detects problematic fallback patterns that cause silent failures.
    """

    rule_id = "FALLBACK-SILENT-FAILURE"
    default_severity = Severity.WARN

    def __init__(
        self, severity: Optional[Severity] = None, config: Optional[Dict[str, Any]] = None
    ) -> None:
        BaseValidator.__init__(self, severity, config)
        ast.NodeVisitor.__init__(self)
        self.findings: List[Finding] = []
        self.current_file: Optional[Path] = None

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Analyze file for problematic fallback patterns."""
        self.findings = []
        self.current_file = file_path

        try:
            tree = ast.parse(content)
            self.visit(tree)
        except SyntaxError as e:
            logger.debug(f"Syntax error in {file_path}: {e}")
            pass

        return iter(self.findings)

    def visit_Try(self, node: ast.Try) -> None:
        """Analyze try/except blocks for problematic fallback patterns."""

        # Pattern 1: Bare except that returns a default value
        for handler in node.handlers:
            if handler.type is None:  # bare except:
                if self._handler_returns_default(handler):
                    assert self.current_file is not None
                    self.findings.append(
                        self.create_finding(
                            message="Bare except block returns default value, potentially masking important errors. Consider catching specific exceptions or at least logging the error.",
                            file_path=self.current_file,
                            line=node.lineno,
                        )
                    )

            # Pattern 2: Exception handler that swallows exceptions silently
            elif self._handler_is_silent(handler):
                exception_type = "Exception" if handler.type is None else ast.unparse(handler.type)
                assert self.current_file is not None
                self.findings.append(
                    self.create_finding(
                        message=f"Exception handler for {exception_type} swallows errors silently. Consider logging the error or re-raising if appropriate.",
                        file_path=self.current_file,
                        line=node.lineno,
                    )
                )

            # Pattern 3: Too broad exception catching
            elif self._handler_too_broad(handler):
                assert self.current_file is not None
                self.findings.append(
                    self.create_finding(
                        message="Catching 'Exception' or 'BaseException' is too broad and may hide programming errors. Catch specific exception types instead.",
                        file_path=self.current_file,
                        line=node.lineno,
                    )
                )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Analyze function calls for problematic patterns."""

        # Pattern 4: dict.get() chains that might hide config issues
        if self._is_chained_dict_get(node):
            assert self.current_file is not None
            self.findings.append(
                self.create_finding(
                    message="Chained dict.get() calls with defaults may hide missing configuration. Consider validating required configuration explicitly.",
                    file_path=self.current_file,
                    line=node.lineno,
                )
            )

        # Pattern 5: getattr with default that might hide attribute errors
        if self._is_problematic_getattr(node):
            assert self.current_file is not None
            self.findings.append(
                self.create_finding(
                    message="getattr() with default value may hide missing attributes. Consider checking if attribute should exist before accessing.",
                    file_path=self.current_file,
                    line=node.lineno,
                )
            )

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyze function definitions for problematic default patterns."""

        # Pattern 6: Functions that return None by default in multiple paths
        return_statements = self._get_return_statements(node)
        none_returns = [r for r in return_statements if self._returns_none_or_empty(r)]

        if len(none_returns) >= 2 and len(return_statements) >= 3:
            assert self.current_file is not None
            self.findings.append(
                self.create_finding(
                    message=f"Function '{node.name}' has multiple return paths that return None/empty values. This may indicate error conditions being masked as normal returns.",
                    file_path=self.current_file,
                    line=node.lineno,
                )
            )

        self.generic_visit(node)

    def _handler_returns_default(self, handler: ast.ExceptHandler) -> bool:
        """Check if exception handler returns a default value."""
        if not handler.body:
            return False

        # Look for return statements with default values
        for stmt in handler.body:
            if isinstance(stmt, ast.Return):
                if stmt.value is None:  # return None
                    return True
                elif isinstance(stmt.value, (ast.Constant, ast.List, ast.Dict, ast.Set)):
                    return True  # return literal value
        return False

    def _handler_is_silent(self, handler: ast.ExceptHandler) -> bool:
        """Check if exception handler swallows exceptions without logging."""
        if not handler.body:
            return True  # Empty handler

        # Check if handler only has pass, return, or continue
        for stmt in handler.body:
            if isinstance(stmt, (ast.Pass, ast.Return, ast.Continue, ast.Break)):
                continue
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # String literal (probably a comment)
            else:
                return False  # Has some other statement (probably logging)

        return True  # Only has pass/return/continue statements

    def _handler_too_broad(self, handler: ast.ExceptHandler) -> bool:
        """Check if exception handler catches too broadly."""
        if handler.type is None:
            return True  # bare except

        if isinstance(handler.type, ast.Name):
            return handler.type.id in ["Exception", "BaseException"]

        return False

    def _is_chained_dict_get(self, node: ast.Call) -> bool:
        """Check for chained dict.get() calls like config.get('a', {}).get('b', default)."""
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and len(node.args) >= 2:

            # Check if this is called on another .get() call
            if isinstance(node.func.value, ast.Call):
                inner_call = node.func.value
                if isinstance(inner_call.func, ast.Attribute) and inner_call.func.attr == "get":
                    return True

        return False

    def _is_problematic_getattr(self, node: ast.Call) -> bool:
        """Check for getattr calls with defaults that might hide real issues."""
        if (
            isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 3
        ):  # getattr(obj, name, default)

            # If the default is None or a simple literal, it might be problematic
            default_arg = node.args[2]
            return isinstance(default_arg, (ast.Constant, ast.NameConstant)) or (
                isinstance(default_arg, ast.Name) and default_arg.id == "None"
            )

        return False

    def _get_return_statements(self, function_node: ast.FunctionDef) -> List[ast.Return]:
        """Get all return statements in a function."""
        returns = []

        class ReturnVisitor(ast.NodeVisitor):
            """A visitor that returns a value from visiting nodes in a syntax tree."""

            def visit_Return(self, node: ast.Return):
                """Visit a Return node in the AST and process it."""
                returns.append(node)
                self.generic_visit(node)

        ReturnVisitor().visit(function_node)
        return returns

    def _returns_none_or_empty(self, return_node: ast.Return) -> bool:
        """Check if return statement returns None or empty container."""
        if return_node.value is None:
            return True

        if isinstance(return_node.value, ast.Constant):
            return return_node.value.value is None

        if isinstance(return_node.value, ast.Name):
            return return_node.value.id == "None"

        if isinstance(return_node.value, (ast.List, ast.Set, ast.Tuple)):
            return len(return_node.value.elts) == 0
        elif isinstance(return_node.value, ast.Dict):
            return len(return_node.value.keys) == 0 and len(return_node.value.values) == 0

        return False