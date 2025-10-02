"""
Print statement validator using BaseValidator plugin system.

Detects print statements that should be replaced with proper logging
for better maintainability, configurability, and production readiness.

vibelint/src/vibelint/validators/print_statements.py
"""

import ast
import fnmatch
import re
from pathlib import Path
from typing import Iterator

from ...validators.types import BaseValidator, Finding, Severity

__all__ = ["PrintStatementValidator"]


class PrintStatementValidator(BaseValidator):
    """Validator for detecting print statements."""

    rule_id = "PRINT-STATEMENT"
    name = "Print Statement Checker"
    description = "Detects print() calls that should be replaced with logging"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Validate print statement usage in a Python file."""
        # Check if file should be excluded based on configuration
        if self._should_exclude_file(file_path):
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        visitor = _PrintVisitor()
        visitor.visit(tree)

        # Split content into lines for suppression comment checking
        lines = content.split("\n")

        for line_num, context, print_content in visitor.print_calls:
            # Check for suppression comments on the same line
            if self._has_suppression_comment(lines, line_num):
                continue

            # Check if this looks like legitimate CLI output
            if self._is_legitimate_cli_print(print_content, context, content, line_num):
                continue

            message = (
                f"Print statement found{context}. Replace with logging for better maintainability."
            )
            suggestion = "Use logger.info(), logger.debug(), or logger.error() instead"

            yield self.create_finding(
                message=message, file_path=file_path, line=line_num, suggestion=suggestion
            )

    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from print statement validation."""
        # Get exclude patterns from configuration
        print_config = self.config.get("print_validation", {})
        exclude_globs = print_config.get(
            "exclude_globs",
            [
                # Default patterns if no configuration is provided
                "test_*.py",
                "*_test.py",
                "conftest.py",
                "tests/**/*.py",
                "cli.py",
                "main.py",
                "__main__.py",
                "*_cli.py",
                "*_cmd.py",
            ],
        )

        # Check if file matches any exclude pattern
        for pattern in exclude_globs:
            # Check against file name
            if fnmatch.fnmatch(file_path.name, pattern):
                return True

            # Check against relative path pattern
            relative_path = str(file_path).replace("\\", "/")  # Normalize path separators
            if fnmatch.fnmatch(relative_path, pattern):
                return True

            # Check against path from parent directories
            for parent in file_path.parents:
                parent_relative = str(file_path.relative_to(parent)).replace("\\", "/")
                if fnmatch.fnmatch(parent_relative, pattern):
                    return True

        return False

    def _has_suppression_comment(self, lines: list[str], line_num: int) -> bool:
        """Check if the line has a suppression comment for print statements.

        Supports:
        - # vibelint: stdout  - Explicit stdout communication marker
        - # vibelint: ignore  - General vibelint suppression
        - # noqa: print       - Specific print suppression
        - # noqa              - General linting suppression
        """
        # Line numbers in AST are 1-indexed
        if line_num <= 0 or line_num > len(lines):
            return False

        line = lines[line_num - 1]

        # Check for suppression patterns in comments
        suppression_patterns = [
            r"#\s*vibelint:\s*stdout",  # Explicit stdout marker
            r"#\s*vibelint:\s*ignore",  # General vibelint ignore
            r"#\s*noqa:\s*print",  # Specific print suppression
            r"#\s*noqa(?:\s|$)",  # General noqa
            r"#\s*type:\s*ignore",  # Type ignore (sometimes used for prints)
            r"#\s*pragma:\s*no\s*cover",  # Coverage pragma
        ]

        for pattern in suppression_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

        return False

    def _is_legitimate_cli_print(
        self, print_content: str, context: str, file_content: str, line_num: int
    ) -> bool:
        """Check if a print statement appears to be legitimate CLI output."""
        # Patterns that suggest legitimate CLI usage
        cli_indicators = [
            # UI symbols and formatting
            r"[[EMOJI][TIP][EMOJI][ALERT]⭐[SUCCESS][ROCKET]]",  # Emoji indicators for user interface
            r"^[-=]{3,}",  # Headers/separators (----, ====)
            r"^\s*\*{2,}",  # Emphasis markers (***, etc.)
            r"^\s*#{2,}",  # Section headers (##, ###)
            # CLI instruction patterns
            r"(run|execute|visit|go to|open)",
            r"(http://|https://)",  # URLs
            r"(localhost|127\.0\.0\.1)",  # Local server addresses
            r"port\s+\d+",  # Port numbers
            # Status/progress indicators
            r"(starting|completed|finished|ready)",
            r"(success|error|warning|info).*:",
            r"^\s*\[.*\]",  # [INFO], [ERROR], etc.
            # Calibration/setup specific
            r"(calibration|configuration|setup)",
            r"(device|microphone|audio)",
            r"(instruction|step \d+)",
        ]

        # Function names that suggest CLI interface
        cli_function_names = [
            "show_",
            "display_",
            "print_",
            "output_",
            "start_",
            "run_",
            "main",
            "cli",
            "calibrat",
            "setup",
            "config",
            "instruction",
            "help",
            "usage",
        ]

        # Check print content against CLI patterns
        if print_content:
            for pattern in cli_indicators:
                if re.search(pattern, print_content, re.IGNORECASE | re.MULTILINE):
                    return True

        # Check if function name suggests CLI usage
        if context:
            func_name = context.replace(" in function ", "").lower()
            for cli_pattern in cli_function_names:
                if cli_pattern in func_name:
                    return True

        # Check file context - look for CLI-related imports or patterns
        file_lines = file_content.split("\n")

        # Look around the print statement for context clues
        start_line = max(0, line_num - 5)
        end_line = min(len(file_lines), line_num + 3)
        surrounding_context = "\n".join(file_lines[start_line:end_line])

        # Check for CLI-related context around the print
        context_patterns = [
            r"def\s+(show|display|print|output|start|run|main|cli)",
            r"(server|port|url|http)",
            r"(calibration|setup|config)",
            r"(instruction|help|usage)",
            r"input\s*\(",  # User input nearby
            r"argparse",  # Command line arguments
        ]

        for pattern in context_patterns:
            if re.search(pattern, surrounding_context, re.IGNORECASE):
                return True

        return False


class _PrintVisitor(ast.NodeVisitor):
    """AST visitor to detect print statements."""

    def __init__(self):
        self.print_calls = []
        self.current_function = None

    def visit_Call(self, node):
        """Visit Call nodes to detect print() function calls."""
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            context = f" in function {self.current_function}" if self.current_function else ""

            # Extract print content for analysis
            print_content = ""
            if node.args:
                try:
                    # Try to extract string literals from print arguments
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            print_content += arg.value + " "
                        elif isinstance(arg, ast.JoinedStr):  # f-strings
                            for value in arg.values:
                                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                                    print_content += value.value
                except (AttributeError, TypeError):
                    # If we can't parse the content, just use empty string
                    pass

            self.print_calls.append((node.lineno, context, print_content.strip()))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Visit FunctionDef nodes to track current function context for print detection."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
