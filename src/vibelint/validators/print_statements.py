"""
Validator for print statement usage in Python code.

Detects print statements that should be replaced with proper logging
for better maintainability, configurability, and production readiness.

vibelint/validators/print_statements.py
"""

import ast
from pathlib import Path

from ..error_codes import VBL701, VBL702, VBL703

__all__ = [
    "PrintValidationResult",
    "validate_print_statements",
    "PrintStatementVisitor",
]

ValidationIssue = tuple[str, str]


class PrintValidationResult:
    """
    Result of a print statement validation.

    vibelint/validators/print_statements.py
    """

    def __init__(self) -> None:
        """Initialize an empty print statement validation result."""
        self.issues: list[ValidationIssue] = []
        self.errors: list[ValidationIssue] = []
        self.warnings: list[ValidationIssue] = []
        self.print_count: int = 0
        self.print_locations: list[tuple[int, str, str]] = []  # (line_num, context, suggestion)

    def add_print_issue(self, code: str, message: str, line_num: int, context: str, suggestion: str) -> None:
        """Add a print statement issue to the result."""
        self.issues.append((code, message))

        # Categorize print issues
        if code == "VBL702":  # Print in __main__ block - warning
            self.warnings.append((code, message))
        else:  # VBL701, VBL703 - errors
            self.errors.append((code, message))

        self.print_locations.append((line_num, context, suggestion))
        self.print_count += 1


class PrintStatementVisitor(ast.NodeVisitor):
    """
    AST visitor to detect print statements and function calls.

    vibelint/validators/print_statements.py
    """

    def __init__(self) -> None:
        """Initialize the visitor."""
        self.print_calls: list[tuple[int, str, str | None]] = []  # (line_num, call_type, context)
        self.current_function: str | None = None
        self.current_class: str | None = None

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call nodes to detect print() calls."""
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            context = self._get_context()
            call_type = "print_function"

            # Analyze the print call to provide better suggestions
            suggestion = self._analyze_print_call(node)

            self.print_calls.append((node.lineno, call_type, context, suggestion))

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions to track context."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions to track context."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to track context."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def _get_context(self) -> str:
        """Get the current code context (class/function)."""
        contexts = []
        if self.current_class:
            contexts.append(f"class {self.current_class}")
        if self.current_function:
            contexts.append(f"function {self.current_function}")

        if contexts:
            return " in " + " -> ".join(contexts)
        return " at module level"

    def _analyze_print_call(self, node: ast.Call) -> str:
        """Analyze print call to provide appropriate logging suggestion."""
        # Check if it's likely debug/info/error based on content
        if node.args:
            first_arg = node.args[0]

            # Try to extract string content for analysis
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                content = first_arg.value.lower()
                if any(word in content for word in ['error', 'fail', 'exception', 'critical']):
                    return "logger.error()"
                elif any(word in content for word in ['warn', 'warning']):
                    return "logger.warning()"
                elif any(word in content for word in ['debug', 'trace']):
                    return "logger.debug()"
                else:
                    return "logger.info()"

            # Check for f-strings or formatted content
            elif isinstance(first_arg, ast.JoinedStr):  # f-string
                return "logger.info() with f-string"

        # Default suggestion
        return "logger.info()"


def validate_print_statements(
    file_path: Path,
    content: str | None = None,
    ignore_test_files: bool = True,
    ignore_cli_files: bool = True,
    allowed_patterns: set[str] | None = None
) -> PrintValidationResult:
    """
    Validate print statement usage in a Python file.

    Args:
        file_path: Path to the Python file to validate
        content: Optional file content (if already loaded)
        ignore_test_files: If True, ignore files that appear to be tests
        ignore_cli_files: If True, ignore files that appear to be CLI scripts
        allowed_patterns: Optional set of patterns to allow (e.g., {"if __name__", "debug"})

    Returns:
        PrintValidationResult containing any issues found
    """
    result = PrintValidationResult()

    # Check if we should ignore this file
    if ignore_test_files and _is_test_file(file_path):
        return result

    if ignore_cli_files and _is_cli_file(file_path):
        return result

    try:
        if content is None:
            content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        result.add_print_issue(
            VBL701,
            f"Could not read file {file_path} due to encoding issues",
            0, "", "Fix file encoding"
        )
        return result
    except Exception:
        result.add_print_issue(
            VBL701,
            f"Could not access file {file_path}",
            0, "", "Check file permissions"
        )
        return result

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        result.add_print_issue(
            VBL701,
            f"Could not parse {file_path}: {e}",
            getattr(e, 'lineno', 0), "", "Fix syntax errors"
        )
        return result

    visitor = PrintStatementVisitor()
    visitor.visit(tree)

    # Check if file has proper logging setup
    has_logging_import = _has_logging_import(content)

    for line_num, call_type, context, suggestion in visitor.print_calls:
        # Check if this print call should be allowed
        if allowed_patterns and _matches_allowed_patterns(content, line_num, allowed_patterns):
            continue

        # Check if it's in a if __name__ == "__main__" block (often acceptable for CLI)
        if _in_main_block(content, line_num):
            result.add_print_issue(
                VBL702,
                f"Print statement found in __main__ block at line {line_num}{context}. "
                f"Consider using {suggestion} for consistency.",
                line_num, context, suggestion
            )
        else:
            severity = VBL703 if has_logging_import else VBL702
            message = (
                f"Print statement found at line {line_num}{context}. "
                f"Replace with {suggestion} for better maintainability."
            )

            if not has_logging_import:
                message += " Consider adding logging import: 'import logging'"

            result.add_print_issue(severity, message, line_num, context, suggestion)

    return result


def _is_test_file(file_path: Path) -> bool:
    """Check if file appears to be a test file."""
    name = file_path.name.lower()
    parent_names = [p.name.lower() for p in file_path.parents]

    return (
        name.startswith('test_')
        or name.endswith('_test.py')
        or name == 'conftest.py'
        or 'test' in parent_names
        or 'tests' in parent_names
    )


def _is_cli_file(file_path: Path) -> bool:
    """Check if file appears to be a CLI script."""
    name = file_path.name.lower()
    cli_indicators = ['cli.py', 'main.py', '__main__.py', 'command.py', 'cmd.py']

    return (
        name in cli_indicators
        or name.endswith('_cli.py')
        or name.endswith('_cmd.py')
    )


def _has_logging_import(content: str) -> bool:
    """Check if file has logging import."""
    lines = content.lower().split('\n')
    for line in lines:
        line = line.strip()
        if (
            line.startswith('import logging')
            or line.startswith('from logging')
            or 'import logging' in line
        ):
            return True
    return False


def _in_main_block(content: str, line_num: int) -> bool:
    """Check if line is inside a if __name__ == "__main__" block."""
    lines = content.split('\n')
    main_line = -1

    # Find the __main__ block
    for i, line in enumerate(lines):
        if '__name__' in line and '__main__' in line:
            main_line = i + 1  # Convert to 1-based indexing
            break

    if main_line == -1:
        return False

    # Check if our line is after the main block and properly indented
    return line_num > main_line


def _matches_allowed_patterns(content: str, line_num: int, allowed_patterns: set[str]) -> bool:
    """Check if print statement matches any allowed patterns."""
    lines = content.split('\n')
    if line_num > len(lines):
        return False

    line = lines[line_num - 1].strip().lower()

    for pattern in allowed_patterns:
        if pattern.lower() in line:
            return True

    return False
