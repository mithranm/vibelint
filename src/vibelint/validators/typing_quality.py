"""
Type quality validator using BaseValidator plugin system.

Detects poor typing practices that reduce code clarity and type safety:
- Raw tuples instead of dataclasses/NamedTuples
- Untyped dictionaries instead of TypedDict
- Excessive use of Any
- Missing type annotations on public functions
- String literals that should be Enums

vibelint/validators/typing_quality.py
"""

import ast
from pathlib import Path
from typing import Iterator, Set

from ..plugin_system import BaseValidator, Finding, Severity

__all__ = ["TypingQualityValidator"]


class TypingQualityValidator(BaseValidator):
    """Validator for detecting poor typing practices."""

    rule_id = "TYPING-POOR-PRACTICE"
    name = "Type Quality Checker"
    description = "Detects poor typing practices that reduce code clarity and type safety"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Validate typing practices in a Python file."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Check for tuple type annotations that should be dataclasses
        visitor = _TypingVisitor()
        visitor.visit(tree)

        # Report tuple type aliases
        for line_num, name, tuple_info in visitor.tuple_type_aliases:
            if self._looks_like_data_structure(name, tuple_info):
                yield self.create_finding(
                    message=f"Type alias '{name}' uses raw Tuple{tuple_info} - consider using a dataclass or NamedTuple for better clarity",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Replace with: @dataclass class {name}: ..."
                )

        # Report untyped dictionaries
        for line_num, context in visitor.untyped_dicts:
            yield self.create_finding(
                message=f"Using untyped Dict{context} - consider using TypedDict for better type safety",
                file_path=file_path,
                line=line_num,
                suggestion="Define a TypedDict with explicit field types"
            )

        # Report excessive Any usage
        for line_num, context in visitor.any_usage:
            if not self._is_acceptable_any_usage(context):
                yield self.create_finding(
                    message=f"Using Any type{context} - specify a more precise type",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Replace Any with a specific type or Union of types",
                    severity=Severity.INFO
                )

        # Report missing type annotations on public functions
        for line_num, func_name in visitor.untyped_public_functions:
            yield self.create_finding(
                message=f"Public function '{func_name}' is missing type annotations",
                file_path=file_path,
                line=line_num,
                suggestion=f"Add type hints: def {func_name}(...) -> ReturnType:",
                severity=Severity.INFO
            )

        # Report string literals that look like enums
        enum_candidates = self._find_enum_candidates(visitor.string_constants)
        for pattern, locations in enum_candidates.items():
            if len(locations) >= 3:  # Same string pattern used 3+ times
                first_line = locations[0]
                yield self.create_finding(
                    message=f"String literal '{pattern}' used {len(locations)} times - consider using an Enum",
                    file_path=file_path,
                    line=first_line,
                    suggestion=f"Create an Enum for these related string constants",
                    severity=Severity.INFO
                )

    def _looks_like_data_structure(self, name: str, tuple_info: str) -> bool:
        """Check if a tuple type alias looks like it should be a data structure."""
        # Skip if it's a simple pair like (bool, str) for return values
        if tuple_info.count(',') == 1 and 'bool' in tuple_info.lower():
            return False

        # If the name suggests it's data (Issue, Result, Info, etc.)
        data_suffixes = ['Issue', 'Result', 'Info', 'Data', 'Record', 'Entry', 'Item']
        return any(name.endswith(suffix) for suffix in data_suffixes)

    def _is_acceptable_any_usage(self, context: str) -> bool:
        """Check if Any usage is acceptable in this context."""
        # Any is acceptable for **kwargs, *args, or when interfacing with external libs
        acceptable_patterns = ['**kwargs', '*args', 'json', 'yaml', 'config']
        return any(pattern in context.lower() for pattern in acceptable_patterns)

    def _find_enum_candidates(self, string_constants: list) -> dict:
        """Find string literals that are used repeatedly and could be enums."""
        from collections import defaultdict

        # Group by string pattern (uppercase, prefix, etc.)
        patterns = defaultdict(list)

        for line_num, value in string_constants:
            # Skip short strings and file paths
            if len(value) < 3 or '/' in value or '\\' in value:
                continue

            # Look for patterns like "ERROR", "WARNING", "INFO"
            if value.isupper() and '_' in value:
                patterns[value].append(line_num)
            # Or prefixed patterns like "RULE101", "ERR102"
            elif any(value.startswith(prefix) for prefix in ['RULE', 'ERR', 'WARN']):
                prefix = value[:3]
                patterns[f"{prefix}*"].append(line_num)

        return patterns


class _TypingVisitor(ast.NodeVisitor):
    """AST visitor to detect typing issues."""

    def __init__(self):
        self.tuple_type_aliases = []  # (line, name, tuple_info)
        self.untyped_dicts = []  # (line, context)
        self.any_usage = []  # (line, context)
        self.untyped_public_functions = []  # (line, func_name)
        self.string_constants = []  # (line, value)

    def visit_AnnAssign(self, node):
        """Visit annotated assignments to find type aliases."""
        if isinstance(node.target, ast.Name):
            name = node.target.id

            # Check for Tuple type annotations
            if self._is_tuple_annotation(node.annotation):
                tuple_info = ast.unparse(node.annotation) if hasattr(ast, 'unparse') else str(node.annotation)
                self.tuple_type_aliases.append((node.lineno, name, tuple_info))

        self.generic_visit(node)

    def visit_Assign(self, node):
        """Visit assignments to find type aliases using old syntax."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.endswith(('Issue', 'Result', 'Info')):
                # Check if it's a type alias assignment like: ValidationIssue = Tuple[str, str]
                if self._is_tuple_annotation(node.value):
                    tuple_info = ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value)
                    self.tuple_type_aliases.append((node.lineno, target.id, tuple_info))

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Visit function definitions to check for type annotations."""
        # Check if public function (doesn't start with _)
        if not node.name.startswith('_'):
            # Check if it has return type annotation
            if node.returns is None:
                self.untyped_public_functions.append((node.lineno, node.name))
            else:
                # Check for Any in return type
                if self._contains_any(node.returns):
                    context = f" in return type of {node.name}"
                    self.any_usage.append((node.lineno, context))

            # Check parameters
            for arg in node.args.args:
                if arg.annotation is None and arg.arg != 'self':
                    self.untyped_public_functions.append((node.lineno, f"{node.name}({arg.arg})"))
                elif arg.annotation and self._contains_any(arg.annotation):
                    context = f" in parameter '{arg.arg}' of {node.name}"
                    self.any_usage.append((node.lineno, context))

        self.generic_visit(node)

    def visit_Constant(self, node):
        """Visit string constants to find potential enums."""
        if isinstance(node.value, str):
            self.string_constants.append((node.lineno, node.value))
        self.generic_visit(node)

    def _is_tuple_annotation(self, node) -> bool:
        """Check if a node is a Tuple type annotation."""
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == 'Tuple':
                return True
            # Check for typing.Tuple
            if isinstance(node.value, ast.Attribute):
                if node.value.attr == 'Tuple':
                    return True
        return False

    def _contains_any(self, node) -> bool:
        """Check if a type annotation contains Any."""
        if isinstance(node, ast.Name) and node.id == 'Any':
            return True
        if isinstance(node, ast.Attribute) and node.attr == 'Any':
            return True
        # Recursively check in subscripts (like Optional[Any], List[Any])
        if isinstance(node, ast.Subscript):
            return self._contains_any(node.value) or any(
                self._contains_any(arg) for arg in
                (node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice])
            )
        return False