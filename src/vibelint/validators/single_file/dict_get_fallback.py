"""
Validator for detecting .get() with fallback values on typed dictionaries.

In strictly-typed code, .get() with defaults hides missing keys. This is appropriate
for ETL/JSON parsing, but not for internal typed structures which should fail fast.
"""

import ast
import logging
from pathlib import Path
from typing import Iterator

from vibelint.plugin_system import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)


class DictGetFallbackValidator(BaseValidator):
    """Detects .get() with fallback on typed dictionaries that should use direct access."""

    rule_id = "DICT-GET-FALLBACK"
    description = "Detect .get() with fallback that should use direct key access"

    def __init__(self, config=None, severity=None):
        super().__init__(config)
        self.config = config or {}
        self.severity = severity or Severity.WARN

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """
        Validate Python file for .get() antipatterns.

        Args:
            file_path: Path to the Python file
            content: File content as string
            config: Optional configuration

        Yields:
            Finding objects for .get() antipatterns found
        """
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            logger.debug(f"Syntax error in {file_path}: {e}")
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if self._is_dict_get_with_fallback(node):
                    yield from self._check_get_pattern(node, file_path)

    def _is_dict_get_with_fallback(self, node: ast.Call) -> bool:
        """Check if this is a .get() call with a fallback value."""
        # Must be a method call
        if not isinstance(node.func, ast.Attribute):
            return False

        # Method name must be 'get'
        if node.func.attr != "get":
            return False

        # Must have 2 arguments (key, default) or 1 with keywords
        if len(node.args) == 2:
            return True
        if len(node.args) == 1 and any(kw.arg == "default" for kw in node.keywords):
            return True

        return False

    def _check_get_pattern(self, node: ast.Call, file_path: Path) -> Iterator[Finding]:
        """Check if this .get() call is an antipattern."""
        # Get the key being accessed
        key = self._extract_key(node)
        fallback = self._extract_fallback(node)

        message = f"Using .get() with fallback hides missing keys - use direct access for typed structures"

        if key:
            message = f"dict.get('{key}', {fallback}) hides missing keys - use direct access dict['{key}'] for typed structures"

        suggestion = self._suggest_direct_access(key, fallback)

        yield Finding(
            rule_id=self.rule_id,
            message=message,
            file_path=file_path,
            line=node.lineno,
            column=node.col_offset,
            severity=self.severity,
            suggestion=suggestion
        )

    def _extract_key(self, node: ast.Call) -> str:
        """Extract the key from .get() call."""
        if not node.args:
            return None

        key_node = node.args[0]
        if isinstance(key_node, ast.Constant):
            return str(key_node.value)
        return None

    def _extract_fallback(self, node: ast.Call) -> str:
        """Extract the fallback value as a string."""
        # Check args
        if len(node.args) >= 2:
            fallback = node.args[1]
            return ast.unparse(fallback) if hasattr(ast, "unparse") else "..."

        # Check keywords
        for kw in node.keywords:
            if kw.arg == "default":
                return ast.unparse(kw.value) if hasattr(ast, "unparse") else "..."

        return "None"

    def _suggest_direct_access(self, key: str, fallback: str) -> str:
        """Suggest direct access replacement."""
        if not key:
            return "Consider: Use direct key access if dictionary is typed/validated"

        suggestion = f"""Consider replacing with direct access:

# If the key is required (fail fast):
value = data["{key}"]

# If the key is truly optional, use explicit check:
value = data.get("{key}")  # Returns None if missing
if "{key}" not in data:
    value = {fallback}

# Best: Use typed structures (dataclass/NamedTuple) instead of dict"""

        return suggestion

    def can_fix(self) -> bool:
        """Returns True if this validator can automatically fix issues."""
        return False  # Requires context to determine if key is required
