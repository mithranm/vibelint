"""
Validator for detecting dictionary anti-patterns that should use dataclasses or objects.

Dictionaries are often misused for structured data that would be better represented
as dataclasses, NamedTuples, or custom classes. This validator identifies these patterns.
"""

import ast
import logging
from pathlib import Path
from typing import Iterator, Set

from vibelint.plugin_system import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)


class DictAntipatternValidator(BaseValidator):
    """Detects dictionaries that should be dataclasses or structured objects."""

    rule_id = "DICT-ANTIPATTERN"
    description = "Detect dictionaries that should use dataclasses or objects"

    def __init__(self, config=None, severity=None):
        super().__init__(config)
        self.config = config or {}
        self.severity = severity or Severity.WARN

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """
        Validate Python file for dictionary anti-patterns.

        Args:
            file_path: Path to the Python file
            content: File content as string
            config: Optional configuration

        Yields:
            Finding objects for dictionary anti-patterns found
        """
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            logger.debug(f"Syntax error in {file_path}: {e}")
            return

        # Track dictionary literals and their keys
        dict_patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                pattern = self._analyze_dict_literal(node)
                if pattern:
                    dict_patterns.append((node, pattern))
            elif isinstance(node, ast.Call) and self._is_dict_call(node):
                pattern = self._analyze_dict_call(node)
                if pattern:
                    dict_patterns.append((node, pattern))

        # Check for anti-patterns
        for node, pattern in dict_patterns:
            findings = self._check_dict_pattern(node, pattern, file_path)
            yield from findings

    def _analyze_dict_literal(self, node: ast.Dict) -> dict:
        """Analyze a dictionary literal for patterns."""
        if not node.keys or len(node.keys) < 2:
            return None

        # Extract string keys
        string_keys = []
        for key in node.keys:
            if key is None:  # **kwargs expansion
                return None
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                string_keys.append(key.value)
            else:
                return None  # Non-string keys, not a candidate

        if len(string_keys) >= 2:
            return {
                "type": "literal",
                "keys": string_keys,
                "key_count": len(string_keys),
                "line": node.lineno,
                "col": node.col_offset
            }
        return None

    def _analyze_dict_call(self, node: ast.Call) -> dict:
        """Analyze a dict() constructor call."""
        if not (isinstance(node.func, ast.Name) and node.func.id == "dict"):
            return None

        keys = []

        # Check keyword arguments
        for keyword in node.keywords:
            if keyword.arg:  # Not **kwargs
                keys.append(keyword.arg)

        if len(keys) >= 2:
            return {
                "type": "constructor",
                "keys": keys,
                "key_count": len(keys),
                "line": node.lineno,
                "col": node.col_offset
            }
        return None

    def _is_dict_call(self, node: ast.Call) -> bool:
        """Check if this is a dict() constructor call."""
        return (isinstance(node.func, ast.Name) and node.func.id == "dict")

    def _check_dict_pattern(self, node: ast.AST, pattern: dict, file_path: Path) -> Iterator[Finding]:
        """Check if a dictionary pattern should be a dataclass."""
        keys = pattern["keys"]
        key_count = pattern["key_count"]

        # Heuristics for when a dict should be a dataclass:
        # 1. Multiple string keys (3+ is strong indicator)
        # 2. Keys follow naming conventions (snake_case, descriptive)
        # 3. No dynamic key access patterns nearby

        if key_count >= 3:
            severity = Severity.WARN
            suggestion = self._suggest_dataclass_replacement(keys)

            yield Finding(
                rule_id=self.rule_id,
                message=f"Dictionary with {key_count} fixed keys should use dataclass/NamedTuple",
                file_path=file_path,
                line=pattern["line"],
                column=pattern["col"],
                severity=severity,
                suggestion=suggestion
            )
        elif key_count == 2 and self._keys_look_structured(keys):
            # More conservative for 2-key dicts
            suggestion = self._suggest_dataclass_replacement(keys)

            yield Finding(
                rule_id=self.rule_id,
                message=f"Dictionary with structured keys '{', '.join(keys)}' might benefit from dataclass",
                file_path=file_path,
                line=pattern["line"],
                column=pattern["col"],
                severity=Severity.INFO,
                suggestion=suggestion
            )

    def _keys_look_structured(self, keys: list) -> bool:
        """Check if keys look like structured data rather than dynamic mapping."""
        # Look for descriptive, snake_case names
        structured_indicators = 0

        for key in keys:
            if len(key) > 3:  # Not single letters
                structured_indicators += 1
            if '_' in key:  # Snake case
                structured_indicators += 1
            if key in ['id', 'name', 'type', 'value', 'data', 'config', 'status',
                      'created', 'updated', 'url', 'path', 'file', 'directory']:
                structured_indicators += 1

        return structured_indicators >= len(keys)

    def _suggest_dataclass_replacement(self, keys: list) -> str:
        """Suggest a dataclass replacement."""
        fields = []
        for key in keys:
            # Generate type hints based on common patterns
            if key in ['id', 'count', 'size', 'length']:
                fields.append(f"{key}: int")
            elif key in ['name', 'path', 'url', 'type', 'status']:
                fields.append(f"{key}: str")
            elif key in ['active', 'enabled', 'valid', 'success']:
                fields.append(f"{key}: bool")
            elif key in ['data', 'config', 'params']:
                fields.append(f"{key}: Any")
            else:
                fields.append(f"{key}: Any  # TODO: specify type")

        suggestion = f'''Consider replacing with dataclass:

@dataclass
class DataStructure:
    {chr(10).join("    " + field for field in fields)}'''

        return suggestion

    def can_fix(self) -> bool:
        """Returns True if this validator can automatically fix issues."""
        return False  # Complex transformation, requires manual review