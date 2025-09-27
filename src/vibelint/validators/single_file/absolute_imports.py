"""
Absolute import enforcer validator.

Detects and can automatically fix relative imports to use absolute imports instead.
This prevents import spaghetti and circular dependency issues.

vibelint/src/vibelint/validators/single_file/absolute_imports.py
"""

import re
from pathlib import Path
from typing import Iterator

from vibelint.plugin_system import BaseValidator, Finding, Severity

__all__ = ["AbsoluteImportValidator"]


class AbsoluteImportValidator(BaseValidator):
    """Enforces absolute imports over relative imports."""

    rule_id = "ABSOLUTE-IMPORTS"
    name = "Absolute Import Enforcer"
    description = "Enforces absolute imports to prevent import spaghetti and circular dependencies"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for relative imports that should be absolute."""

        # Get the module path relative to the vibelint source
        try:
            # Find the vibelint source root
            vibelint_root = self._find_vibelint_root(file_path)
            if not vibelint_root:
                # Not in a vibelint project, skip validation
                return

            # Calculate module path
            relative_path = file_path.relative_to(vibelint_root)
            module_parts = list(relative_path.parts[:-1])  # Remove .py file

        except (ValueError, AttributeError):
            # Can't determine module structure, skip
            return

        lines = content.splitlines()
        for line_num, line in enumerate(lines, 1):
            # Match relative imports
            rel_import_match = re.match(r'^(\s*)from (\.+)([^.\s]*) import (.+)$', line.strip())
            if rel_import_match:
                _, dots, module_name, imports = rel_import_match.groups()

                # Calculate what the absolute import should be
                absolute_import = self._calculate_absolute_import(
                    module_parts, dots, module_name
                )

                if absolute_import:
                    yield self.create_finding(
                        message=f"Relative import detected: {line.strip()}",
                        file_path=file_path,
                        line=line_num,
                        suggestion=f"Replace with: from {absolute_import} import {imports}",
                    )

    def can_fix(self, finding: "Finding") -> bool:
        """Check if this finding can be automatically fixed."""
        return finding.rule_id == self.rule_id

    def apply_fix(self, content: str, finding: "Finding") -> str:
        """Automatically convert relative import to absolute import."""
        lines = content.splitlines(True)  # Keep line endings
        if finding.line <= len(lines):
            line = lines[finding.line - 1]

            # Extract the suggestion from the finding
            suggestion = finding.suggestion or ""
            if "Replace with:" in suggestion:
                replacement = suggestion.split("Replace with:", 1)[1].strip()

                # Get the indentation from the original line
                original_stripped = line.lstrip()
                indentation = line[:len(line) - len(original_stripped)]

                # Apply the replacement with original indentation
                lines[finding.line - 1] = indentation + replacement + '\n'

        return "".join(lines)

    def _find_vibelint_root(self, file_path: Path) -> Path:
        """Find the vibelint source root directory."""
        current = file_path

        # Walk up until we find the vibelint package root
        while current.parent != current:
            if current.name == "vibelint" and (current / "__init__.py").exists():
                return current
            # Also check if we're in a src/vibelint structure
            if (current / "src" / "vibelint" / "__init__.py").exists():
                return current / "src" / "vibelint"
            current = current.parent
        return None

    def _calculate_absolute_import(self, module_parts: list, dots: str, module_name: str) -> str:
        """Calculate the absolute import path."""
        levels_up = len(dots) - 1  # Number of parent directories to go up

        if levels_up == 0:
            # Same directory: from .module import something
            if module_name:
                absolute_module = "vibelint." + ".".join(module_parts + [module_name])
            else:
                # from . import something (importing from __init__.py)
                absolute_module = "vibelint." + ".".join(module_parts)
        else:
            # Parent directories: from ..parent.module import something
            if len(module_parts) >= levels_up:
                base_parts = module_parts[:-levels_up] if levels_up > 0 else module_parts
                if module_name:
                    absolute_module = "vibelint." + ".".join(base_parts + [module_name])
                else:
                    absolute_module = "vibelint." + ".".join(base_parts)
            else:
                # Too many levels up, can't determine
                return None

        return absolute_module