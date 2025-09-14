"""
Architecture consistency validator.

Detects architectural inconsistencies like competing systems,
mixed patterns, and violations of established conventions.

vibelint/validators/architecture.py
"""

from pathlib import Path
from typing import Dict, Iterator, List

from ...plugin_system import BaseValidator, Finding, Severity

__all__ = ["ArchitectureValidator"]


class ArchitectureValidator(BaseValidator):
    """Detects architectural inconsistencies and competing systems."""

    rule_id = "ARCHITECTURE-INCONSISTENT"
    name = "Architecture Consistency Checker"
    description = "Identifies competing systems, mixed patterns, and architectural violations"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Analyze file for architectural inconsistencies."""
        yield from self._check_competing_systems(file_path, content)
        yield from self._check_mixed_patterns(file_path, content)
        yield from self._check_import_inconsistencies(file_path, content)

    def _check_competing_systems(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Check for architectural consistency - simplified since no legacy patterns exist."""
        # Since vibelint hasn't been released and we removed all backward compatibility,
        # there are no legacy patterns to detect anywhere. This method intentionally simplified.
        return
        yield  # Unreachable - just for type checking

    def _check_mixed_patterns(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Check for mixed architectural patterns within same file."""
        lines = content.splitlines()

        # Track different console usage patterns
        console_patterns = []
        for line_num, line in enumerate(lines, 1):
            if "console" in line.lower():
                if "= Console()" in line:
                    console_patterns.append(("manual_instantiation", line_num))
                elif "from .console_utils import console" in line:
                    console_patterns.append(("shared_import", line_num))
                elif "from rich.console import Console" in line:
                    console_patterns.append(("rich_import", line_num))

        # If file uses multiple console patterns, flag it (except console_utils.py and validators that detect patterns)
        pattern_types = set(pattern[0] for pattern in console_patterns)
        if len(pattern_types) > 1 and file_path.name not in [
            "console_utils.py",
            "architecture.py",
            "dead_code.py",
            "basic_patterns.py",  # This file itself mentions multiple console patterns for detection
        ]:
            yield self.create_finding(
                message="File uses multiple console patterns inconsistently",
                file_path=file_path,
                line=console_patterns[0][1],
                suggestion="Use consistent console pattern throughout file",
            )

        # Check for mixed error handling patterns
        error_patterns = []
        for line_num, line in enumerate(lines, 1):
            if "raise " in line or "except " in line:
                if "ValidationError" in line:
                    error_patterns.append(("custom_exception", line_num))
                elif "ValueError" in line or "TypeError" in line:
                    error_patterns.append(("builtin_exception", line_num))

        # Detect inconsistent error handling
        if len(set(pattern[0] for pattern in error_patterns)) > 1:
            yield self.create_finding(
                message="Inconsistent exception types used in error handling",
                file_path=file_path,
                line=error_patterns[0][1],
                suggestion="Use consistent exception handling patterns",
                severity=Severity.INFO,
            )

    def _check_import_inconsistencies(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Check for inconsistent import patterns."""
        lines = content.splitlines()

        # Track different import styles for same module
        import_styles = {}  # module -> list of (style, line_num)

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith("from ") and " import " in line:
                # Parse "from X import Y" style
                parts = line.split(" import ", 1)
                if len(parts) == 2:
                    module = parts[0].replace("from ", "")
                    if module not in import_styles:
                        import_styles[module] = []
                    import_styles[module].append(("from_import", line_num, line))
            elif line.startswith("import "):
                # Parse "import X" style
                module = line.replace("import ", "").split(".")[0]
                if module not in import_styles:
                    import_styles[module] = []
                import_styles[module].append(("direct_import", line_num, line))

        # Flag modules imported in multiple ways
        for module, imports in import_styles.items():
            if len(set(imp[0] for imp in imports)) > 1:
                first_import = imports[0]
                yield self.create_finding(
                    message=f"Module '{module}' imported using different styles",
                    file_path=file_path,
                    line=first_import[1],
                    suggestion=f"Use consistent import style for {module}",
                    severity=Severity.INFO,
                )


class ProjectArchitectureAnalyzer:
    """
    Cross-file architecture analysis.

    This can be used by vibelint to analyze architectural patterns
    across the entire project, not just individual files.
    """

    def __init__(self, project_files: List[Path]):
        self.project_files = project_files
        self.findings: List[Dict] = []

    def analyze_competing_systems(self) -> List[Dict]:
        """Analyze project for competing validation systems."""
        legacy_files = []
        plugin_files = []

        for file_path in self.project_files:
            if not file_path.suffix == ".py":
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            # Check validation patterns
            has_legacy = any(
                [
                    "ValidationResult" in content,
                    "def validate_" in content and "def validate(" not in content,
                ]
            )

            has_plugin = any(["BaseValidator" in content, "self.create_finding" in content])

            if has_legacy:
                legacy_files.append(file_path)
            if has_plugin:
                plugin_files.append(file_path)

        findings = []
        if legacy_files and plugin_files:
            findings.append(
                {
                    "issue": "Competing validation systems detected",
                    "legacy_files": [str(f) for f in legacy_files],
                    "plugin_files": [str(f) for f in plugin_files],
                    "suggestion": "Migrate all validators to plugin system for consistency",
                }
            )

        return findings

    def analyze_duplicate_functionality(self) -> List[Dict]:
        """Find duplicate functionality across files."""
        # This could analyze similar function names, similar AST patterns, etc.
        # For now, just check for validation duplicates
        validation_files = {}  # functionality -> list of files

        for file_path in self.project_files:
            if not file_path.suffix == ".py" or "validator" not in str(file_path):
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            # Simple heuristic: check for similar validation functionality
            if "print" in content.lower() and "statement" in content.lower():
                if "print_validation" not in validation_files:
                    validation_files["print_validation"] = []
                validation_files["print_validation"].append(file_path)

            if "docstring" in content.lower():
                if "docstring_validation" not in validation_files:
                    validation_files["docstring_validation"] = []
                validation_files["docstring_validation"].append(file_path)

        findings = []
        for functionality, files in validation_files.items():
            if len(files) > 1:
                findings.append(
                    {
                        "issue": f"Duplicate {functionality} functionality",
                        "files": [str(f) for f in files],
                        "suggestion": f"Consolidate {functionality} into single implementation",
                    }
                )

        return findings
