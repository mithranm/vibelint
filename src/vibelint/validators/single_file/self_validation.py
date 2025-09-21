"""
Self-validation hooks for vibelint.

This module implements validation hooks that ensure vibelint follows
its own coding standards and architectural principles.

Key principles enforced:
- Single-file validators must not access other files
- Project-wide validators must implement validate_project()
- No emoji in code or comments (project rule)
- Proper validator categorization
- Adherence to killeraiagent project standards

vibelint/src/vibelint/self_validation.py
"""

import logging
import re
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional
from ...plugin_system import BaseValidator, Finding, Severity


logger = logging.getLogger(__name__)


class SelfValidationHook:
    """
    Hook that validates vibelint's own code against its standards.

    This runs automatically when vibelint analyzes its own codebase
    to ensure we follow our own rules.
    """

    def __init__(self):
        self.project_root = None
        self.violations_found = []

    def should_apply_self_validation(self, file_path: Path) -> bool:
        """Check if self-validation should apply to this file."""
        try:
            # Check if we're analyzing vibelint's own code
            path_str = str(file_path.absolute())
            return (
                'vibelint' in path_str and
                '/src/vibelint/' in path_str and
                file_path.suffix == '.py'
            )
        except Exception:
            return False

    def validate_single_file_validator(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Validate that single-file validators don't violate isolation."""
        if not self._is_validator_file(file_path):
            return

        # Check if this is a single-file validator
        if self._is_single_file_validator(content):
            # Check for project context violations
            violations = self._check_project_context_violations(content)
            for violation in violations:
                yield Finding(
                    file_path=file_path,
                    line=violation['line'],
                    message=f"Single-file validator violates isolation: {violation['issue']}",
                    rule_id="VIBELINT-SINGLE-FILE-ISOLATION",
                    severity=Severity.BLOCK,
                    suggestion="Single-file validators should not access other files or require project context"
                )

    def validate_project_standards(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Validate adherence to killeraiagent project standards."""
        if not self.should_apply_self_validation(file_path):
            return

        # Check for emoji violations (project rule: no emoji)
        emoji_violations = self._check_emoji_violations(content)
        for line_num, line_content in emoji_violations:
            yield Finding(
                file_path=file_path,
                line=line_num,
                message="Code contains emoji characters (violates project standards)",
                rule_id="VIBELINT-NO-EMOJI",
                severity=Severity.BLOCK,
                suggestion="Remove emoji characters from code and comments"
            )

        # Check for proper absolute path usage
        path_violations = self._check_path_violations(content)
        for line_num, issue in path_violations:
            yield Finding(
                file_path=file_path,
                line=line_num,
                message=f"Path usage issue: {issue}",
                rule_id="VIBELINT-ABSOLUTE-PATHS",
                severity=Severity.WARN,
                suggestion="Use absolute paths for file operations"
            )

    def validate_validator_categorization(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Validate that validators are properly categorized."""
        if not self._is_validator_file(file_path):
            return

        # Check if validator is in correct directory
        is_single_file = self._is_single_file_validator(content)
        is_project_wide = self._is_project_wide_validator(content)

        path_str = str(file_path)
        in_single_file_dir = '/single_file/' in path_str
        in_project_wide_dir = '/project_wide/' in path_str
        in_architecture_dir = '/architecture/' in path_str

        if is_single_file and not in_single_file_dir and not in_architecture_dir:
            yield Finding(
                file_path=file_path,
                line=1,
                message="Single-file validator should be in validators/single_file/ directory",
                rule_id="VIBELINT-VALIDATOR-ORGANIZATION",
                severity=Severity.WARN,
                suggestion="Move to validators/single_file/ or implement project-wide validation"
            )

        if is_project_wide and not in_project_wide_dir and not in_architecture_dir:
            yield Finding(
                file_path=file_path,
                line=1,
                message="Project-wide validator should be in validators/project_wide/ directory",
                rule_id="VIBELINT-VALIDATOR-ORGANIZATION",
                severity=Severity.WARN,
                suggestion="Move to validators/project_wide/ or implement single-file validation"
            )

    def _is_validator_file(self, file_path: Path) -> bool:
        """Check if file is a validator."""
        return (
            '/validators/' in str(file_path) and
            file_path.name != '__init__.py' and
            file_path.suffix == '.py'
        )

    def _is_single_file_validator(self, content: str) -> bool:
        """Check if validator is designed for single-file analysis."""
        patterns = [
            r'class\s+\w+Validator.*BaseValidator',
            r'def validate\(self, file_path.*content.*\)',
            r'requires_project_context.*False'
        ]

        project_patterns = [
            r'validate_project',
            r'project_files',
            r'requires_project_context.*True'
        ]

        has_single_file_indicators = any(re.search(pattern, content) for pattern in patterns)
        has_project_indicators = any(re.search(pattern, content) for pattern in project_patterns)

        return has_single_file_indicators and not has_project_indicators

    def _is_project_wide_validator(self, content: str) -> bool:
        """Check if validator is designed for project-wide analysis."""
        patterns = [
            r'validate_project',
            r'project_files.*Dict',
            r'requires_project_context.*True'
        ]
        return any(re.search(pattern, content) for pattern in patterns)

    def requires_project_context(self) -> bool:
        """Self-validation does not require project context."""
        return False

    def _check_project_context_violations(self, content: str) -> List[Dict[str, Any]]:
        """Check for violations of single-file validator isolation."""
        violations = []
        lines = content.split('\n')

        violation_patterns = [
            (r'import.*discovery', 'Should not import discovery module'),
            (r'import.*project_map', 'Should not import project mapping'),
            (r'glob\.glob', 'Should not use glob to find other files'),
            (r'os\.walk', 'Should not walk directory tree'),
            (r'Path.*glob', 'Should not glob for other files'),
            (r'open\(.*\.py', 'Should not open other Python files'),
        ]

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith('#'):  # Skip comments
                continue

            for pattern, issue in violation_patterns:
                if re.search(pattern, line):
                    violations.append({
                        'line': line_num,
                        'issue': issue,
                        'content': line
                    })

        return violations

    def _check_emoji_violations(self, content: str) -> List[tuple]:
        """Check for emoji characters in code."""
        violations = []
        lines = content.split('\n')

        # Unicode ranges for emoji
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "]+", flags=re.UNICODE
        )

        for line_num, line in enumerate(lines, 1):
            if emoji_pattern.search(line):
                violations.append((line_num, line.strip()))

        return violations

    def _check_path_violations(self, content: str) -> List[tuple]:
        """Check for improper path usage."""
        violations = []
        lines = content.split('\n')

        # Patterns that suggest relative path usage in file operations
        problematic_patterns = [
            (r'open\(["\'][^/]', 'Relative path in open()'),
            (r'Path\(["\'][^/]', 'Relative path in Path()'),
            (r'glob\(["\'][^/]', 'Relative path in glob()'),
        ]

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith('#'):  # Skip comments
                continue

            for pattern, issue in problematic_patterns:
                if re.search(pattern, line):
                    violations.append((line_num, issue))

        return violations


class VibelintSelfValidator(BaseValidator):
    """
    Validator that applies vibelint's self-validation hooks.

    This ensures vibelint follows its own standards when analyzing
    its own codebase.
    """

    def __init__(self, severity: Severity = Severity.WARN):
        super().__init__(severity)
        self.hook = SelfValidationHook()

    @property
    def rule_id(self) -> str:
        return "VIBELINT-SELF-VALIDATION"

    @property
    def name(self) -> str:
        return "Vibelint Self-Validation"

    @property
    def description(self) -> str:
        return "Ensures vibelint follows its own coding standards and architectural principles"

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Apply all self-validation checks."""
        if not self.hook.should_apply_self_validation(file_path):
            return

        # Apply all self-validation checks
        yield from self.hook.validate_single_file_validator(file_path, content)
        yield from self.hook.validate_project_standards(file_path, content)
        yield from self.hook.validate_validator_categorization(file_path, content)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m vibelint.validators.self_validation <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    try:
        content = file_path.read_text(encoding='utf-8')
        validator = VibelintSelfValidator()
        findings = list(validator.validate(file_path, content))

        if findings:
            print(f"Self-validation violations found in {file_path}:")
            for finding in findings:
                print(f"  Line {finding.line}: {finding.message}")
                if finding.suggestion:
                    print(f"    Suggestion: {finding.suggestion}")
            sys.exit(1)
        else:
            print(f"Self-validation passed for {file_path}")
            sys.exit(0)

    except Exception as e:
        print(f"Error during self-validation: {e}")
        sys.exit(1)