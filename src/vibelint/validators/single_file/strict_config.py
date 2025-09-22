"""
Strict Configuration Validator

Enforces strict configuration management by detecting and flagging fallback patterns.
All configuration should go through the CM (Configuration Management) system without fallbacks.
"""

import ast
import re
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from pathlib import Path

# Standalone versions for CLI usage
class ValidationResult(NamedTuple):
    rule_name: str
    severity: str
    message: str
    line_number: int
    column: int
    suggestion: str
    fix_suggestion: str = ""
    category: str = "general"

class CodeContext(NamedTuple):
    file_path: Path
    content: str

class ValidationRule:
    def __init__(self, name: str, description: str, category: str, severity: str):
        self.name = name
        self.description = description
        self.category = category
        self.severity = severity


class StrictConfigRule(ValidationRule):
    """Detects configuration fallbacks and enforces strict config management."""

    def __init__(self):
        super().__init__(
            name="strict-config",
            description="Enforce strict configuration management - no fallbacks",
            category="configuration",
            severity="error"
        )

    def validate(self, context: CodeContext) -> List[ValidationResult]:
        """Check for configuration fallback patterns."""
        results = []

        # Check Python files for .get() patterns with fallbacks
        if context.file_path.suffix == '.py':
            results.extend(self._check_python_config_fallbacks(context))

        # Check for hardcoded workers.dev URLs
        results.extend(self._check_hardcoded_endpoints(context))

        # Check TOML/YAML config files for hardcoded fallbacks
        if context.file_path.suffix in ['.toml', '.yaml', '.yml']:
            results.extend(self._check_config_file_fallbacks(context))

        return results

    def _check_python_config_fallbacks(self, context: CodeContext) -> List[ValidationResult]:
        """Check Python code for config.get() patterns with fallbacks."""
        results = []

        try:
            tree = ast.parse(context.content)
        except SyntaxError:
            return results

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for .get() calls with default values
                if (isinstance(node.func, ast.Attribute) and
                    node.func.attr == 'get' and
                    len(node.args) >= 2):

                    # Get the object being called (e.g., 'config', 'embeddings_config')
                    if isinstance(node.func.value, ast.Name):
                        var_name = node.func.value.id
                    elif isinstance(node.func.value, ast.Attribute):
                        var_name = ast.unparse(node.func.value)
                    else:
                        continue

                    # Check if this looks like a config object
                    if self._is_config_variable(var_name):
                        # Get the key and default value
                        key_node = node.args[0]
                        default_node = node.args[1]

                        key = self._extract_string_value(key_node)
                        default_value = self._extract_node_value(default_node)

                        # Flag as error
                        results.append(ValidationResult(
                            rule_name=self.name,
                            severity="error",
                            message=f"Configuration fallback detected: {var_name}.get('{key}', {default_value})",
                            line_number=node.lineno,
                            column=node.col_offset,
                            suggestion=f"Use strict config: {var_name}['{key}'] and ensure value exists in config",
                            fix_suggestion=f"{var_name}['{key}']  # STRICT: No fallbacks",
                            category="configuration"
                        ))

        return results

    def _check_hardcoded_endpoints(self, context: CodeContext) -> List[ValidationResult]:
        """Check for hardcoded endpoints that bypass CM."""
        results = []

        # Patterns that indicate hardcoded endpoints
        dangerous_patterns = [
            (r'workers\.dev', 'Cloudflare Workers endpoint'),
            (r'https?://[^/]*\.workers\.dev', 'Cloudflare Workers URL'),
            (r'https?://\d+\.\d+\.\d+\.\d+:\d+', 'Hardcoded IP endpoint'),
            (r'localhost:\d+', 'Hardcoded localhost endpoint'),
            (r'127\.0\.0\.1:\d+', 'Hardcoded localhost endpoint'),
        ]

        lines = context.content.splitlines()
        for line_num, line in enumerate(lines, 1):
            for pattern, description in dangerous_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    # Skip if it's in a comment explaining the pattern
                    if '#' in line and line.index('#') < match.start():
                        continue

                    results.append(ValidationResult(
                        rule_name=self.name,
                        severity="error",
                        message=f"Hardcoded endpoint detected: {description}",
                        line_number=line_num,
                        column=match.start(),
                        suggestion="Move endpoint configuration to dev.pyproject.toml or pyproject.toml",
                        fix_suggestion="# FIXME: Move to configuration management",
                        category="configuration"
                    ))

        return results

    def _check_config_file_fallbacks(self, context: CodeContext) -> List[ValidationResult]:
        """Check TOML/YAML files for fallback patterns."""
        results = []

        # Check for production URLs in config files
        production_patterns = [
            r'workers\.dev',
            r'\.vercel\.app',
            r'\.netlify\.app',
            r'\.herokuapp\.com'
        ]

        lines = context.content.splitlines()
        for line_num, line in enumerate(lines, 1):
            for pattern in production_patterns:
                if re.search(pattern, line):
                    results.append(ValidationResult(
                        rule_name=self.name,
                        severity="warning",
                        message=f"Production URL in config file may need dev override",
                        line_number=line_num,
                        column=0,
                        suggestion="Ensure dev.pyproject.toml overrides production URLs",
                        category="configuration"
                    ))

        return results

    def _is_config_variable(self, var_name: str) -> bool:
        """Check if variable name suggests it's a config object."""
        config_indicators = [
            'config', 'settings', 'cfg', 'conf',
            'embedding_config', 'embeddings_config',
            'llm_config', 'kaia_config', 'tool_config',
            'vibelint_config', 'guardrails_config'
        ]

        var_lower = var_name.lower()
        return any(indicator in var_lower for indicator in config_indicators)

    def _extract_string_value(self, node: ast.AST) -> str:
        """Extract string value from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Str):  # Python < 3.8 compatibility
            return node.s
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else '<complex>'

    def _extract_node_value(self, node: ast.AST) -> str:
        """Extract a readable representation of the node value."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Str):
            return repr(node.s)
        elif isinstance(node, ast.Num):
            return str(node.n)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Dict)):
            return ast.unparse(node) if hasattr(ast, 'unparse') else '<collection>'
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else '<complex>'


class ConfigFallbackDetector:
    """Standalone utility for detecting configuration fallbacks."""

    def __init__(self):
        self.rule = StrictConfigRule()

    def scan_directory(self, directory: Path) -> Dict[str, List[ValidationResult]]:
        """Scan a directory for configuration fallbacks."""
        results = {}

        for file_path in directory.rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                context = CodeContext(file_path=file_path, content=content)
                file_results = self.rule.validate(context)

                if file_results:
                    results[str(file_path)] = file_results

            except Exception as e:
                print(f"Error scanning {file_path}: {e}")

        return results

    def generate_report(self, results: Dict[str, List[ValidationResult]]) -> str:
        """Generate a human-readable report."""
        if not results:
            return "✅ No configuration fallbacks detected!"

        report = ["🚨 CONFIGURATION FALLBACKS DETECTED", "=" * 50, ""]

        total_issues = sum(len(issues) for issues in results.values())
        report.append(f"Total files with issues: {len(results)}")
        report.append(f"Total fallback patterns: {total_issues}")
        report.append("")

        for file_path, issues in results.items():
            report.append(f"📁 {file_path}")
            report.append("-" * 50)

            for issue in issues:
                report.append(f"  ❌ Line {issue.line_number}: {issue.message}")
                report.append(f"     💡 {issue.suggestion}")
                if issue.fix_suggestion:
                    report.append(f"     🔧 Fix: {issue.fix_suggestion}")
                report.append("")

        report.append("=" * 50)
        report.append("🎯 RECOMMENDATION: Move all configuration to CM system")
        report.append("   1. Add required config to dev.pyproject.toml")
        report.append("   2. Replace .get() calls with strict [] access")
        report.append("   3. Let configuration errors fail loudly")

        return "\n".join(report)


# CLI interface for standalone usage
if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("Usage: python strict_config.py <directory>")
        sys.exit(1)

    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"Directory not found: {directory}")
        sys.exit(1)

    detector = ConfigFallbackDetector()
    results = detector.scan_directory(directory)
    report = detector.generate_report(results)
    print(report)

    # Exit with error code if issues found
    if results:
        sys.exit(1)