"""
Base utilities for emoji removal safety testing.

Extracted from monolithic test_emoji_removal_safety.py for reuse.
"""
import ast
import tempfile
from pathlib import Path

import pytest

from vibelint.validators.single_file.emoji import EmojiUsageValidator


class EmojiSafetyTestBase:
    """Base class for emoji removal safety tests."""

    def setup_method(self):
        """Set up test validator."""
        self.validator = EmojiUsageValidator()

    def _compile_check(self, code: str) -> bool:
        """Check if code compiles without syntax errors."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _exec_check(self, code: str) -> bool:
        """Check if code executes without runtime errors."""
        try:
            # Create a temporary file to execute the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name

            # Try to compile and execute
            with open(temp_path, 'r') as f:
                code_content = f.read()

            # Compile first
            compiled = compile(code_content, temp_path, 'exec')

            # Execute in isolated namespace
            namespace = {'__name__': '__main__'}
            exec(compiled, namespace)

            # Clean up
            Path(temp_path).unlink()
            return True

        except Exception:
            # Clean up on failure
            try:
                Path(temp_path).unlink()
            except:
                pass
            return False

    def _validate_and_fix(self, original_code: str) -> str:
        """Run validator and return fixed code."""
        findings = list(self.validator.validate(Path("test.py"), original_code))

        # Apply fixes
        fixed_code = original_code
        for finding in findings:
            if finding.suggestion:
                # Simple replacement for testing
                fixed_code = fixed_code.replace(finding.message.split("'")[1], "")

        return fixed_code