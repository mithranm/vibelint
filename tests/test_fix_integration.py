"""Integration tests for the --fix functionality with real LLM."""

import tempfile
from pathlib import Path
import pytest
import asyncio
from unittest.mock import Mock

from vibelint.cli import check_files_command
from vibelint.config import Config
from vibelint.plugin_system import Finding, Severity


class TestFixIntegration:
    """Integration tests for vibelint --fix functionality.

    These tests require a local LLM endpoint and are marked as integration tests.
    They should be run manually or in local CI, not in GitHub Actions.
    """

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,  # Skip by default - require explicit --integration flag
        reason="Integration test requiring local LLM - run with pytest --integration"
    )
    def test_fix_missing_docstrings_and_paths(self):
        """Test that --fix properly adds missing docstrings and path references."""
        # Create test file with missing docstrings and path references
        test_content = '''"""Test file for vibelint --fix functionality."""

def hello_world():
    """Say hello to the world."""
    print("Hello, World!")

def add_numbers(a, b):
    return a + b

class TestClass:
    """A simple test class."""

    def test_method(self):
        """Test method without path reference."""
        pass
'''

        expected_fixes = {
            "add_numbers": "missing docstring",
            "module_docstring": "missing path reference",
            "hello_world_docstring": "missing path reference",
            "TestClass_docstring": "missing path reference",
            "test_method_docstring": "missing path reference",
            "__all__": "missing exports"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            test_file = Path(f.name)

        try:
            # Test that issues are found before fix
            config = self._create_test_config()
            ctx = Mock()
            ctx.obj = {'config': config}

            # Run check with --fix
            result = check_files_command(
                ctx=ctx,
                paths=[str(test_file)],
                categories=['core'],
                rules=[],
                format="natural",
                max_issues=0,
                fix=True  # Enable fix mode
            )

            # Read the fixed file
            fixed_content = test_file.read_text()

            # Verify fixes were applied
            assert "__all__" in fixed_content, "Should add __all__ exports"
            assert "def add_numbers(a, b):" in fixed_content, "Should preserve function"

            # Count docstrings with path references
            path_refs = fixed_content.count(test_file.name)
            assert path_refs >= 4, f"Should add path references to docstrings, found {path_refs}"

            # Verify syntax is valid
            compile(fixed_content, str(test_file), 'exec')

            # Re-run vibelint to ensure all issues are fixed
            result_after = check_files_command(
                ctx=ctx,
                paths=[str(test_file)],
                categories=['core'],
                rules=[],
                format="natural",
                max_issues=0,
                fix=False  # Just check, don't fix
            )

            # Should have no remaining core issues
            # (This would need to be verified through the return value/output)

        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,  # Skip by default
        reason="Integration test requiring local LLM - run with pytest --integration"
    )
    def test_fix_preserves_functionality(self):
        """Test that fixes don't break code functionality."""
        test_content = '''def calculate(x, y):
    return x * 2 + y

class Calculator:
    def add(self, a, b):
        return a + b
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            test_file = Path(f.name)

        try:
            config = self._create_test_config()
            ctx = Mock()
            ctx.obj = {'config': config}

            # Apply fixes
            check_files_command(
                ctx=ctx,
                paths=[str(test_file)],
                categories=['core'],
                rules=[],
                format="natural",
                max_issues=0,
                fix=True
            )

            # Verify the fixed code still works
            fixed_content = test_file.read_text()

            # Execute the code to ensure it's valid
            namespace = {}
            exec(fixed_content, namespace)

            # Test functionality is preserved
            calculate = namespace['calculate']
            Calculator = namespace['Calculator']

            assert calculate(3, 4) == 10, "Function should still work correctly"

            calc = Calculator()
            assert calc.add(2, 3) == 5, "Method should still work correctly"

        finally:
            if test_file.exists():
                test_file.unlink()

    def _create_test_config(self) -> Config:
        """Create a test config with LLM settings."""
        # This would load from the project's pyproject.toml
        # which should have the LLM configuration
        config_dict = {
            "llm_analysis": {
                "api_base_url": "http://100.94.250.88:11434",
                "model": "/home/mithranmohanraj/models/gpt-oss-20b-mxfp4.gguf",
                "max_tokens": 2048,
                "temperature": 0.1,
                "remove_thinking_tokens": True,
                "thinking_format": "harmony"
            }
        }
        return Config(project_root=Path.cwd(), config_dict=config_dict)


# Helper to run integration tests locally
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--integration",
        "-s"  # Show output
    ])