"""Tests for the --fix functionality to ensure it works correctly."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from vibelint.fix import FixEngine, can_fix_finding, apply_fixes
from vibelint.plugin_system import Finding, Severity
from vibelint.config import Config


class TestFixFunctionality:
    """Test the fix functionality."""

    def test_can_fix_finding_with_fixable_rules(self):
        """Test that can_fix_finding correctly identifies fixable rules."""
        # Test fixable rules
        finding1 = Finding(
            rule_id="DOCSTRING-MISSING",
            message="Missing docstring",
            file_path=Path("test.py"),
            line=1,
            severity=Severity.WARN,
        )
        assert can_fix_finding(finding1) is True

        finding2 = Finding(
            rule_id="DOCSTRING-PATH-REFERENCE",
            message="Missing path reference",
            file_path=Path("test.py"),
            line=1,
            severity=Severity.WARN,
        )
        assert can_fix_finding(finding2) is True

        finding3 = Finding(
            rule_id="EXPORTS-MISSING-ALL",
            message="Missing __all__",
            file_path=Path("test.py"),
            line=1,
            severity=Severity.WARN,
        )
        assert can_fix_finding(finding3) is True

    def test_can_fix_finding_with_non_fixable_rules(self):
        """Test that can_fix_finding correctly identifies non-fixable rules."""
        finding = Finding(
            rule_id="SOME-OTHER-RULE",
            message="Some other issue",
            file_path=Path("test.py"),
            line=1,
            severity=Severity.WARN,
        )
        assert can_fix_finding(finding) is False

    def test_fix_engine_can_fix_finding(self):
        """Test FixEngine.can_fix_finding method."""
        config = MagicMock()
        engine = FixEngine(config)

        finding = Finding(
            rule_id="DOCSTRING-MISSING",
            message="Missing docstring",
            file_path=Path("test.py"),
            line=1,
            severity=Severity.WARN,
        )
        assert engine.can_fix_finding(finding) is True

        finding2 = Finding(
            rule_id="NON-FIXABLE-RULE",
            message="Some other issue",
            file_path=Path("test.py"),
            line=1,
            severity=Severity.WARN,
        )
        assert engine.can_fix_finding(finding2) is False

    def test_build_fix_prompt(self):
        """Test that _build_fix_prompt correctly uses rule_id."""
        config = MagicMock()
        engine = FixEngine(config)

        findings = [
            Finding(
                rule_id="DOCSTRING-MISSING",
                message="Missing docstring",
                file_path=Path("test.py"),
                line=5,
                severity=Severity.WARN,
            ),
            Finding(
                rule_id="EXPORTS-MISSING-ALL",
                message="Missing __all__",
                file_path=Path("test.py"),
                line=1,
                severity=Severity.WARN,
            ),
        ]

        prompt = engine._build_fix_prompt(Path("test.py"), "# content", findings)

        # Check that rule_id is used in the prompt
        assert "DOCSTRING-MISSING" in prompt
        assert "EXPORTS-MISSING-ALL" in prompt
        assert "Line 5: DOCSTRING-MISSING" in prompt
        assert "Line 1: EXPORTS-MISSING-ALL" in prompt

    def test_extract_fixed_code(self):
        """Test _extract_fixed_code method."""
        config = MagicMock()
        engine = FixEngine(config)

        # Test with markdown code blocks
        response_with_blocks = '''Here's the fixed code:

```python
def hello():
    """Say hello."""
    print("Hello!")
```

That should fix the issue.'''

        result = engine._extract_fixed_code(response_with_blocks)
        expected = '''def hello():
    """Say hello."""
    print("Hello!")'''
        assert result == expected

        # Test without code blocks
        response_without_blocks = '''def hello():
    """Say hello."""
    print("Hello!")'''

        result = engine._extract_fixed_code(response_without_blocks)
        assert result == response_without_blocks

    @pytest.mark.asyncio
    async def test_apply_fixes_integration(self):
        """Test apply_fixes function integration."""
        config = MagicMock()

        # Mock findings by file
        findings = [
            Finding(
                rule_id="DOCSTRING-MISSING",
                message="Missing docstring",
                file_path=Path("test.py"),
                line=5,
                severity=Severity.WARN,
            )
        ]

        file_findings = {Path("test.py"): findings}

        # Mock the FixEngine to avoid actual LLM calls
        with pytest.MonkeyPatch().context() as m:
            mock_fix_file = AsyncMock(return_value=False)
            m.setattr("vibelint.fix.FixEngine.fix_file", mock_fix_file)

            results = await apply_fixes(config, file_findings)

            assert Path("test.py") in results
            assert isinstance(results[Path("test.py")], bool)


if __name__ == "__main__":
    pytest.main([__file__])
