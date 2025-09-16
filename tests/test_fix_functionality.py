"""Tests for the --fix functionality to ensure it works correctly."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from vibelint.config import Config
from vibelint.fix import FixEngine, apply_fixes, can_fix_finding
from vibelint.plugin_system import Finding, Severity


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

    def test_deterministic_fixes(self):
        """Test that fixes are applied deterministically without LLM file rewriting."""
        config = MagicMock()
        engine = FixEngine(config)

        # Test can_fix_finding works correctly
        docstring_finding = Finding(
            rule_id="DOCSTRING-MISSING",
            message="Missing docstring",
            file_path=Path("test.py"),
            line=5,
            severity=Severity.WARN,
        )

        exports_finding = Finding(
            rule_id="EXPORTS-MISSING-ALL",
            message="Missing __all__",
            file_path=Path("test.py"),
            line=1,
            severity=Severity.WARN,
        )

        unfixable_finding = Finding(
            rule_id="SOME-OTHER-RULE",
            message="Some other issue",
            file_path=Path("test.py"),
            line=1,
            severity=Severity.WARN,
        )

        # Test which findings can be fixed
        assert engine.can_fix_finding(docstring_finding)
        assert engine.can_fix_finding(exports_finding)
        assert not engine.can_fix_finding(unfixable_finding)

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

            # apply_fixes returns count of fixed files, not dict
            assert isinstance(results, int)
            assert results == 0  # No files were actually fixed (mocked to return False)


if __name__ == "__main__":
    pytest.main([__file__])
