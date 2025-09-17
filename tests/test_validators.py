"""
Tests for validators.
"""

from pathlib import Path

from vibelint.validators.emoji import EmojiUsageValidator

# Import validators from their individual modules
from vibelint.validators.print_statements import PrintStatementValidator
from vibelint.validators.typing_quality import TypingQualityValidator


def test_print_statement_validator():
    """Test PrintStatementValidator functionality."""
    validator = PrintStatementValidator()

    code = """
print("hello")
print("debug info")
"""

    findings = list(validator.validate(Path("test.py"), code))
    assert len(findings) == 2

    # Check that print statements are detected
    for finding in findings:
        assert "Print statement found" in finding.message
        assert "logging" in finding.message


def test_emoji_usage_validator():
    """Test EmojiUsageValidator."""
    validator = EmojiUsageValidator()

    code = '''# This has emojis 🚀 that cause issues
print("Hello world! ⭐")
def process_data():
    """Process data with fancy output 🎉"""
    return True
'''

    findings = list(validator.validate(Path("test.py"), code))
    assert len(findings) == 2  # Two lines with emojis

    # Check that suggestions mention MCP compatibility
    for finding in findings:
        assert (
            finding.suggestion is None
            or "MCP" in finding.suggestion
            or "encoding" in finding.suggestion
        )


def test_typing_quality_validator():
    """Test TypingQualityValidator functionality."""
    validator = TypingQualityValidator()

    code = '''
def untyped_function(a, b, c):
    """Function without type annotations."""
    return a + b + c
'''

    findings = list(validator.validate(Path("test.py"), code))

    # Should find missing type annotations for function and parameters
    assert len(findings) >= 3

    # Check that type annotation issues are detected
    messages = [f.message for f in findings]
    assert any("missing type annotations" in msg for msg in messages)


def test_llm_formatter():
    """Test that LLM formatter is available."""
    from vibelint.formatters import BUILTIN_FORMATTERS

    assert "llm" in BUILTIN_FORMATTERS
