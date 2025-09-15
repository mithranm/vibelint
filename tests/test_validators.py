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


def test_print_statement_suppression():
    """Test print statement suppression comments."""
    # Initialize with empty config to avoid auto-exclusion
    validator = PrintStatementValidator(config={})

    code = """
print("hello")  # Regular print - should be flagged
print("Starting server...")  # vibelint: stdout - intentional CLI output
print("Debug output")  # noqa: print - suppressed
print("More debug")  # noqa - general suppression
print("Type issue")  # type: ignore - also suppressed
print("Server ready at http://localhost:8000")  # Should be auto-detected as CLI
"""

    findings = list(validator.validate(Path("example.py"), code, config={}))

    # Debug: print findings to understand what's happening
    print(f"Found {len(findings)} findings")
    for f in findings:
        print(f"  Line {f.line}: {f.message}")

    # Should only flag the first and third print (debug output without suppression)
    # The server URL print should be auto-detected as legitimate CLI output
    assert len(findings) == 2

    # Check line numbers
    line_numbers = [f.line for f in findings]
    assert 2 in line_numbers  # "hello" print
    assert 4 not in line_numbers  # vibelint: stdout suppressed
    assert 5 not in line_numbers  # noqa: print suppressed
    assert 6 not in line_numbers  # noqa suppressed
    assert 7 not in line_numbers  # type: ignore suppressed
    assert 8 not in line_numbers  # Auto-detected as CLI (contains URL)


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
