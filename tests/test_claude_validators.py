"""
Tests for Claude Code-specific validators.
"""

from pathlib import Path
from vibelint.validators.builtin_validators import (
    TodoTrackingValidator,
    EmojiUsageValidator,
    FunctionParameterValidator
)


def test_todo_tracking_validator():
    """Test TodoTrackingValidator functionality."""
    validator = TodoTrackingValidator()

    code = '''
# TODO: This is a regular todo
print("hello")
# TODO: URGENT fix this critical bug immediately
# todo: implement this later
'''

    findings = list(validator.validate(Path("test.py"), code))
    assert len(findings) == 3

    # Check that urgent TODO is marked as warning
    urgent_finding = next(f for f in findings if "critical" in f.message)
    assert urgent_finding.severity.value == "WARN"


def test_emoji_usage_validator():
    """Test EmojiUsageValidator."""
    validator = EmojiUsageValidator()

    code = '''
# This has emojis that cause issues
print("Hello world!")
def process_data():
    """Process data with fancy output"""
    return True
'''

    findings = list(validator.validate(Path("test.py"), code))
    assert len(findings) == 3  # Three lines with emojis

    # Check that suggestions mention MCP compatibility
    for finding in findings:
        assert "MCP" in finding.suggestion or "encoding" in finding.suggestion


def test_function_parameter_validator():
    """Test FunctionParameterValidator functionality."""
    validator = FunctionParameterValidator()

    code = '''
def many_params(a, b, c, d, e, f, g, h, i):
    """Function with too many parameters."""
    return a + b + c + d + e + f + g + h + i

def needs_keywords(a, b, c, d):
    """Function that should use keyword-only arguments."""
    return a + b + c + d

def good_function(a, b, *, c=None, d=None):
    """Function with proper keyword-only parameters."""
    return a + b + (c or 0) + (d or 0)

def test_calls():
    # Too many positional arguments
    result = some_function(1, 2, 3, 4, 5)
    return result
'''

    findings = list(validator.validate(Path("test.py"), code))

    # Should find: many_params (too many total), needs_keywords (no keyword-only),
    # and some_function call (too many positional args)
    assert len(findings) >= 3

    # Check specific issues
    messages = [f.message for f in findings]
    assert any("too many parameters" in msg for msg in messages)
    assert any("no keyword-only arguments" in msg for msg in messages)
    assert any("positional arguments" in msg for msg in messages)


def test_claude_formatter():
    """Test that ClaudeFormatter is available."""
    from vibelint.formatters import BUILTIN_FORMATTERS
    assert "claude" in BUILTIN_FORMATTERS