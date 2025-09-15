"""
Direct test for print statement suppression functionality.
"""

from pathlib import Path

from vibelint.validators.print_statements import PrintStatementValidator


def test_print_suppression_comments():
    """Test that suppression comments work correctly."""
    # Create validator with explicit empty config to avoid defaults
    validator = PrintStatementValidator()
    # Override the config to ensure no exclusions
    validator.config = {"print_validation": {"exclude_globs": []}}

    # Test basic suppression with vibelint: stdout
    code1 = 'print("CLI output")  # vibelint: stdout'
    findings1 = list(validator.validate(Path("script.py"), code1))
    assert len(findings1) == 0, "vibelint: stdout comment should suppress warning"

    # Test noqa: print suppression
    code2 = 'print("Debug")  # noqa: print'
    findings2 = list(validator.validate(Path("script.py"), code2))
    assert len(findings2) == 0, "noqa: print should suppress warning"

    # Test general noqa
    code3 = 'print("Debug")  # noqa'
    findings3 = list(validator.validate(Path("script.py"), code3))
    assert len(findings3) == 0, "noqa should suppress warning"

    # Test unsuppressed print
    code4 = 'print("Debug output")'
    findings4 = list(validator.validate(Path("script.py"), code4))
    print(f"Unsuppressed test: found {len(findings4)} findings")
    assert len(findings4) == 1, "Print without suppression should be flagged"

    # Test auto-detection of CLI patterns (URL)
    code5 = 'print("Server at http://localhost:8000")'
    findings5 = list(validator.validate(Path("script.py"), code5))
    assert len(findings5) == 0, "URL in print should be auto-detected as CLI output"

    # Test each case independently since multiline has issues
    codes = [
        ('print("Debug 1")', True, "Regular print should flag"),
        ('print("Status update")  # vibelint: stdout', False, "vibelint: stdout should suppress"),
        ('print("Debug 2")  # NOQA: PRINT', False, "NOQA: PRINT should suppress"),
        ('print("Starting server on port 8080")', False, "port number should auto-detect as CLI"),
    ]

    for code, should_flag, description in codes:
        findings = list(validator.validate(Path("script.py"), code))
        if should_flag:
            assert len(findings) == 1, f"{description} - expected 1 finding, got {len(findings)}"
        else:
            assert len(findings) == 0, f"{description} - expected 0 findings, got {len(findings)}"


if __name__ == "__main__":
    test_print_suppression_comments()
    print("All tests passed!")
