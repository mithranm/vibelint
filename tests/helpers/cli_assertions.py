"""
Pure CLI test assertion helpers.

Extracted from cli_utils.py to follow single responsibility principle.
"""
import re
from pathlib import Path

import pytest
from click.testing import Result


def clean_output(output: str) -> str:
    """Removes ANSI escape codes and strips leading/trailing whitespace from each line."""
    cleaned = re.sub(r"\x1b\[.*?m", "", output)  # Remove ANSI codes
    cleaned = re.sub(r"\r\n?", "\n", cleaned)  # Normalize line endings
    # Process line by line to strip, then rejoin. Filter removes empty lines.
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]  # Filter empty lines
    return "\n".join(lines)


def assert_output_contains(result: Result, substring: str, msg: str = ""):
    """Asserts substring is in cleaned output."""
    cleaned = clean_output(result.output)
    if substring not in cleaned:
        pytest.fail(f"{msg}\nExpected substring '{substring}' not found in:\n{cleaned}")


def assert_output_not_contains(result: Result, substring: str, msg: str = ""):
    """Asserts substring is NOT in cleaned output."""
    cleaned = clean_output(result.output)
    if substring in cleaned:
        pytest.fail(f"{msg}\nUnexpected substring '{substring}' found in:\n{cleaned}")


def assert_exit_code(result: Result, expected_code: int, msg: str = ""):
    """Asserts the CLI exit code matches expected."""
    if result.exit_code != expected_code:
        cleaned = clean_output(result.output)
        pytest.fail(
            f"{msg}\nExpected exit code {expected_code}, got {result.exit_code}\n"
            f"Output:\n{cleaned}"
        )


def get_fixture_path(fixture_name: str) -> Path:
    """Get path to a test fixture directory."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    return fixtures_dir / fixture_name