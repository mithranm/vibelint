"""
Validator for emoji usage in Python code.

Detects and reports emoji usage which can cause encoding issues,
reduce readability, and create compatibility problems across different
terminals and systems.

vibelint/validators/emoji.py
"""

import re
from pathlib import Path

from ..error_codes import VBL601, VBL602, VBL603

__all__ = [
    "EmojiValidationResult",
    "validate_emoji_usage",
    "detect_emoji_in_text",
]

ValidationIssue = tuple[str, str]


class EmojiValidationResult:
    """
    Result of an emoji validation.

    vibelint/validators/emoji.py
    """

    def __init__(self) -> None:
        """Initialize an empty emoji validation result."""
        self.issues: list[ValidationIssue] = []
        self.errors: list[ValidationIssue] = []
        self.warnings: list[ValidationIssue] = []
        self.emoji_count: int = 0
        self.emoji_locations: list[tuple[int, str, str]] = []  # (line_num, line_text, emoji)

    def add_emoji_issue(self, code: str, message: str, line_num: int, line_text: str, emoji: str) -> None:
        """Add an emoji-related issue to the result."""
        self.issues.append((code, message))

        # All emoji issues are treated as errors for now
        if code == "VBL603":  # Emoji in code
            self.errors.append((code, message))
        else:  # VBL601, VBL602 - treat as warnings
            self.warnings.append((code, message))

        self.emoji_locations.append((line_num, line_text.strip(), emoji))
        self.emoji_count += 1


def detect_emoji_in_text(text: str) -> list[tuple[str, int, int]]:
    """
    Detect emoji characters in text.

    Returns:
        List of tuples containing (emoji, start_pos, end_pos)
    """
    # Comprehensive emoji regex pattern covering most Unicode emoji ranges
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F]|'  # Emoticons
        r'[\U0001F300-\U0001F5FF]|'  # Misc Symbols and Pictographs
        r'[\U0001F680-\U0001F6FF]|'  # Transport and Map Symbols
        r'[\U0001F1E0-\U0001F1FF]|'  # Regional Indicator Symbols (flags)
        r'[\U00002600-\U000026FF]|'  # Miscellaneous Symbols
        r'[\U00002700-\U000027BF]|'  # Dingbats
        r'[\U0001F900-\U0001F9FF]|'  # Supplemental Symbols and Pictographs
        r'[\U0001FA70-\U0001FAFF]|'  # Symbols and Pictographs Extended-A
        r'[\U00002B50]|'             # Star
        r'[\U000023CF]|'             # Eject symbol
        r'[\U000023E9-\U000023F3]|'  # Play/pause symbols
        r'[\U000025AA-\U000025AB]|'  # Small squares
        r'[\U000025B6]|'             # Play button
        r'[\U000025C0]|'             # Reverse button
        r'[\U000025FB-\U000025FE]|'  # Squares
        r'[\U00002B05-\U00002B07]|'  # Arrows
        r'[\U00002B1B-\U00002B1C]|'  # Squares
        r'[\U00002B55]|'             # Circle
        r'[\U00003030]|'             # Wavy dash
        r'[\U0000303D]|'             # Part alternation mark
        r'[\U00003297]|'             # Congratulations symbol
        r'[\U00003299]'              # Secret symbol
    )

    results = []
    for match in emoji_pattern.finditer(text):
        results.append((match.group(), match.start(), match.end()))

    return results


def validate_emoji_usage(file_path: Path, content: str | None = None) -> EmojiValidationResult:
    """
    Validate emoji usage in a Python file.

    Args:
        file_path: Path to the Python file to validate
        content: Optional file content (if already loaded)

    Returns:
        EmojiValidationResult containing any issues found
    """
    result = EmojiValidationResult()

    try:
        if content is None:
            content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Handle encoding issues
        try:
            content = file_path.read_text(encoding='latin1')
        except Exception:
            result.add_emoji_issue(
                VBL601,
                f"Could not read file {file_path} to check for emojis",
                0, "", ""
            )
            return result
    except Exception:
        result.add_emoji_issue(
            VBL601,
            f"Could not access file {file_path}",
            0, "", ""
        )
        return result

    lines = content.splitlines()

    for line_num, line in enumerate(lines, 1):
        emojis = detect_emoji_in_text(line)

        for emoji, start_pos, end_pos in emojis:
            # Determine context (string, comment, or code)
            context = _determine_emoji_context(line, start_pos)

            if context == "string":
                result.add_emoji_issue(
                    VBL602,
                    f"Emoji '{emoji}' found in string literal at line {line_num}. "
                    f"Consider using text description instead for better compatibility.",
                    line_num, line, emoji
                )
            elif context == "comment":
                result.add_emoji_issue(
                    VBL602,
                    f"Emoji '{emoji}' found in comment at line {line_num}. "
                    f"Consider using text description instead for better readability.",
                    line_num, line, emoji
                )
            else:
                result.add_emoji_issue(
                    VBL603,
                    f"Emoji '{emoji}' found in code at line {line_num}. "
                    f"This can cause encoding and compatibility issues.",
                    line_num, line, emoji
                )

    return result


def _determine_emoji_context(line: str, emoji_pos: int) -> str:
    """
    Determine if emoji is in a string, comment, or code.

    Args:
        line: The line of code
        emoji_pos: Position of the emoji in the line

    Returns:
        "string", "comment", or "code"
    """
    # Simple heuristic - could be improved with AST parsing

    # Check if it's in a comment (after #)
    comment_pos = line.find('#')
    if comment_pos != -1 and comment_pos < emoji_pos:
        return "comment"

    # Check if it's in a string literal
    # This is a simplified check - a full parser would be more accurate
    before_emoji = line[:emoji_pos]

    # Count quotes to determine if we're inside a string
    single_quotes = before_emoji.count("'") - before_emoji.count("\\'")
    double_quotes = before_emoji.count('"') - before_emoji.count('\\"')

    if single_quotes % 2 == 1 or double_quotes % 2 == 1:
        return "string"

    # Check for f-strings, triple quotes, etc.
    if (
        'f"' in before_emoji
        or "f'" in before_emoji
        or '"""' in before_emoji
        or "'''" in before_emoji
    ):
        return "string"

    return "code"
