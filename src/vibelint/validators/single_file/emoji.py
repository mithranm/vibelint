"""Emoji usage validator using BaseValidator plugin system.

Detects emoji usage that can cause encoding issues, reduce readability,
and create compatibility problems across different terminals and systems.

vibelint/src/vibelint/validators/emoji.py
"""

import re
from pathlib import Path
from typing import Iterator

from ...validators.types import BaseValidator, Finding, Severity

__all__ = ["EmojiUsageValidator"]


class EmojiUsageValidator(BaseValidator):
    """Detects emoji usage that can cause encoding issues."""

    rule_id = "EMOJI-IN-STRING"
    name = "Emoji Usage Detector"
    description = "Detects emojis that can cause MCP and Windows shell issues"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for emoji usage in code."""
        # Comprehensive emoji regex pattern - matches ALL possible emojis
        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F"  # Emoticons
            r"\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
            r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
            r"\U0001F1E0-\U0001F1FF"  # Regional Indicator Symbols
            r"\U0001F700-\U0001F77F"  # Alchemical Symbols
            r"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            r"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            r"\U0001FA00-\U0001FA6F"  # Chess Symbols
            r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            r"\U00002600-\U000026FF"  # Miscellaneous Symbols
            r"\U00002700-\U000027BF"  # Dingbats
            r"\U0000FE00-\U0000FE0F"  # Variation Selectors
            r"\U0001F018-\U0001F270"  # Various Asian characters
            r"\U0000238C-\U00002454"  # Misc technical (fixed with leading zeros)
            r"\U000020D0-\U000020FF"  # Combining Diacritical Marks for Symbols (fixed)
            r"]+"
        )

        lines = content.splitlines()
        for line_num, line in enumerate(lines, 1):
            emoji_matches = emoji_pattern.findall(line)
            if emoji_matches:
                emojis_found = "".join(emoji_matches)

                # Check if emoji is in code vs strings/comments
                if self._is_emoji_in_code_context(line):
                    # More severe for emojis in actual code
                    yield self.create_finding(
                        message=f"Emoji in code: {emojis_found}",
                        file_path=file_path,
                        line=line_num,
                        suggestion="Replace emoji in code with ASCII alternatives immediately",
                    )
                else:
                    # Less severe for emojis in strings/comments
                    yield self.create_finding(
                        message=f"Emoji usage detected: {emojis_found}",
                        file_path=file_path,
                        line=line_num,
                        suggestion="Replace emojis with text descriptions to avoid encoding issues in MCP and Windows shells",
                    )

    def can_fix(self, finding: "Finding") -> bool:
        """Check if this finding can be automatically fixed."""
        return finding.rule_id == self.rule_id

    def apply_fix(self, content: str, finding: "Finding") -> str:
        """Automatically remove emojis from content."""
        lines = content.splitlines(True)  # Keep line endings
        if finding.line <= len(lines):
            line = lines[finding.line - 1]

            # Comprehensive emoji pattern for removal
            emoji_pattern = re.compile(
                r"[\U0001F600-\U0001F64F"  # Emoticons
                r"\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
                r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                r"\U0001F1E0-\U0001F1FF"  # Regional Indicator Symbols
                r"\U0001F700-\U0001F77F"  # Alchemical Symbols
                r"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                r"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                r"\U0001FA00-\U0001FA6F"  # Chess Symbols
                r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                r"\U00002600-\U000026FF"  # Miscellaneous Symbols
                r"\U00002700-\U000027BF"  # Dingbats
                r"\U0000FE00-\U0000FE0F"  # Variation Selectors
                r"\U0001F018-\U0001F270"  # Various Asian characters
                r"\U0000238C-\U00002454"  # Misc technical (fixed with leading zeros)
                r"\U000020D0-\U000020FF"  # Combining Diacritical Marks for Symbols (fixed)
                r"]+"
            )

            # Remove emojis from the line
            fixed_line = emoji_pattern.sub("", line)

            # Clean up any double spaces that might result, but preserve indentation and newlines
            # Extract leading whitespace (indentation)
            leading_whitespace = ""
            content_start = 0
            for char in fixed_line:
                if char in [" ", "\t"]:
                    leading_whitespace += char
                    content_start += 1
                else:
                    break

            # Split into content and line ending
            if fixed_line.endswith("\n"):
                content = fixed_line[content_start:-1]  # Remove leading whitespace and newline
                line_ending = "\n"
            elif fixed_line.endswith("\r\n"):
                content = fixed_line[content_start:-2]  # Remove leading whitespace and CRLF
                line_ending = "\r\n"
            else:
                content = fixed_line[content_start:]  # Remove leading whitespace
                line_ending = ""

            # Clean up only multiple consecutive spaces in content
            content = re.sub(r" {2,}", " ", content)  # Multiple spaces to single space
            content = content.rstrip()  # Remove trailing spaces only

            # Reconstruct the line with original indentation
            fixed_line = leading_whitespace + content + line_ending

            lines[finding.line - 1] = fixed_line

        return "".join(lines)

    def _is_emoji_in_code_context(self, line: str) -> bool:
        """Check if emoji appears to be in code rather than in strings or comments.

        This is a heuristic - emojis in strings/comments are less problematic
        than emojis used as identifiers or in code structure.
        """
        # Remove string literals and comments to see if emoji remains
        line_without_strings = re.sub(r'["\'].*?["\']', "", line)
        line_without_comments = re.sub(r"#.*$", "", line_without_strings)

        # If emoji still exists after removing strings/comments, it's likely in code
        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]"
        )
        return bool(emoji_pattern.search(line_without_comments))
