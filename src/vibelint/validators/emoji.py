"""
Emoji usage validator using BaseValidator plugin system.

Detects emoji usage that can cause encoding issues, reduce readability,
and create compatibility problems across different terminals and systems.

vibelint/validators/emoji.py
"""

import re
from pathlib import Path
from typing import Iterator

from ..plugin_system import BaseValidator, Finding, Severity

__all__ = ["EmojiUsageValidator"]


class EmojiUsageValidator(BaseValidator):
    """Detects emoji usage that can cause encoding issues."""

    rule_id = "EMOJI-IN-STRING"
    name = "Emoji Usage Detector"
    description = "Detects emojis that can cause MCP and Windows shell issues"
    default_severity = Severity.WARN


    def validate(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Check for emoji usage in code."""
        # Emoji regex pattern - matches most common emojis
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]')

        lines = content.splitlines()
        for line_num, line in enumerate(lines, 1):
            emoji_matches = emoji_pattern.findall(line)
            if emoji_matches:
                emojis_found = ''.join(emoji_matches)

                # Check if emoji is in code vs strings/comments
                if self._is_emoji_in_code_context(line):
                    # More severe for emojis in actual code
                    yield self.create_finding(
                        message=f"Emoji in code: {emojis_found}",
                        file_path=file_path,
                        line=line_num,
                        suggestion="Replace emoji in code with ASCII alternatives immediately",
                        severity=Severity.BLOCK
                    )
                else:
                    # Less severe for emojis in strings/comments
                    yield self.create_finding(
                        message=f"Emoji usage detected: {emojis_found}",
                        file_path=file_path,
                        line=line_num,
                        suggestion="Replace emojis with text descriptions to avoid encoding issues in MCP and Windows shells"
                    )

    def _is_emoji_in_code_context(self, line: str) -> bool:
        """
        Check if emoji appears to be in code rather than in strings or comments.

        This is a heuristic - emojis in strings/comments are less problematic
        than emojis used as identifiers or in code structure.
        """
        # Remove string literals and comments to see if emoji remains
        line_without_strings = re.sub(r'["\'].*?["\']', '', line)
        line_without_comments = re.sub(r'#.*$', '', line_without_strings)

        # If emoji still exists after removing strings/comments, it's likely in code
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]')
        return bool(emoji_pattern.search(line_without_comments))