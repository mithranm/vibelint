"""
Tests for emoji removal edge cases and error conditions.

Focused tests for unusual scenarios and boundary conditions.
"""
from textwrap import dedent

from .base import EmojiSafetyTestBase


class TestEmojiEdgeCases(EmojiSafetyTestBase):
    """Test edge cases in emoji removal."""

    def test_emoji_removal_edge_cases(self):
        """Test various edge cases for emoji removal."""
        original_code = dedent('''
            # Edge case: emoji at start and end of strings
            start_emoji = "🔥 This starts with emoji"
            end_emoji = "This ends with emoji 🔥"
            both_emoji = "🔥 Both sides 🔥"

            # Edge case: multiple consecutive emojis
            consecutive = "Multiple 🔥🔥🔥 emojis"
            mixed = "Mixed 🔥 content 🌟 with 🎯 emojis"

            # Edge case: emoji in different contexts
            in_f_string = f"Dynamic content 🚀 here"
            in_raw_string = r"Raw string with emoji 📝"
            in_bytes = b"No emojis in bytes"  # Should be unchanged

            # Edge case: Unicode variations
            unicode_emoji = "Unicode 🌈 and \\u2764\\ufe0f variations"

            # Edge case: Empty strings and whitespace
            empty = ""
            whitespace = "   "
            emoji_only = "🔥"
            whitespace_emoji = "  🔥  "
        ''')

        assert self._compile_check(original_code)
        fixed_code = self._validate_and_fix(original_code)
        assert self._compile_check(fixed_code)
        assert self._exec_check(fixed_code)

    def test_no_double_space_cleanup(self):
        """Test that emoji removal doesn't create double spaces."""
        original_code = dedent('''
            # These should not result in double spaces after emoji removal
            sentence1 = "Word 🔥 word"
            sentence2 = "Start 🌟 middle 🎯 end"
            sentence3 = "Multiple 🔥🔥 consecutive"

            # Test with punctuation
            punctuated = "Hello 👋, world 🌍!"
            question = "How are you 😊?"
            exclamation = "Great job 🎉!"

            print(sentence1)
            print(sentence2)
            print(sentence3)
            print(punctuated)
            print(question)
            print(exclamation)
        ''')

        assert self._compile_check(original_code)
        fixed_code = self._validate_and_fix(original_code)
        assert self._compile_check(fixed_code)
        assert self._exec_check(fixed_code)

        # Ensure no double spaces are created
        assert "  " not in fixed_code  # No double spaces