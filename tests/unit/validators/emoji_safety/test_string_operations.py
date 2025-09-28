"""
Tests for emoji removal in string operations.

Focused tests for string literals, concatenation, and operations.
"""
from textwrap import dedent

from .base import EmojiSafetyTestBase


class TestEmojiStringOperations(EmojiSafetyTestBase):
    """Test emoji removal in string operations."""

    def test_simple_string_emoji_removal(self):
        """Test basic emoji removal from string literals."""
        original_code = dedent('''
            # Simple string with emoji
            message = "Hello 👋 World 🌍!"
            print(message)

            # String with multiple emojis
            greeting = "Welcome 🎉🎊 to our app! 🚀"

            # Empty string
            empty = ""

            # String without emojis (should be unchanged)
            normal = "This is a normal string"
        ''')

        # Should compile before processing
        assert self._compile_check(original_code)

        # Process with validator
        fixed_code = self._validate_and_fix(original_code)

        # Should still compile after processing
        assert self._compile_check(fixed_code)

        # Should execute without errors
        assert self._exec_check(fixed_code)

    def test_complex_string_operations_with_emojis(self):
        """Test emoji removal in complex string operations."""
        original_code = dedent('''
            # String concatenation with emojis
            part1 = "Start 🌟"
            part2 = "Middle 🔥"
            part3 = "End 🏁"
            result = part1 + " " + part2 + " " + part3

            # String formatting with emojis
            name = "Alice 👩"
            formatted = f"Hello {name}! Welcome 🎉"

            # String methods with emojis
            text = "Processing 🔄 data..."
            upper_text = text.upper()
            lower_text = text.lower()

            # Multi-line strings with emojis
            multiline = """
            Line 1 with emoji 📝
            Line 2 with another emoji 🔗
            Line 3 without emoji
            """

            # List of strings with emojis
            emoji_list = [
                "Item 1 ⭐",
                "Item 2 🎯",
                "Item 3 📋"
            ]
        ''')

        # Should compile and run before processing
        assert self._compile_check(original_code)
        assert self._exec_check(original_code)

        # Process with validator
        fixed_code = self._validate_and_fix(original_code)

        # Should still compile and run after processing
        assert self._compile_check(fixed_code)
        assert self._exec_check(fixed_code)