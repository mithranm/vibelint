"""
Tests for syntax preservation during emoji removal.

Focused tests for indentation, structure, and syntax integrity.
"""
from textwrap import dedent

from .base import EmojiSafetyTestBase


class TestEmojiSyntaxPreservation(EmojiSafetyTestBase):
    """Test that emoji removal preserves Python syntax."""

    def test_emoji_removal_preserves_indentation(self):
        """Test that emoji removal doesn't break indentation."""
        original_code = dedent('''
            def process_data():
                """Process data with emojis 📊"""
                if True:
                    print("Processing 🔄")
                    for i in range(3):
                        if i == 0:
                            result = "First 🥇"
                        elif i == 1:
                            result = "Second 🥈"
                        else:
                            result = "Third 🥉"
                        print(f"Result: {result}")

                try:
                    data = {"status": "success ✅", "error": None}
                    return data
                except Exception as e:
                    print(f"Error occurred 🚨: {e}")
                    return {"status": "failed ❌", "error": str(e)}
        ''')

        assert self._compile_check(original_code)
        fixed_code = self._validate_and_fix(original_code)
        assert self._compile_check(fixed_code)
        assert self._exec_check(fixed_code)

    def test_syntax_preservation_comprehensive(self):
        """Comprehensive test for syntax preservation."""
        original_code = dedent('''
            # Class definition with emojis
            class DataProcessor:
                """A class for processing data 🔧"""

                def __init__(self, name="Processor 🤖"):
                    self.name = name
                    self.status = "ready 🟢"

                def process(self, data):
                    """Process the data 📋"""
                    print(f"Processing with {self.name} 🔄")

                    # Dictionary with emoji values
                    config = {
                        "mode": "fast ⚡",
                        "output": "file 📁",
                        "format": "json 📋"
                    }

                    # List comprehension with emojis
                    results = [f"item_{i} ✨" for i in range(3)]

                    # Lambda with emoji
                    formatter = lambda x: f"formatted_{x} 🎨"

                    return {
                        "config": config,
                        "results": results,
                        "formatter": formatter("test")
                    }

            # Create instance and test
            processor = DataProcessor()
            result = processor.process(["data"])
            print("Done ✅")
        ''')

        assert self._compile_check(original_code)
        assert self._exec_check(original_code)

        fixed_code = self._validate_and_fix(original_code)
        assert self._compile_check(fixed_code)
        assert self._exec_check(fixed_code)