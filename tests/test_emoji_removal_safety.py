"""
Comprehensive tests for emoji removal safety.

Tests to ensure emoji removal doesn't break Python syntax or functionality.
"""

import ast
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from vibelint.validators.single_file.emoji import EmojiUsageValidator


class TestEmojiRemovalSafety:
    """Test that emoji removal doesn't break Python syntax."""

    def setup_method(self):
        """Set up test validator."""
        self.validator = EmojiUsageValidator()

    def _compile_check(self, code: str) -> bool:
        """Check if code compiles without syntax errors."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _exec_check(self, code: str) -> bool:
        """Check if code executes without runtime errors."""
        try:
            # Create a temporary file to execute the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name

            # Try to compile and execute
            with open(temp_path, 'r') as f:
                code_content = f.read()

            # Compile first
            compiled = compile(code_content, temp_path, 'exec')

            # Execute in isolated namespace
            namespace = {'__name__': '__main__'}
            exec(compiled, namespace)

            Path(temp_path).unlink()  # Clean up
            return True
        except Exception:
            try:
                Path(temp_path).unlink()  # Clean up on error
            except:
                pass
            return False

    def test_simple_string_emoji_removal(self):
        """Test emoji removal from simple strings."""
        code = '''
def greet():
    """Greet with emoji 🚀"""
    print("Hello! 👋")
    return "Done ✅"
'''

        # Original should compile
        assert self._compile_check(code)

        # Find emojis and apply fixes
        findings = list(self.validator.validate(Path("test.py"), code))
        assert len(findings) > 0  # Should find emojis

        # Apply all fixes
        fixed_code = code
        for finding in findings:
            fixed_code = self.validator.apply_fix(fixed_code, finding)

        # Fixed code should still compile and execute
        assert self._compile_check(fixed_code)
        assert self._exec_check(fixed_code)

        # Should have no emojis left
        remaining_findings = list(self.validator.validate(Path("test.py"), fixed_code))
        assert len(remaining_findings) == 0

    def test_complex_string_operations_with_emojis(self):
        """Test emoji removal from complex string operations."""
        code = '''
def process_messages():
    messages = [
        "Status: ✅ Complete",
        "Error: ❌ Failed",
        "Info: ℹ️ Processing",
        f"Result: {'🎉' if True else '😞'}"
    ]

    combined = " | ".join(messages)
    formatted = f"Summary: {combined} 📊"

    return {
        "status": "🟢 Active",
        "data": {"emoji_key": "🔑", "value": 42}
    }

result = process_messages()
'''

        # Original should work
        assert self._compile_check(code)
        assert self._exec_check(code)

        # Apply emoji fixes
        findings = list(self.validator.validate(Path("test.py"), code))
        fixed_code = code
        for finding in findings:
            fixed_code = self.validator.apply_fix(fixed_code, finding)

        # Fixed code should still work
        assert self._compile_check(fixed_code)
        assert self._exec_check(fixed_code)

    def test_emoji_in_function_names_and_variables(self):
        """Test handling of emojis in places where they'd break syntax."""
        # This is malformed code that should be detected but not break the fixer
        code = '''
def process_data🚀():  # Emoji in function name (should be detected)
    variable_name = "normal string"
    emoji_var🎯 = 42  # Emoji in variable name
    return variable_name + str(emoji_var🎯)
'''

        # This code is actually invalid Python, but our fixer should handle it gracefully
        findings = list(self.validator.validate(Path("test.py"), code))

        # Apply fixes
        fixed_code = code
        for finding in findings:
            fixed_code = self.validator.apply_fix(fixed_code, finding)

        # The fixed code should at least not crash the parser worse
        # (Though it may still be invalid due to remaining syntax issues)
        try:
            ast.parse(fixed_code)
        except SyntaxError:
            # This is expected for malformed code, but we shouldn't crash
            pass

    def test_emoji_removal_preserves_indentation(self):
        """Test that emoji removal preserves Python indentation."""
        code = '''
class DataProcessor:
    """Process data with emojis 🔄"""

    def __init__(self):
        self.status = "🟡 Ready"

    def process(self, data):
        """Process the data 📝"""
        if data:
            print("Processing... ⚙️")
            for item in data:
                if item:
                    print(f"  Item: {item} ✅")
                else:
                    print(f"  Skipped ⏭️")
        return "Done 🎉"

processor = DataProcessor()
result = processor.process(["a", "b", None, "c"])
'''

        # Original should work
        assert self._compile_check(code)
        assert self._exec_check(code)

        # Apply emoji fixes
        findings = list(self.validator.validate(Path("test.py"), code))
        fixed_code = code
        for finding in findings:
            fixed_code = self.validator.apply_fix(fixed_code, finding)

        # Fixed code should preserve indentation and work
        assert self._compile_check(fixed_code)
        assert self._exec_check(fixed_code)

        # Check that indentation levels are preserved
        original_lines = code.split('\n')
        fixed_lines = fixed_code.split('\n')

        assert len(original_lines) == len(fixed_lines)

        for orig, fixed in zip(original_lines, fixed_lines):
            # Indentation (leading whitespace) should be the same
            orig_indent = len(orig) - len(orig.lstrip())
            fixed_indent = len(fixed) - len(fixed.lstrip())
            assert orig_indent == fixed_indent

    def test_emoji_in_docstrings_and_comments(self):
        """Test emoji removal from docstrings and comments."""
        code = '''
def calculate_score(data):
    """
    Calculate score for data 📊

    Args:
        data: Input data 📥

    Returns:
        Score value 🔢
    """
    # Process the data 🔄
    score = 0
    for item in data:  # Iterate through items 🔍
        score += len(str(item))  # Add length 📏

    return score  # Return final score ✅

# Main execution 🚀
if __name__ == "__main__":
    result = calculate_score([1, 2, 3])  # Test data 🧪
    print(f"Final score: {result}")  # Output result 📤
'''

        # Original should work
        assert self._compile_check(code)
        assert self._exec_check(code)

        # Apply emoji fixes
        findings = list(self.validator.validate(Path("test.py"), code))
        fixed_code = code
        for finding in findings:
            fixed_code = self.validator.apply_fix(fixed_code, finding)

        # Fixed code should work
        assert self._compile_check(fixed_code)
        assert self._exec_check(fixed_code)

    def test_emoji_removal_edge_cases(self):
        """Test edge cases for emoji removal."""

        # Empty strings with emojis
        code1 = 'empty = "🚀"'
        findings = list(self.validator.validate(Path("test.py"), code1))
        fixed = code1
        for finding in findings:
            fixed = self.validator.apply_fix(fixed, finding)
        assert self._compile_check(fixed)

        # Multi-line strings with emojis
        code2 = '''
text = """
Multi-line string with emoji 🚀
And another emoji ✅
"""
'''
        findings = list(self.validator.validate(Path("test.py"), code2))
        fixed = code2
        for finding in findings:
            fixed = self.validator.apply_fix(fixed, finding)
        assert self._compile_check(fixed)

        # Raw strings with emojis
        code3 = r'raw_string = r"Raw string with emoji 🚀"'
        findings = list(self.validator.validate(Path("test.py"), code3))
        fixed = code3
        for finding in findings:
            fixed = self.validator.apply_fix(fixed, finding)
        assert self._compile_check(fixed)

    def test_no_double_space_cleanup(self):
        """Test that double spaces are properly cleaned up after emoji removal."""
        code = '''
message = "Before 🚀 After"
another = "Start ✅ Middle 🎉 End"
'''

        findings = list(self.validator.validate(Path("test.py"), code))
        fixed_code = code
        for finding in findings:
            fixed_code = self.validator.apply_fix(fixed_code, finding)

        # Should not have multiple consecutive spaces
        assert "  " not in fixed_code.replace('\n', ' ')
        assert self._compile_check(fixed_code)

    def test_syntax_preservation_comprehensive(self):
        """Comprehensive test of syntax preservation."""

        # Complex realistic code with emojis
        code = dedent('''
            import json
            from typing import List, Dict, Any

            class APIClient:
                """API client with emoji status indicators 🌐"""

                def __init__(self, base_url: str):
                    self.base_url = base_url
                    self.session_id = None
                    print("Initializing client... 🔧")

                async def authenticate(self, credentials: Dict[str, str]) -> bool:
                    """Authenticate with the API 🔐"""
                    try:
                        # Simulate authentication
                        if credentials.get("token"):
                            self.session_id = "session_123"
                            print("Authentication successful! ✅")
                            return True
                        else:
                            print("Authentication failed! ❌")
                            return False
                    except Exception as e:
                        print(f"Error during auth: {e} 🚨")
                        return False

                def fetch_data(self, endpoint: str) -> List[Dict[str, Any]]:
                    """Fetch data from endpoint 📡"""
                    if not self.session_id:
                        raise ValueError("Not authenticated 🔒")

                    # Mock data
                    return [
                        {"id": 1, "status": "active 🟢", "name": "Item 1"},
                        {"id": 2, "status": "pending 🟡", "name": "Item 2"},
                        {"id": 3, "status": "error 🔴", "name": "Item 3"}
                    ]

                def process_batch(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
                    """Process a batch of items 📦"""
                    counts = {"success": 0, "error": 0}

                    for item in items:
                        try:
                            # Process item
                            processed = self._process_single_item(item)
                            if processed:
                                counts["success"] += 1
                                print(f"Processed {item['name']} ✅")
                            else:
                                counts["error"] += 1
                                print(f"Failed {item['name']} ❌")
                        except Exception:
                            counts["error"] += 1
                            print(f"Exception for {item['name']} 💥")

                    print(f"Batch complete! Success: {counts['success']}, Errors: {counts['error']} 📊")
                    return counts

                def _process_single_item(self, item: Dict[str, Any]) -> bool:
                    """Process a single item 🔧"""
                    return "error" not in item.get("status", "")

            # Usage example
            if __name__ == "__main__":
                client = APIClient("https://api.example.com")
                print("Starting API client demo 🚀")
        ''')

        # Original should compile and be valid
        assert self._compile_check(code)

        # Apply all emoji fixes
        findings = list(self.validator.validate(Path("test.py"), code))
        assert len(findings) > 10  # Should find many emojis

        fixed_code = code
        for finding in findings:
            fixed_code = self.validator.apply_fix(fixed_code, finding)

        # Fixed code should still be perfectly valid Python
        assert self._compile_check(fixed_code)

        # Should have no emojis remaining
        remaining_findings = list(self.validator.validate(Path("test.py"), fixed_code))
        assert len(remaining_findings) == 0

        # Verify the code structure is intact
        original_ast = ast.parse(code)
        fixed_ast = ast.parse(fixed_code)

        # Should have the same number of top-level nodes
        assert len(original_ast.body) == len(fixed_ast.body)


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])