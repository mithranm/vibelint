#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the docstring validator.

tests/test_docstring.py
"""

import unittest
import re
from vibelint.validators.docstring import (
    validate_module_docstring,
    fix_module_docstring,
)


class DocstringValidatorTests(unittest.TestCase):
    """Test cases for the docstring validator."""

    def test_valid_docstring(self):
        """Test validation of a valid docstring."""
        content = '"""Module example with a valid docstring.\n\npath/to/module.py\n"""\n\ndef hello():\n    print("Hello")'
        result = validate_module_docstring(content, "path/to/module.py", r"^[A-Z].+\.$")
        self.assertFalse(result.has_issues())

    def test_missing_docstring(self):
        """Test validation of a missing docstring."""
        content = 'def hello():\n    print("Hello")'
        result = validate_module_docstring(content, "path/to/module.py", r"^[A-Z].+\.$")
        self.assertTrue(result.has_issues())
        self.assertTrue(any("docstring missing" in e.lower() for e in result.errors))

    def test_docstring_without_capitalized_first_line(self):
        """Test validation of a docstring without a capitalized first line."""
        content = '"""module example without capitalization.\n\npath/to/module.py\n"""\n\ndef hello():\n    print("Hello")'
        result = validate_module_docstring(content, "path/to/module.py", r"^[A-Z].+\.$")
        self.assertTrue(result.has_issues())
        self.assertTrue(any("first line" in e.lower() for e in result.errors))

    def test_docstring_without_period(self):
        """Test validation of a docstring without a period at the end of the first line."""
        content = '"""Module example without a period\n\npath/to/module.py\n"""\n\ndef hello():\n    print("Hello")'
        result = validate_module_docstring(content, "path/to/module.py", r"^[A-Z].+\.$")
        self.assertTrue(result.has_issues())
        self.assertTrue(any("first line" in e.lower() for e in result.errors))

    def test_docstring_without_path(self):
        """Test validation of a docstring without the module path."""
        content = '"""Module example with a valid first line.\n\nThis is missing the module path.\n"""\n\ndef hello():\n    print("Hello")'
        result = validate_module_docstring(content, "path/to/module.py", r"^[A-Z].+\.$")
        self.assertTrue(result.has_issues())
        self.assertTrue(any("path" in e.lower() for e in result.errors))

    def test_single_line_docstring(self):
        """Test validation of a single-line docstring."""
        content = '"""Module example with a single-line docstring including path/to/module.py."""\n\ndef hello():\n    print("Hello")'
        result = validate_module_docstring(content, "path/to/module.py", r"^[A-Z].+\.$")
        self.assertFalse(result.has_issues())

    def test_docstring_after_shebang_and_encoding(self):
        """Test validation of a docstring after shebang and encoding."""
        content = '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\n"""Module example.\n\npath/to/module.py\n"""\n\ndef hello():\n    print("Hello")'
        result = validate_module_docstring(content, "path/to/module.py", r"^[A-Z].+\.$")
        self.assertFalse(result.has_issues())

    def test_fix_missing_docstring(self):
        """Test fixing a missing docstring."""
        content = 'def hello():\n    print("Hello")'
        result = validate_module_docstring(content, "path/to/module.py", r"^[A-Z].+\.$")
        fixed = fix_module_docstring(content, result, "path/to/module.py")
        # Check if the fixed content contains the expected docstring elements
        self.assertIn('"""', fixed)
        self.assertIn("path/to/module.py", fixed)
        self.assertIn("Module", fixed)  # Should have capitalized module name

    def test_fix_docstring_first_line(self):
        """Test fixing a docstring's first line."""
        content = '"""module example without capitalization or period\n\npath/to/module.py\n"""\n\ndef hello():\n    print("Hello")'
        result = validate_module_docstring(content, "path/to/module.py", r"^[A-Z].+\.$")
        fixed = fix_module_docstring(content, result, "path/to/module.py")
        # Check if the first content line is properly capitalized and has a period
        lines = fixed.split("\n")
        # In multi-line docstrings, the first line is just the opening quotes,
        # and the actual content starts on the second line
        content_line = (
            lines[1] if lines[0].strip() == '"""' else lines[0].replace('"""', "", 1)
        )
        self.assertTrue(re.match(r"^[A-Z].+\.$", content_line.strip()))

    def test_fix_docstring_add_path(self):
        """Test fixing a docstring by adding the module path."""
        content = '"""Module example with a valid first line.\n\nThis is missing the module path.\n"""\n\ndef hello():\n    print("Hello")'
        result = validate_module_docstring(content, "path/to/module.py", r"^[A-Z].+\.$")
        fixed = fix_module_docstring(content, result, "path/to/module.py")
        # Check if the path was added to the docstring
        self.assertIn("path/to/module.py", fixed)


if __name__ == "__main__":
    unittest.main()
