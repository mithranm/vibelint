#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the shebang validator.

tests/test_shebang.py
"""

import unittest
from vibelint.validators.shebang import validate_shebang, fix_shebang


class ShebangValidatorTests(unittest.TestCase):
    """Test cases for the shebang validator."""

    def test_valid_shebang_in_script(self):
        """Test validation of a valid shebang in a script."""
        content = (
            "#!/usr/bin/env python3\n\nif __name__ == '__main__':\n    print('Hello')"
        )
        result = validate_shebang(content, True, ["#!/usr/bin/env python3"])
        self.assertFalse(result.has_issues())

    def test_invalid_shebang_in_script(self):
        """Test validation of an invalid shebang in a script."""
        content = "#!/usr/bin/python\n\nif __name__ == '__main__':\n    print('Hello')"
        result = validate_shebang(content, True, ["#!/usr/bin/env python3"])
        self.assertTrue(result.has_issues())
        self.assertTrue(any("Invalid shebang" in e for e in result.errors))

    def test_shebang_in_non_script(self):
        """Test validation of a shebang in a non-script file."""
        content = "#!/usr/bin/env python3\n\ndef hello():\n    print('Hello')"
        result = validate_shebang(content, False, ["#!/usr/bin/env python3"])
        self.assertTrue(result.has_issues())
        self.assertTrue(any("no '__main__' block" in e for e in result.errors))

    def test_missing_shebang_in_script(self):
        """Test validation of a missing shebang in a script."""
        content = "if __name__ == '__main__':\n    print('Hello')"
        result = validate_shebang(content, True, ["#!/usr/bin/env python3"])
        self.assertTrue(result.has_issues())
        self.assertTrue(any("should have a shebang" in w for w in result.warnings))

    def test_fix_shebang_in_script(self):
        """Test fixing a shebang in a script."""
        content = "#!/usr/bin/python\n\nif __name__ == '__main__':\n    print('Hello')"
        result = validate_shebang(content, True, ["#!/usr/bin/env python3"])
        fixed = fix_shebang(content, result, True, "#!/usr/bin/env python3")
        self.assertEqual(
            fixed,
            "#!/usr/bin/env python3\n\nif __name__ == '__main__':\n    print('Hello')",
        )

    def test_fix_missing_shebang_in_script(self):
        """Test fixing a missing shebang in a script."""
        content = "if __name__ == '__main__':\n    print('Hello')"
        result = validate_shebang(content, True, ["#!/usr/bin/env python3"])
        fixed = fix_shebang(content, result, True, "#!/usr/bin/env python3")
        self.assertEqual(
            fixed,
            "#!/usr/bin/env python3\nif __name__ == '__main__':\n    print('Hello')",
        )

    def test_fix_remove_shebang_in_non_script(self):
        """Test removing a shebang from a non-script file."""
        content = "#!/usr/bin/env python3\n\ndef hello():\n    print('Hello')"
        result = validate_shebang(content, False, ["#!/usr/bin/env python3"])
        fixed = fix_shebang(content, result, False, "#!/usr/bin/env python3")
        self.assertEqual(fixed, "\ndef hello():\n    print('Hello')")


if __name__ == "__main__":
    unittest.main()
