#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the encoding cookie validator.

tests/test_encoding.py
"""

import unittest
from vibelint.validators.encoding import validate_encoding_cookie, fix_encoding_cookie


class EncodingCookieValidatorTests(unittest.TestCase):
    """Test cases for the encoding cookie validator."""

    def test_valid_encoding_cookie(self):
        """Test validation of a valid encoding cookie."""
        content = "# -*- coding: utf-8 -*-\n\ndef hello():\n    print('Hello')"
        result = validate_encoding_cookie(content)
        self.assertFalse(result.has_issues())

    def test_valid_encoding_cookie_after_shebang(self):
        """Test validation of a valid encoding cookie after a shebang."""
        content = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\ndef hello():\n    print('Hello')"
        result = validate_encoding_cookie(content)
        self.assertFalse(result.has_issues())

    def test_invalid_encoding_cookie(self):
        """Test validation of an invalid encoding cookie."""
        content = "# -*- coding: latin-1 -*-\n\ndef hello():\n    print('Hello')"
        result = validate_encoding_cookie(content)
        self.assertTrue(result.has_issues())
        self.assertTrue(any("Invalid encoding cookie" in e for e in result.errors))

    def test_missing_encoding_cookie(self):
        """Test validation of a missing encoding cookie (which is allowed)."""
        content = "def hello():\n    print('Hello')"
        result = validate_encoding_cookie(content)
        self.assertFalse(result.has_issues())

    def test_fix_invalid_encoding_cookie(self):
        """Test fixing an invalid encoding cookie."""
        content = "# -*- coding: latin-1 -*-\n\ndef hello():\n    print('Hello')"
        result = validate_encoding_cookie(content)
        fixed = fix_encoding_cookie(content, result)
        self.assertEqual(
            fixed, "# -*- coding: utf-8 -*-\n\ndef hello():\n    print('Hello')"
        )

    def test_fix_invalid_encoding_cookie_after_shebang(self):
        """Test fixing an invalid encoding cookie after a shebang."""
        content = "#!/usr/bin/env python3\n# -*- coding: latin-1 -*-\n\ndef hello():\n    print('Hello')"
        result = validate_encoding_cookie(content)
        fixed = fix_encoding_cookie(content, result)
        self.assertEqual(
            fixed,
            "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\ndef hello():\n    print('Hello')",
        )


if __name__ == "__main__":
    unittest.main()
