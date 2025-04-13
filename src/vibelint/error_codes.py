"""
Defines error and warning codes used by vibelint, along with descriptions.

Codes follow the pattern VBL<category><id>
Categories:
1xx: Docstrings
2xx: Encoding
3xx: Exports (__all__)
4xx: Shebang
5xx: Namespace (Reserved for future use if needed for collision reporting)
9xx: Internal/Processing Errors

vibelint/error_codes.py
"""

VBL101 = "VBL101"
VBL102 = "VBL102"
# VBL103 removed


VBL201 = "VBL201"


VBL301 = "VBL301"
VBL302 = "VBL302"
VBL303 = "VBL303"
VBL304 = "VBL304"


VBL401 = "VBL401"
VBL402 = "VBL402"
VBL403 = "VBL403"


VBL901 = "VBL901"
VBL902 = "VBL902"
VBL903 = "VBL903"
VBL904 = "VBL904"
VBL905 = "VBL905"


CODE_DESCRIPTIONS = {
    VBL101: "Missing docstring for module, class, or function.",
    VBL102: "Docstring does not end with the expected relative file path reference.",
    VBL201: "Invalid encoding cookie value (must be 'utf-8').",
    VBL301: "`__all__` definition is missing in a module where it is required.",
    VBL302: "`__all__` definition is missing in `__init__.py` (Optional based on config).",
    VBL303: "`__all__` is assigned a value that is not a List or Tuple.",
    VBL304: "SyntaxError parsing file during `__all__` validation.",
    VBL401: "File has a shebang line (`#!...`) but no `if __name__ == '__main__'` block.",
    VBL402: "Shebang line value is not in the list of allowed shebangs (check config).",
    VBL403: "File contains `if __name__ == '__main__'` block but lacks a shebang line.",
    VBL901: "Error reading file content (permissions, encoding, etc.).",
    VBL902: "SyntaxError parsing file during validation.",
    VBL903: "Internal error during validation phase for a file.",
    VBL904: "Error occurred in file processing thread.",
    VBL905: "Critical unhandled error during processing of a file.",
}


__all__ = [
    "VBL101",
    "VBL102",
    "VBL201",
    "VBL301",
    "VBL302",
    "VBL303",
    "VBL304",
    "VBL401",
    "VBL402",
    "VBL403",
    "VBL901",
    "VBL902",
    "VBL903",
    "VBL904",
    "VBL905",
    "CODE_DESCRIPTIONS",
]
