# Project Documentation


## Directory Structure
```

├── .github
│   └── workflows
│       ├── ci.yml
│       └── publish.yml
├── examples
│   └── pre-commit-config.yaml
├── src
│   └── vibelint
│       ├── validators
│       │   ├── __init__.py
│       │   ├── docstring.py
│       │   ├── encoding.py
│       │   └── shebang.py
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── lint.py
│       ├── namespace.py
│       ├── report.py
│       └── utils.py
├── .gitignore
├── .pre-commit-hooks.yaml
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
├── README.md
├── setup.py
└── tox.ini

```


## File Contents


File: .github/workflows/ci.yml

--------------------------------------------------------------------------------

name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install tox tox-gh-actions
    - name: Test with tox
      run: tox

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install flake8 black isort
    - name: Lint with flake8
      run: |
        flake8 src tests
    - name: Check formatting with black
      run: |
        black --check src tests
    - name: Check imports with isort
      run: |
        isort --check-only --profile black src tests

--------------------------------------------------------------------------------


File: .github/workflows/publish.yml

--------------------------------------------------------------------------------

name: Publish to PyPI

on:
  release:
    types: [created]
  
  # Allow manual triggering
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: pypi
    
    permissions:
      # This permission is required for OIDC authentication with PyPI
      id-token: write
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build
    
    - name: Build package
      run: python -m build
    
    - name: Publish package
      uses: pypa/gh-action-pypi-publish@release/v1

--------------------------------------------------------------------------------


File: examples/pre-commit-config.yaml

--------------------------------------------------------------------------------

repos:
-   repo: https://github.com/mithranm/vibelint
    rev: 0.1.0
    hooks:
    -   id: vibelint

--------------------------------------------------------------------------------


File: src/vibelint/validators/__init__.py

--------------------------------------------------------------------------------

"""
Validators package initialization module.

vibelint/validators/__init__.py
"""

from .shebang import validate_shebang, fix_shebang
from .encoding import validate_encoding_cookie, fix_encoding_cookie
from .docstring import validate_module_docstring, fix_module_docstring

__all__ = [
    "validate_shebang",
    "fix_shebang",
    "validate_encoding_cookie",
    "fix_encoding_cookie",
    "validate_module_docstring",
    "fix_module_docstring"
]

--------------------------------------------------------------------------------


File: src/vibelint/validators/docstring.py

--------------------------------------------------------------------------------

"""
Validator for Python module docstrings, extended to auto-create missing docstrings.

vibelint/validators/docstring.py
"""

import os
import ast
from typing import List, Optional, Dict, Tuple, Any

# Sentinel used for docstring detection
MISSING_DOCSTRING = object()


class DocstringValidationResult:
    """
    Class to store the result of a validation.

    vibelint/validators/docstring.py (vibelint.validators.docstring)
    """
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.line_number: int = -1
        self.needs_fix: bool = False

        # If the file has a module-level docstring, store it for reference
        self.module_docstring: Optional[str] = None

        # Key = (start_line, end_line) for existing docstrings
        #   If docstring is missing, we store (lineno, lineno) for that node
        #   Value = { "type":..., "name":..., "message":..., "missing": bool, etc. }
        self.docstring_issues: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def has_issues(self) -> bool:
        """
        Check if there are any issues.

        vibelint/validators/docstring.py (vibelint.validators.docstring)
        """
        return (
            len(self.errors) > 0
            or len(self.warnings) > 0
            or len(self.docstring_issues) > 0
        )


def get_normalized_filepath(relative_path: str) -> str:
    """
    Extract the normalized filepath to be included in docstrings.

    For files in src/, returns path relative to src/.
    Otherwise, returns path relative to project root (best guess).
    """
    base = os.path.basename(relative_path)
    if base == relative_path:
        return relative_path

    if "/src/" in relative_path:
        return relative_path.split("/src/")[-1]

    parts = relative_path.split("/")
    if "vibelint" in parts:
        idx = parts.index("vibelint")
        return "/".join(parts[idx:])

    return relative_path


def extract_all_docstrings_with_missing(content: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Return a dictionary mapping line ranges to docstring info. If the docstring is missing,
    store docstring=None and use (node.lineno, node.lineno) as the dict key.

    E.g. dict[(start_line, end_line)] = {
       'type': 'module'|'class'|'function'|'method',
       'name': str,   # e.g. "MyClass.__init__"
       'docstring': str or None,
    }
    """
    results = {}
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return results

    # 1) Possibly capture the module docstring
    if tree.body and isinstance(tree.body[0], ast.Expr):
        maybe_doc = tree.body[0].value
        if (isinstance(maybe_doc, ast.Constant) and isinstance(maybe_doc.value, str)):
            doc_str = maybe_doc.value
            lineno = tree.body[0].lineno
            end_lineno = lineno + len(doc_str.splitlines()) - 1
            results[(lineno, end_lineno)] = {
                "type": "module",
                "name": "module",
                "docstring": doc_str,
            }
        elif (isinstance(maybe_doc, ast.Str)):  # older Python
            doc_str = maybe_doc.s
            lineno = tree.body[0].lineno
            end_lineno = lineno + len(doc_str.splitlines()) - 1
            results[(lineno, end_lineno)] = {
                "type": "module",
                "name": "module",
                "docstring": doc_str,
            }
        # else no module docstring
    else:
        # If you want to force a missing "module docstring," you could do:
        results[(1,1)] = { "type":"module", "name":"module", "docstring": None }
        pass

    def record_doc(node, node_type, parent_name=""):
        """Record a docstring or note that it's missing."""
        doc_text = None
        if node.body and isinstance(node.body[0], ast.Expr):
            val = node.body[0].value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                doc_text = val.value
            elif isinstance(val, ast.Str):
                doc_text = val.s

        # Build a fully qualified name if there's a parent
        name = getattr(node, "name", node_type)
        if parent_name and node_type != "module":
            name = f"{parent_name}.{name}"

        if doc_text is None:
            # docstring is missing
            results[(node.lineno, node.lineno)] = {
                "type": node_type,
                "name": name,
                "docstring": None
            }
        else:
            lineno = node.body[0].lineno
            end_lineno = lineno + len(doc_text.splitlines()) - 1
            results[(lineno, end_lineno)] = {
                "type": node_type,
                "name": name,
                "docstring": doc_text
            }

    # 2) For classes / functions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            record_doc(node, "class")
            # Also for each method
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    record_doc(child, "method", class_name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # if it's top-level
            parent = getattr(node, "parent", None)
            if not parent or not isinstance(parent, ast.ClassDef):
                record_doc(node, "function")

    return results


def _docstring_includes_path(doc_lines: List[str], relative_path: str, package_path: str) -> bool:
    """
    Check if doc_lines contain either slash-based or dot-based path references.
    """
    import_path = package_path.replace("/", ".").rstrip(".py")
    for ln in doc_lines:
        # Check slash-based, direct relative path, or dot-based
        if package_path in ln or relative_path in ln:
            return True
        if import_path and import_path in ln:
            return True
        if import_path and ln.strip().startswith(import_path + "."):
            return True
    return False


def _split_docstring_lines(docstring: Optional[str]) -> List[str]:
    """Split docstring text into stripped lines (filtering out empty lines)."""
    if not docstring:
        return []
    lines = docstring.splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def validate_module_docstring(
    content: str,
    relative_path: str,
    check_all_docstrings: bool = False,
) -> DocstringValidationResult:
    """
    Validate docstrings for module/class/function/method. If missing, or missing the path reference, log an error.
    """
    vr = DocstringValidationResult()
    package_path = get_normalized_filepath(relative_path)
    doc_map = extract_all_docstrings_with_missing(content)
    if not doc_map:
        # None found at all
        vr.errors.append("No docstrings found in file (all missing).")
        vr.needs_fix = True
        return vr

    for (start_line, end_line), info in doc_map.items():
        d_type = info["type"]
        d_name = info["name"]
        text = info["docstring"]
        lines = _split_docstring_lines(text)

        # If docstring is missing => text=None
        if text is None:
            # Only enforce for module docstring if !check_all_docstrings.
            if d_type == "module" or check_all_docstrings:
                msg = f"Missing docstring for {d_type} '{d_name}'."
                vr.errors.append(msg)
                vr.docstring_issues[(start_line, end_line)] = {
                    "type": d_type,
                    "name": d_name,
                    "missing": True,
                    "message": msg,
                }
                vr.needs_fix = True
        else:
            # The docstring exists, but may be missing the path reference
            if d_type == "module" or check_all_docstrings:
                if not _docstring_includes_path(lines, relative_path, package_path):
                    msg = f"Docstring for {d_type} '{d_name}' must include file path: {package_path}"
                    vr.errors.append(msg)
                    vr.docstring_issues[(start_line, end_line)] = {
                        "type": d_type,
                        "name": d_name,
                        "missing": False,
                        "message": msg,
                    }
                    vr.needs_fix = True

            if d_type == "module":
                vr.module_docstring = text
                vr.line_number = start_line

    return vr


def fix_module_docstring(
    content: str,
    validation_result: DocstringValidationResult,
    relative_path: str,
    fix_all_docstrings: bool = False,
) -> str:
    """
    Attempt to fix:
      - Missing docstrings (insert a brand new docstring after `def`/`class`).
      - Missing path references (remove old references, add the correct path line).
    """
    if not validation_result.needs_fix:
        return content

    lines = content.split("\n")
    package_path = get_normalized_filepath(relative_path)
    import_path = package_path.replace("/", ".").rstrip(".py")

    # Sort docstring issues from bottom to top so our line slicing doesn't shift everything
    doc_issues_sorted = sorted(
        validation_result.docstring_issues.items(),
        key=lambda x: x[0][0],
        reverse=True
    )

    def create_docstring_block(indent: str, doc_type: str, doc_name: str) -> List[str]:
        """
        Return a multi-line docstring block, referencing the file path.
        Something like:
            \"\"\"
            Docstring for MyClass.__init__.

            vibelint/validators/foo.py (vibelint.validators.foo)
            \"\"\"
        """
        block = []
        block.append(f'{indent}"""')
        block.append(f"{indent}Docstring for {doc_name}.")
        block.append("")  # blank line for spacing
        block.append(f"{indent}{package_path} ({import_path})")
        block.append(f'{indent}"""')
        return block

    def fix_existing_block(original_lines: List[str]) -> List[str]:
        """
        For an existing docstring, remove any old path references, then add exactly one reference line.
        """
        if not original_lines:
            return original_lines

        # Detect triple quote style
        joined = "\n".join(original_lines)
        triple_quote = '"""'
        if joined.strip().startswith("'''"):
            triple_quote = "'''"

        # Strip the triple quotes from the ends for easier processing
        trimmed = joined.strip()
        if trimmed.startswith(triple_quote):
            trimmed = trimmed[len(triple_quote):].lstrip()
        if trimmed.endswith(triple_quote):
            trimmed = trimmed[:-len(triple_quote)].rstrip()

        # Split lines
        lines_no_quotes = trimmed.splitlines()

        # Collect indentation from first line of original_lines
        first_line = original_lines[0]
        lead_spaces = ""
        for ch in first_line:
            if ch in [" ", "\t"]:
                lead_spaces += ch
            else:
                break

        # Remove old references
        new_body = []
        for ln in lines_no_quotes:
            if (package_path in ln or import_path in ln):
                continue
            new_body.append(ln)

        # Insert the path line
        new_body.append(f"{package_path} ({import_path})")

        # Rebuild the docstring block
        updated = []
        updated.append(f"{lead_spaces}{triple_quote}")
        for nb in new_body:
            updated.append(f"{lead_spaces}{nb}")
        updated.append(f"{lead_spaces}{triple_quote}")

        return updated

    for (lines_range, issue_info) in doc_issues_sorted:
        (start_line, end_line) = lines_range
        d_type = issue_info["type"]
        d_name = issue_info["name"]
        missing = issue_info.get("missing", False)

        # skip if doc_type != "module" and fix_all_docstrings==False
        if d_type != "module" and not fix_all_docstrings:
            continue

        if missing:
            # Insert a docstring after the line of "def" / "class"
            # The node's definition line is (start_line-1) in zero-based
            def_index = start_line - 1
            if def_index < 0:
                def_index = 0

            # figure out indentation from that line
            def_line = lines[def_index]
            leading_spaces = ""
            for ch in def_line:
                if ch in (" ", "\t"):
                    leading_spaces += ch
                else:
                    break

            block = create_docstring_block(leading_spaces, d_type, d_name)
            # Insert after the signature line
            insertion_point = def_index + 1
            lines = lines[:insertion_point] + block + lines[insertion_point:]
        else:
            # We have an existing docstring from lines[start_line-1 : end_line]
            slice_start = start_line - 1
            slice_end = end_line
            existing_block = lines[slice_start:slice_end]
            fixed_block = fix_existing_block(existing_block)
            lines[slice_start:slice_end] = fixed_block

    return "\n".join(lines)


__all__ = [
    "DocstringValidationResult",
    "validate_module_docstring",
    "fix_module_docstring",
    "extract_all_docstrings_with_missing",
    "get_normalized_filepath",
]

--------------------------------------------------------------------------------


File: src/vibelint/validators/encoding.py

--------------------------------------------------------------------------------

"""
Validator for Python encoding cookies.

vibelint/validators/encoding.py (vibelint.validators.encoding)
"""

import re
from typing import List


class EncodingValidationResult:
    """
    Class to store the result of a validation for encoding cookies.

    vibelint/validators/encoding.py (vibelint.validators.encoding)
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.line_number: int = -1
        self.needs_fix: bool = False

    def has_issues(self) -> bool:
        """
        Check if there are any issues.

        vibelint/validators/encoding.py (vibelint.validators.encoding)
        """
        return (len(self.errors) > 0) or (len(self.warnings) > 0)


def validate_encoding_cookie(content: str) -> EncodingValidationResult:
    """
    Validate the encoding cookie in a Python file.

    vibelint/validators/encoding.py (vibelint.validators.encoding)
    """
    result = EncodingValidationResult()
    lines = content.splitlines()
    encoding_pattern = r"^# -\*- coding: (.+) -\*-$"

    start_line = 0
    if lines and lines[0].startswith("#!"):
        start_line = 1

    if start_line < len(lines):
        match = re.match(encoding_pattern, lines[start_line])
        if match:
            encoding = match.group(1)
            result.line_number = start_line
            if encoding.lower() != "utf-8":
                result.errors.append(
                    f"Invalid encoding cookie: {encoding}. Use 'utf-8' instead."
                )
                result.needs_fix = True

    return result


def fix_encoding_cookie(content: str, result: EncodingValidationResult) -> str:
    """
    Fix encoding cookie issues in a Python file.

    vibelint/validators/encoding.py (vibelint.validators.encoding)
    """
    if not result.needs_fix:
        return content

    lines = content.splitlines()
    if result.line_number >= 0 and result.line_number < len(lines):
        lines[result.line_number] = "# -*- coding: utf-8 -*-"
    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")


__all__ = [
    "EncodingValidationResult",
    "validate_encoding_cookie",
    "fix_encoding_cookie",
]

--------------------------------------------------------------------------------


File: src/vibelint/validators/shebang.py

--------------------------------------------------------------------------------

"""
Validator for Python shebangs.

vibelint/validators/shebang.py (vibelint.validators.shebang)
"""

from typing import List


class ShebangValidationResult:
    """
    Class to store the result of a shebang validation.

    vibelint/validators/shebang.py (vibelint.validators.shebang)
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.line_number: int = 0
        self.needs_fix: bool = False

    def has_issues(self) -> bool:
        """
        Check if there are any issues.

        vibelint/validators/shebang.py (vibelint.validators.shebang)
        """
        return bool(self.errors or self.warnings)


def validate_shebang(
    content: str,
    is_script: bool,
    allowed_shebangs: List[str]
) -> ShebangValidationResult:
    """
    Validate the shebang in a Python file.

    vibelint/validators/shebang.py (vibelint.validators.shebang)
    """
    result = ShebangValidationResult()
    lines = content.splitlines()

    has_shebang = (len(lines) > 0) and lines[0].startswith("#!")
    if has_shebang:
        result.line_number = 0
        sb = lines[0]
        if not is_script:
            result.errors.append(
                f"File has a shebang ({sb}) but no '__main__' block. "
                "Shebangs should only be used in executable scripts."
            )
            result.needs_fix = True
        elif sb not in allowed_shebangs:
            result.errors.append(
                f"Invalid shebang: {sb}. Allowed shebangs: {', '.join(allowed_shebangs)}"
            )
            result.needs_fix = True
    else:
        if is_script:
            result.warnings.append(
                "Script with '__main__' block should have a shebang line."
            )
            result.needs_fix = True
            result.line_number = 0

    return result


def fix_shebang(
    content: str,
    result: ShebangValidationResult,
    is_script: bool,
    preferred_shebang: str
) -> str:
    """
    Fix shebang issues in a Python file.

    vibelint/validators/shebang.py (vibelint.validators.shebang)
    """
    if not result.needs_fix:
        return content

    lines = content.splitlines()

    if result.line_number == 0 and lines and lines[0].startswith("#!"):
        if not is_script:
            lines.pop(0)
        else:
            lines[0] = preferred_shebang
    elif is_script and (not lines or not lines[0].startswith("#!")):
        lines.insert(0, preferred_shebang)

    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")


__all__ = [
    "ShebangValidationResult",
    "validate_shebang",
    "fix_shebang",
]

--------------------------------------------------------------------------------


File: src/vibelint/__init__.py

--------------------------------------------------------------------------------

"""
Vibelint package initialization module.

vibelint/__init__.py
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("vibelint")
except PackageNotFoundError:
    # Package is not installed
    try:
        from ._version import version as __version__  # type: ignore
    except ImportError:
        __version__ = "unknown"

--------------------------------------------------------------------------------


File: src/vibelint/cli.py

--------------------------------------------------------------------------------

#!/usr/bin/env python3
"""
Command-line interface for vibelint.

vibelint/cli.py (vibelint.cli)
"""

import sys
from pathlib import Path
from typing import List

import click
from rich.console import Console
from rich.table import Table

from .lint import LintRunner
from .config import load_config
from .namespace import (
    build_namespace_tree_representation, 
    get_namespace_collisions_str, 
    detect_namespace_collisions,
    detect_soft_member_collisions
)
from .report import generate_markdown_report


console = Console()

@click.group()
@click.version_option()
def cli():
    """
    vibelint - A linting tool to make Python codebases more LLM-friendly.

    vibelint/cli.py (vibelint.cli)
    """
    pass


@cli.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=".",
    help="Path to directory to analyze (default: current directory)",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Check for violations without fixing them",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Skip confirmation for large directories",
)
@click.option(
    "--include-vcs-hooks",
    is_flag=True,
    help="Include version control hooks in analysis",
)
@click.option(
    "--check-all-docstrings",
    is_flag=True,
    help="Check all docstrings (functions, classes, methods) for filepath requirements, not just module docstrings",
)
@click.argument("paths", nargs=-1, type=click.Path(exists=True, readable=True))
def headers(
    path: str,
    check_only: bool,
    yes: bool,
    include_vcs_hooks: bool,
    check_all_docstrings: bool,
    paths: List[str],
):
    """Lint and fix Python module headers.

    If PATHS are provided, only those files/directories will be analyzed.
    Otherwise, all Python files under PATH will be analyzed.



vibelint/cli.py (vibelint.cli)
    """
    root_path = Path(path).resolve()
    config = load_config(root_path)

    # Use provided paths if available, otherwise use the root path
    target_paths = [Path(p).resolve() for p in paths] if paths else [root_path]

    lint_runner = LintRunner(
        config=config,
        check_only=check_only,
        skip_confirmation=yes,
        include_vcs_hooks=include_vcs_hooks,
        check_all_docstrings=check_all_docstrings,
    )

    exit_code = lint_runner.run(target_paths)
    sys.exit(exit_code)


@cli.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=".",
    help="Path to directory to analyze (default: current directory)",
)
@click.option(
    "--include-vcs-hooks",
    is_flag=True,
    help="Include version control hooks in analysis",
)
@click.option(
    "--show-collisions",
    is_flag=True,
    help="Additionally show namespace collisions",
)
@click.argument("paths", nargs=-1, type=click.Path(exists=True, readable=True))
def namespace(
    path: str,
    include_vcs_hooks: bool,
    show_collisions: bool,
    paths: List[str],
):
    """Visualize the namespace tree of a Python project.
    
    Displays the hierarchical structure of modules and their members.
    
    If PATHS are provided, only those files/directories will be analyzed.
    Otherwise, all Python files under PATH will be analyzed.



    vibelint/cli.py (vibelint.cli)
    """
    root_path = Path(path).resolve()
    config = load_config(root_path)

    # Use provided paths if available, otherwise use the root path
    target_paths = [Path(p).resolve() for p in paths] if paths else [root_path]

    # Show namespace tree
    tree = build_namespace_tree_representation(target_paths, config)
    console.print(tree)
    
    # Optionally show collisions - pass the console instance
    if show_collisions:
        collision_str = get_namespace_collisions_str(target_paths, config, console=console)
        if collision_str:
            console.print(collision_str)


@cli.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=".",
    help="Path to directory to analyze (default: current directory)",
)
@click.option(
    "--ignore-inheritance",
    is_flag=True,
    help="Don't consider inheritance when detecting soft collisions",
)
@click.option(
    "--soft-only",
    is_flag=True,
    help="Only show soft collisions (member names reused in unrelated modules)",
)
@click.option(
    "--hard-only",
    is_flag=True,
    help="Only show hard collisions (namespace conflicts)",
)
@click.argument("paths", nargs=-1, type=click.Path(exists=True, readable=True))
def collisions(
    path: str,
    ignore_inheritance: bool,
    soft_only: bool,
    hard_only: bool,
    paths: List[str],
):
    """Detect namespace collisions in Python code.
    
    Finds both hard collisions (naming conflicts that break Python) and soft collisions.
    (member names that appear in unrelated modules and may confuse humans or LLMs).
    
    If PATHS are provided, only those files/directories will be analyzed.
    Otherwise, all Python files under PATH will be analyzed.



    vibelint/cli.py (vibelint.cli)
    """
    root_path = Path(path).resolve()
    config = load_config(root_path)

    # Use provided paths if available, otherwise use the root path
    target_paths = [Path(p).resolve() for p in paths] if paths else [root_path]
    
    console.print("[bold]Checking for namespace collisions...[/bold]")
    
    # Detect collisions - pass the console instance to soft collisions 
    hard_collisions = [] if soft_only else detect_namespace_collisions(target_paths, config)
    soft_collisions = [] if hard_only else detect_soft_member_collisions(
        target_paths, config, use_inheritance_check=not ignore_inheritance, console=console
    )
    
    # Create summary table
    table = Table(title="Collision Results Summary")
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="magenta")
    
    table.add_row("Hard Collisions", str(len(hard_collisions)))
    table.add_row("Soft Collisions", str(len(soft_collisions)))
    table.add_row("Total", str(len(hard_collisions) + len(soft_collisions)))
    
    console.print(table)
    
    # Display detailed results
    if not hard_collisions and not soft_collisions:
        console.print("[green]✓ No namespace collisions detected[/green]")
        sys.exit(0)
    
    # Show detailed reports
    if hard_collisions:
        console.print("\n[bold red]Hard Collisions:[/bold red]")
        console.print("[dim](These can break Python imports)[/dim]")
        for collision in hard_collisions:
            console.print(
                f"- [red]'{collision.name}'[/red] in [cyan]{collision.path1}[/cyan] and [cyan]{collision.path2}[/cyan]"
            )
            
    if soft_collisions:
        console.print("\n[bold yellow]Soft Collisions:[/bold yellow]")
        console.print("[dim](These don't break Python but may confuse humans and LLMs)[/dim]")
        for collision in soft_collisions:
            console.print(
                f"- [yellow]'{collision.name}'[/yellow] in [cyan]{collision.path1}[/cyan] and [cyan]{collision.path2}[/cyan]"
            )
    
    # Exit with error code if hard collisions found (soft collisions are just warnings)
    sys.exit(1 if hard_collisions else 0)


@cli.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=".",
    help="Path to directory to analyze (default: current directory)",
)
@click.option(
    "-o", "--output",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="./vibelint_reports",
    help="Output directory for the report (default: ./vibelint_reports)",
)
@click.option(
    "-f", "--filename",
    type=str,
    default=None,
    help="Output filename for the report (default: vibelint_report_TIMESTAMP.md)",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Only report issues without suggesting fixes",
)
@click.option(
    "--include-vcs-hooks",
    is_flag=True,
    help="Include version control hooks in analysis",
)
@click.option(
    "--ignore-inheritance",
    is_flag=True,
    help="Don't consider inheritance when detecting soft collisions",
)
@click.argument("paths", nargs=-1, type=click.Path(exists=True, readable=True))
def report(
    path: str,
    output: str,
    filename: str,
    check_only: bool,
    include_vcs_hooks: bool,
    ignore_inheritance: bool,
    paths: List[str],
):
    """Generate a comprehensive markdown report of the codebase.
    
    The report includes linting errors, namespace structure, collisions,.
    and file contents organized by namespace hierarchy.
    
    If PATHS are provided, only those files/directories will be analyzed.
    Otherwise, all Python files under PATH will be analyzed.



    vibelint/cli.py (vibelint.cli)
    """
    root_path = Path(path).resolve()
    output_path = Path(output).resolve()
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(root_path)
    
    # Use provided paths if available, otherwise use the root path
    target_paths = [Path(p).resolve() for p in paths] if paths else [root_path]
    
    console.print(f"[bold]Generating comprehensive report for {len(target_paths)} path(s)...[/bold]")
    
    # Generate the report
    report_path = generate_markdown_report(
        target_paths=target_paths,
        output_dir=output_path,
        config=config,
        check_only=check_only,
        include_vcs_hooks=include_vcs_hooks,
        ignore_inheritance=ignore_inheritance,
        output_filename=filename
    )
    
    console.print(f"[green]✓ Report generated successfully at:[/green] {report_path}")
    return 0


def main():
    """
    Entry point for the CLI.

    vibelint/cli.py (vibelint.cli)
    """
    cli()


if __name__ == "__main__":
    main()

--------------------------------------------------------------------------------


File: src/vibelint/config.py

--------------------------------------------------------------------------------

"""
Configuration handling for vibelint.


vibelint/config.py
vibelint/config.py
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import copy

# Import tomllib for Python 3.11+, fallback to tomli for earlier versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


DEFAULT_CONFIG = {
    "package_root": "",
    "allowed_shebangs": ["#!/usr/bin/env python3"],
    "docstring_regex": r"^[A-Z].+\.$",
    "include_globs": ["**/*.py"],
    "exclude_globs": [
        "**/tests/**",
        "**/migrations/**",
        "**/site-packages/**",
        "**/dist-packages/**",
    ],
    "large_dir_threshold": 500,
}


def find_pyproject_toml(directory: Path) -> Optional[Path]:
    """
    Find the pyproject.toml file by traversing up from the given directory.

vibelint/config.py
    """
    current = directory.absolute()
    while current != current.parent:
        pyproject_path = current / "pyproject.toml"
        if pyproject_path.exists():
            return pyproject_path
        current = current.parent
    return None


def load_toml_config(config_path: Path, section: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a TOML file.
    
    Args:
        config_path: Path to the TOML config file
        section: Optional section to read (e.g. "tool.vibelint")
        
    Returns:
        Dictionary with configuration values

vibelint/config.py
    """
    try:
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
            
        if section:
            # Navigate nested sections (e.g. "tool.vibelint")
            parts = section.split('.')
            for part in parts:
                if part in config_data:
                    config_data = config_data[part]
                else:
                    return {}
        
        return config_data
    except (tomllib.TOMLDecodeError, OSError) as e:
        print(f"Warning: Error loading {config_path}: {str(e)}", file=sys.stderr)
        return {}


def load_user_config() -> Dict[str, Any]:
    """
    Load user configuration from ~/.config/vibelint/config.toml if it exists.

vibelint/config.py
    """
    config_path = Path.home() / ".config" / "vibelint" / "config.toml"
    if not config_path.exists():
        return {}

    return load_toml_config(config_path)


def load_project_config(directory: Path) -> Dict[str, Any]:
    """
    Load project configuration from pyproject.toml under [tool.vibelint].

vibelint/config.py
    """
    pyproject_path = find_pyproject_toml(directory)
    if not pyproject_path:
        return {}
        
    return load_toml_config(pyproject_path, "tool.vibelint")


def load_config(directory: Path) -> Dict[str, Any]:
    """
    Load configuration by merging default, user, and project configurations.

vibelint/config.py
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    user_config = load_user_config()
    project_config = load_project_config(directory)

    # Update with user config first, then project config (project has higher precedence)
    config.update(user_config)
    config.update(project_config)

    return config

--------------------------------------------------------------------------------


File: src/vibelint/lint.py

--------------------------------------------------------------------------------

"""
Core linting functionality for vibelint.

vibelint/lint.py
"""

import re
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import fnmatch

import click
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

# Our updated docstring validator/fixer
from .validators.docstring import (
    validate_module_docstring,
    fix_module_docstring,
)
# Shebang and encoding remain the same
from .validators.shebang import validate_shebang, fix_shebang
from .validators.encoding import validate_encoding_cookie, fix_encoding_cookie

console = Console()


class LintResult:
    """
    Class to store the result of a linting operation.

    vibelint/lint.py (vibelint.lint)
    """

    def __init__(self):
        self.file_path: Path = Path()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.fixed: bool = False

    @property
    def has_issues(self) -> bool:
        return bool(self.errors or self.warnings)


class LintRunner:
    """
    Runner class for linting operations, updated to handle missing docstrings.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        check_only: bool = False,
        skip_confirmation: bool = False,
        include_vcs_hooks: bool = False,
        check_all_docstrings: bool = False,
    ):
        self.config = config
        self.check_only = check_only
        self.skip_confirmation = skip_confirmation
        self.include_vcs_hooks = include_vcs_hooks
        self.check_all_docstrings = check_all_docstrings

        self.results: List[LintResult] = []
        self.files_fixed: int = 0
        self.files_with_errors: int = 0
        self.files_with_warnings: int = 0

    def run(self, paths: List[Path]) -> int:
        """
        Run the linting process on the specified paths.
        """
        python_files = self._collect_python_files(paths)
        if not python_files:
            console.print("[yellow]No Python files found to lint.[/yellow]")
            return 0

        # Confirm if large
        threshold = self.config.get("large_dir_threshold", 500)
        if not self.skip_confirmation and len(python_files) > threshold:
            if not self._confirm_large_directory(len(python_files)):
                console.print("[yellow]Operation cancelled.[/yellow]")
                return 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                f"Linting {len(python_files)} Python files...", total=len(python_files)
            )
            with ThreadPoolExecutor() as executor:
                for result in executor.map(self._process_file, python_files):
                    self.results.append(result)
                    progress.advance(task_id)

        # Summarize
        for r in self.results:
            if r.fixed:
                self.files_fixed += 1
            if r.errors:
                self.files_with_errors += 1
            elif r.warnings:
                self.files_with_warnings += 1

        self._print_summary()

        # Return non-zero if errors, or if we found fixable issues but the user was in check-only
        if self.files_with_errors > 0 or (self.check_only and self.files_fixed > 0):
            return 1
        return 0

    def _collect_python_files(self, paths: List[Path]) -> List[Path]:
        """
        Collect all Python files based on config globs/excludes.
        """
        python_files: List[Path] = []
        for path in paths:
            if path.is_file() and path.suffix == ".py":
                python_files.append(path)
            elif path.is_dir():
                for include_glob in self.config.get("include_globs", ["**/*.py"]):
                    for file_path in path.glob(include_glob):
                        if not file_path.is_file() or file_path.suffix != ".py":
                            continue
                        if not self.include_vcs_hooks and any(
                            part.startswith(".") and part in {".git", ".hg", ".svn"}
                            for part in file_path.parts
                        ):
                            continue
                        # Check excludes
                        exclude_globs = self.config.get("exclude_globs", [])
                        if any(
                            fnmatch.fnmatch(str(file_path), str(path / ex))
                            for ex in exclude_globs
                        ):
                            continue
                        python_files.append(file_path)
        return python_files

    def _confirm_large_directory(self, file_count: int) -> bool:
        console.print(
            f"[yellow]Warning:[/yellow] Found {file_count} Python files to lint, "
            f"which exceeds the large_dir_threshold of {self.config['large_dir_threshold']}."
        )
        return click.confirm("Do you want to continue?", default=True)

    def _process_file(self, file_path: Path) -> LintResult:
        """
        Process a single Python file (shebang, encoding, docstring).
        """
        lr = LintResult()
        lr.file_path = file_path

        try:
            content = file_path.read_text(encoding="utf-8")
            new_content = content

            # is_script if we see 'if __name__ == "__main__":'
            is_script = bool(re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", content))

            # Figure out a "relative" path if user has package_root
            relative_path = str(file_path)
            package_root = self.config.get("package_root", "")
            if package_root:
                try:
                    relative_path = str(file_path.relative_to(package_root))
                except ValueError:
                    pass

            # 1) Shebang
            shebang_res = validate_shebang(content, is_script, self.config["allowed_shebangs"])
            if shebang_res.has_issues():
                lr.errors.extend(shebang_res.errors)
                lr.warnings.extend(shebang_res.warnings)
                if not self.check_only:
                    new_content = fix_shebang(new_content, shebang_res, is_script, self.config["allowed_shebangs"][0])

            # 2) Encoding
            enc_res = validate_encoding_cookie(content)
            if enc_res.has_issues():
                lr.errors.extend(enc_res.errors)
                lr.warnings.extend(enc_res.warnings)
                if not self.check_only:
                    new_content = fix_encoding_cookie(new_content, enc_res)

            # 3) Docstrings (extended logic for missing docstrings)
            doc_res = validate_module_docstring(
                content,
                relative_path,
                check_all_docstrings=self.check_all_docstrings,
            )
            if doc_res.has_issues():
                lr.errors.extend(doc_res.errors)
                lr.warnings.extend(doc_res.warnings)
                if not self.check_only:
                    new_content = fix_module_docstring(
                        new_content,
                        doc_res,
                        relative_path,
                        fix_all_docstrings=self.check_all_docstrings,
                    )

            if new_content != content and not self.check_only:
                file_path.write_text(new_content, encoding="utf-8")
                lr.fixed = True

        except Exception as e:
            lr.errors.append(f"Error processing file {file_path}: {e}")

        return lr

    def _print_summary(self):
        """
        Print summary table of results.
        """
        table = Table(title="vibelint Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")

        table.add_row("Files processed", str(len(self.results)))
        table.add_row("Files fixed", str(self.files_fixed))
        table.add_row("Files with errors", str(self.files_with_errors))
        table.add_row("Files with warnings", str(self.files_with_warnings))

        console.print(table)

        # List them out if any
        if self.files_with_errors > 0 or self.files_with_warnings > 0:
            console.print("\n[bold]Files with issues:[/bold]")
            for r in self.results:
                if r.has_issues:
                    status = "[red]ERROR[/red]" if r.errors else "[yellow]WARNING[/yellow]"
                    console.print(f"{status} {r.file_path}")
                    for err in r.errors:
                        console.print(f"  - [red]{err}[/red]")
                    for w in r.warnings:
                        console.print(f"  - [yellow]{w}[/yellow]")

--------------------------------------------------------------------------------


File: src/vibelint/namespace.py

--------------------------------------------------------------------------------

"""
Namespace representation and collision detection for vibelint.

vibelint/namespace.py
"""

import os
import ast
import fnmatch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING

from rich.tree import Tree
# Keep this import for type checking
if TYPE_CHECKING:
    from rich.console import Console


class CollisionType:
    """Enum-like class for collision types."""
    HARD = "hard"  # Name conflicts that break Python imports
    SOFT = "soft"  # Same name in different modules, potentially confusing


class NamespaceCollision:
    """
    Class to store information about a namespace collision.

    vibelint/namespace.py (vibelint.namespace)
    """

    def __init__(
        self,
        name: str,
        path1: Path,
        path2: Path,
        collision_type: str = CollisionType.HARD,
        note: str = ""
    ):
        self.name = name
        self.path1 = path1
        self.path2 = path2
        self.collision_type = collision_type
        self.note = note  # e.g. "[class duplication]" for repeated class names

    def __str__(self) -> str:
        type_str = "Hard" if self.collision_type == CollisionType.HARD else "Soft"
        note_str = f" {self.note}" if self.note else ""
        return f"{type_str} collision{note_str}: '{self.name}' in {self.path1} and {self.path2}"


class ClassInheritanceTracker:
    """Track inheritance relationships so we can see if name reuse is legitimate."""
    
    def __init__(self):
        self.inheritance_map: Dict[str, List[str]] = {}  # class -> parent classes
        self.class_locations: Dict[str, Path] = {}

    def add_class(self, class_name: str, parent_classes: List[str], file_path: Path, module_path: List[str]) -> None:
        qualified_name = ".".join([*module_path, class_name])
        if qualified_name not in self.inheritance_map:
            self.inheritance_map[qualified_name] = []
            self.class_locations[qualified_name] = file_path

        for p in parent_classes:
            self.inheritance_map[qualified_name].append(p)

    def is_related_through_inheritance(self, class1: str, class2: str) -> bool:
        """Check if two classes are related through inheritance in either direction."""
        if class1 == class2:
            return True

        # Check if class1 inherits from class2
        if class1 in self.inheritance_map:
            if class2 in self.inheritance_map[class1]:
                return True
            for parent in self.inheritance_map[class1]:
                if self.is_related_through_inheritance(parent, class2):
                    return True

        # Check if class2 inherits from class1
        if class2 in self.inheritance_map:
            if class1 in self.inheritance_map[class2]:
                return True
            for parent in self.inheritance_map[class2]:
                if self.is_related_through_inheritance(parent, class1):
                    return True

        return False


class NamespaceNode:
    """
    Class to represent a node in the namespace tree.

    vibelint/namespace.py (vibelint.namespace)
    """

    def __init__(self, name: str, path: Optional[Path] = None, is_package: bool = False):
        self.name = name
        self.path = path
        self.is_package = is_package
        self.children: Dict[str, "NamespaceNode"] = {}
        self.members: Dict[str, Path] = {}  # top-level names (func, class, etc.)
        self.file_path = path if path and path.is_file() else None

    def add_child(
        self, name: str, path: Path, is_package: bool = False
    ) -> "NamespaceNode":
        if name not in self.children:
            self.children[name] = NamespaceNode(name, path, is_package)
        return self.children[name]

    def add_member(self, name: str, path: Path):
        self.members[name] = path

    def get_collisions(self) -> List["NamespaceCollision"]:
        """
        Check collisions between child nodes and members,
        ensuring we do not pass None for path2.
        """
        collisions: List[NamespaceCollision] = []
        for member_name, member_path in self.members.items():
            if member_name in self.children:
                child_node = self.children[member_name]
                if child_node.path is not None:
                    collisions.append(NamespaceCollision(member_name, member_path, child_node.path))
        # Recurse
        for child in self.children.values():
            collisions.extend(child.get_collisions())
        return collisions

    def to_tree(self, parent_tree: Optional[Tree] = None) -> Tree:
        if parent_tree is None:
            tree = Tree(f":package: {self.name}" if self.is_package else self.name)
        else:
            tree = parent_tree.add(f":package: {self.name}" if self.is_package else self.name)

        if self.members:
            mem_branch = tree.add(":page_facing_up: Members")
            for nm in sorted(self.members.keys()):
                mem_branch.add(nm)
        for cname, cnode in sorted(self.children.items()):
            cnode.to_tree(tree)
        return tree


def _extract_module_members(file_path: Path) -> List[str]:
    """
    Return top-level names for functions, classes, or assigned variables in a Python module.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except Exception:
        return []

    members = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            members.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    members.append(target.id)
    return members


def _build_namespace_tree(
    paths: List[Path], config: Dict[str, Any], include_vcs_hooks: bool = False
) -> NamespaceNode:
    """
    Build a namespace tree, skipping excluded globs / VCS dirs, etc.
    """
    root = NamespaceNode("root")
    python_files: List[Path] = []

    for path in paths:
        if path.is_file() and path.suffix == ".py":
            python_files.append(path)
        elif path.is_dir():
            includes = config.get("include_globs", ["**/*.py"])
            excludes = config.get("exclude_globs", [])
            for pat in includes:
                for fpath in path.glob(pat):
                    if not fpath.is_file() or fpath.suffix != ".py":
                        continue
                    if not include_vcs_hooks and any(
                        part.startswith(".") and part in {".git", ".hg", ".svn"}
                        for part in fpath.parts
                    ):
                        continue
                    if any(fnmatch.fnmatch(str(fpath), str(path / e)) for e in excludes):
                        continue
                    python_files.append(fpath)

    if python_files:
        files_str = [str(p) for p in python_files]
        common_prefix = os.path.commonpath(files_str)

        for f in python_files:
            rel_path = str(f).replace(common_prefix, "").lstrip(os.sep)
            parts = rel_path.split(os.sep)
            file_name = parts[-1]
            current = root
            for i, part in enumerate(parts[:-1]):
                init_potential = Path(common_prefix, *parts[:i+1], "__init__.py")
                is_pack = init_potential.exists()
                current = current.add_child(part, Path(common_prefix, *parts[:i+1]), is_pack)

            # Is it an __init__.py?
            mod_name = file_name[:-3]
            is_pack = (mod_name == "__init__")
            if is_pack:
                # add members to parent
                mmbrs = _extract_module_members(f)
                for mm in mmbrs:
                    current.add_member(mm, f)
            else:
                mnode = current.add_child(mod_name, f)
                mmbrs = _extract_module_members(f)
                for mm in mmbrs:
                    mnode.add_member(mm, f)

    return root


def build_namespace_tree_representation(
    paths: List[Path], config: Dict[str, Any]
) -> Tree:
    ns_tree = _build_namespace_tree(paths, config, include_vcs_hooks=False)
    return ns_tree.to_tree()


def detect_namespace_collisions(paths: List[Path], config: Dict[str, Any]) -> List[NamespaceCollision]:
    """
    Basic collisions (child vs. member) plus collisions in __init__ files vs. local modules.
    """
    ns_tree = _build_namespace_tree(paths, config, include_vcs_hooks=False)
    collisions = ns_tree.get_collisions()

    init_files = []
    python_modules = {}
    for path in paths:
        if path.is_file():
            if path.name == "__init__.py":
                init_files.append(path)
            elif path.suffix == ".py":
                python_modules[path.stem] = path
        elif path.is_dir():
            for f in path.rglob("*.py"):
                if f.name == "__init__.py":
                    init_files.append(f)
                else:
                    python_modules[f.stem] = f

    for initf in init_files:
        package_dir = initf.parent
        i_map, i_mods, i_all = _extract_imports_and_all(initf)

        # name conflict between imported symbols and local modules
        for modname, mpath in python_modules.items():
            if mpath.parent == package_dir and modname in i_map:
                collisions.append(NamespaceCollision(
                    modname, mpath, initf, CollisionType.HARD
                ))

        # repeated names in __all__
        name_counts = {}
        for nm in i_all:
            name_counts[nm] = name_counts.get(nm, 0) + 1
        for nm, cnt in name_counts.items():
            if cnt > 1:
                collisions.append(NamespaceCollision(
                    nm, initf, initf, CollisionType.HARD
                ))

        # check if __all__ references a local module also imported
        for nm in i_all:
            test_mod = package_dir / f"{nm}.py"
            if test_mod.exists() and nm in i_map:
                collisions.append(NamespaceCollision(
                    nm, test_mod, initf, CollisionType.HARD
                ))

    return collisions


def _extract_imports_and_all(fpath: Path):
    """
    Parse a file, returning:
       - import_map (dict of local_name -> origin)
       - imported_modules (list of modules imported)
       - all_names (list from __all__)
    """
    import_map = {}
    imported_mods = []
    all_names = []
    try:
        src = fpath.read_text(encoding="utf-8")
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                for nm in node.names:
                    local_name = nm.asname or nm.name
                    origin = f"{mod_name}.{nm.name}" if mod_name else nm.name
                    import_map[local_name] = origin
            elif isinstance(node, ast.Import):
                for nm in node.names:
                    if nm.asname:
                        import_map[nm.asname] = nm.name
                    else:
                        imported_mods.append(nm.name)
                        # map local name => full name
                        local = nm.name.split(".")[-1]
                        import_map[local] = nm.name
            elif isinstance(node, ast.Assign):
                # check for __all__
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_names.append(elt.value)
                                elif isinstance(elt, ast.Str):
                                    all_names.append(elt.s)
    except Exception:
        pass
    return import_map, imported_mods, all_names


def detect_soft_member_collisions(
    paths: List[Path],
    config: Dict[str, Any],
    use_inheritance_check: bool = True,
    console: Optional["Console"] = None
) -> List[NamespaceCollision]:
    """
    Return a list of collisions for repeated member names that are not inheritance-related.
    
    Args:
        paths: List of paths to analyze
        config: Configuration dictionary
        use_inheritance_check: Whether to check class inheritance relationships
        console: Optional console for output (if None, no output is printed)
    """
    # Remove the local console creation
    ns_tree = _build_namespace_tree(paths, config, include_vcs_hooks=False)
    # Gather all members => name -> list of (file_path, is_class, module_path)
    member_map: Dict[str, List[Tuple[Path, bool, List[str]]]] = {}

    # We track inheritance via ClassInheritanceTracker
    tracker = ClassInheritanceTracker() if use_inheritance_check else None

    def get_class_inheritance(file_path: Path, mod_path: List[str]):
        """
        Extract class definitions with their direct parent classes.
        """
        try:
            txt = file_path.read_text(encoding="utf-8")
            t = ast.parse(txt)
            for nd in ast.walk(t):
                if isinstance(nd, ast.ClassDef):
                    # get base classes
                    parents = []
                    for base in nd.bases:
                        if isinstance(base, ast.Name):
                            parents.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            chain = []
                            cur = base
                            while isinstance(cur, ast.Attribute):
                                chain.append(cur.attr)
                                cur = cur.value
                            if isinstance(cur, ast.Name):
                                chain.append(cur.id)
                            chain.reverse()
                            parents.append(".".join(chain))
                    if tracker:
                        tracker.add_class(nd.name, parents, file_path, mod_path)
        except Exception:
            pass

    def traverse(node: NamespaceNode, mod_path: List[str]):
        # This node might be a module if node.file_path is a .py file
        # Add members
        for mname, mp in node.members.items():
            # We guess if it's a class by reading the file AST:
            # but let's do a simpler approach: if there's a ClassDef with that name
            # We'll do a quick check here or rely on the next method:
            # For best detection, we rely on the AST pass for classes.
            is_class = False
            # quick guess: if we see 'class <mname>' in the file
            # that might be good enough. But let's do a small parse:

            # We'll store (file_path, is_class, mod_path).
            # We'll do a second pass with the class inheritance if needed
            # For now, let's skip an extra parse for each symbol. We'll do a global parse for classes below.
            member_map.setdefault(mname, []).append((mp, is_class, mod_path))

        # traverse children
        for cname, cnode in node.children.items():
            # if cnode is actual .py file and not __init__
            new_path = mod_path + [cname] if cnode.file_path else mod_path
            traverse(cnode, new_path)

    # Build the tree, then gather
    traverse(ns_tree, [])

    # Now let's refine which members are actually classes by building an AST for each file once
    # and telling the tracker about them
    # Then we can set is_class = True for each "class" in that file
    if tracker:
        # 1) gather all classes into tracker
        for path in paths:
            if path.is_file() and path.suffix == ".py":
                # we guess the "module path" from the tree approach, but let's do a partial approach
                # the collisions logic won't be perfect if we can't find it. We'll do best-effort
                # For demonstration, let's do something simpler:
                mod_path = []
                get_class_inheritance(path, mod_path)

        # 2) Mark each top-level name that is in the tracker's inheritance_map as class
        #    We'll do a naive approach: if "ClassName" is in the file, that is a class
        #    There's no perfect 1:1 mapping from "ValidationResult" to "some.path.ValidationResult"
        #    so we do best effort
        all_class_names = set()
        for full_name, _parents in tracker.inheritance_map.items():
            # e.g. "SomeModule.MyClass"
            # the last part is the class name
            raw_class_name = full_name.split(".")[-1]
            all_class_names.add(raw_class_name)

        # Now we set is_class = True for any symbol that matches a known class name
        for sym_name, entries in member_map.items():
            if sym_name in all_class_names:
                updated = []
                for (pth, is_class_flag, mpth) in entries:
                    updated.append((pth, True, mpth))
                member_map[sym_name] = updated

    # Now we do the collision detection
    soft_collisions: List[NamespaceCollision] = []
    for mname, occurrences in member_map.items():
        if len(occurrences) < 2:
            continue
        # For each pair
        for i in range(len(occurrences)):
            p1, is_class1, modp1 = occurrences[i]
            for j in range(i+1, len(occurrences)):
                p2, is_class2, modp2 = occurrences[j]
                if p1 == p2:
                    continue  # same file

                # if we have a tracker, see if they're related
                if tracker and is_class1 and is_class2:
                    # we guess the "qualified_name" as just the class name
                    # This is not 100% accurate for a large codebase
                    # but let's do it for demonstration
                    c1 = mname
                    c2 = mname
                    # The same name -> we check if that might be the same or inherited
                    related = tracker.is_related_through_inheritance(c1, c2)
                    if related:
                        continue

                    # Not related => collision. Mark it with [class duplication]
                    note = "[class duplication]"
                    sc = NamespaceCollision(mname, p1, p2, collision_type=CollisionType.SOFT, note=note)
                    soft_collisions.append(sc)
                else:
                    # normal collision. They share the same symbol but are not the same file
                    sc = NamespaceCollision(mname, p1, p2, collision_type=CollisionType.SOFT)
                    soft_collisions.append(sc)

    return soft_collisions


def get_namespace_collisions_str(paths: List[Path], config: Dict[str, Any], 
                               console: Optional["Console"] = None) -> str:
    """
    Get a string representation of namespace collisions.
    
    Args:
        paths: List of paths to analyze
        config: Configuration dictionary
        console: Optional console for output (if None, a new one is created)
    """
    # Import here to avoid circular imports but make it available for runtime
    from rich.console import Console
    
    # If no console is provided, create one for this function only
    if console is None:
        console = Console(width=100, record=True)
    
    ns_tree = _build_namespace_tree(paths, config, include_vcs_hooks=False)
    collisions = ns_tree.get_collisions()
    if collisions:
        console.print("\n[bold red]Namespace Collisions:[/bold red]")
        for c in collisions:
            console.print(f"- [red]{c}[/red]")

    return console.export_text()


def get_files_in_namespace_order(namespace_tree: NamespaceNode) -> List[Path]:
    files = []
    visited = set()

    def traverse(node: NamespaceNode):
        if node.file_path and node.file_path not in visited:
            visited.add(node.file_path)
            files.append(node.file_path)
        for cname in sorted(node.children.keys()):
            traverse(node.children[cname])

    traverse(namespace_tree)
    return files

--------------------------------------------------------------------------------


File: src/vibelint/report.py

--------------------------------------------------------------------------------

"""
Report generation functionality for vibelint.



vibelint/report.py
vibelint/report.py
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from rich.console import Console

from .lint import LintRunner
from .namespace import (
    build_namespace_tree_representation,
    detect_namespace_collisions,
    detect_soft_member_collisions,
    get_files_in_namespace_order,
    _build_namespace_tree  # Import the internal function directly
)
from .utils import find_package_root

console = Console()

def generate_markdown_report(
    target_paths: List[Path],
    output_dir: Path,
    config: Dict[str, Any],
    check_only: bool = True,
    include_vcs_hooks: bool = False,
    ignore_inheritance: bool = False,
    output_filename: Optional[str] = None
) -> Path:
    """
Generate a comprehensive markdown report of linting results, namespace structure,.
    and file contents.
    
    Args:
        target_paths: List of paths to analyze
        output_dir: Directory where the report will be saved
        config: Configuration dictionary
        check_only: Only check for issues without suggesting fixes
        include_vcs_hooks: Whether to include version control hooks
        ignore_inheritance: Whether to ignore inheritance when checking for soft collisions
        output_filename: Optional filename for the report (default: vibelint_report_{timestamp}.md)
        
    Returns:
        Path to the generated report file


vibelint/report.py
    """
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"vibelint_report_{timestamp}.md"
    elif not output_filename.endswith('.md'):
        # Ensure the file has a markdown extension
        output_filename = f"{output_filename}.md"
        
    report_path = output_dir / output_filename
    
    # Run linting
    lint_runner = LintRunner(
        config=config,
        check_only=True,  # Always check only for reports
        skip_confirmation=True,
        include_vcs_hooks=include_vcs_hooks,
    )
    
    console.print("[bold blue]Running linting checks...[/bold blue]")
    lint_runner.run(target_paths)
    
    # Build namespace tree representation for display
    console.print("[bold blue]Building namespace structure...[/bold blue]")
    tree_repr = build_namespace_tree_representation(target_paths, config)
    
    # Get the actual namespace tree node for file ordering
    # Use _build_namespace_tree directly instead of trying to extract from tree_repr
    namespace_tree = _build_namespace_tree(target_paths, config, include_vcs_hooks)
    
    # Detect collisions
    console.print("[bold blue]Detecting namespace collisions...[/bold blue]")
    hard_collisions = detect_namespace_collisions(target_paths, config)
    soft_collisions = detect_soft_member_collisions(
        target_paths, config, use_inheritance_check=not ignore_inheritance
    )
    
    # Generate report
    console.print("[bold blue]Generating markdown report...[/bold blue]")
    with open(report_path, "w", encoding="utf-8") as f:
        # Report header
        package_roots = [find_package_root(path) for path in target_paths]
        package_names = []
        for p in package_roots:
            if p is not None and p.exists() and p.name:
                package_names.append(p.name)
        
        f.write("# vibelint Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(f"**Project(s):** {', '.join(package_names) or 'Unknown'}\n\n")
        f.write(f"**Paths analyzed:** {', '.join(str(p) for p in target_paths)}\n\n")
        
        # Table of Contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Summary](#summary)\n")
        f.write("2. [Linting Results](#linting-results)\n")
        f.write("3. [Namespace Structure](#namespace-structure)\n")
        f.write("4. [Namespace Collisions](#namespace-collisions)\n")
        f.write("5. [File Contents](#file-contents)\n\n")
        
        # Summary section
        f.write("## Summary\n\n")
        f.write("| Metric | Count |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Files analyzed | {len(lint_runner.results)} |\n")
        f.write(f"| Files with errors | {lint_runner.files_with_errors} |\n")
        f.write(f"| Files with warnings | {lint_runner.files_with_warnings} |\n")
        f.write(f"| Hard namespace collisions | {len(hard_collisions)} |\n")
        f.write(f"| Soft namespace collisions | {len(soft_collisions)} |\n\n")
        
        # Linting Results
        f.write("## Linting Results\n\n")
        if not lint_runner.results:
            f.write("*No linting results available.*\n\n")
        else:
            f.write("| File | Errors | Warnings |\n")
            f.write("|------|--------|----------|\n")
            for result in lint_runner.results:
                if result.has_issues:
                    errors = "; ".join(result.errors) or "None"
                    warnings = "; ".join(result.warnings) or "None"
                    f.write(f"| `{result.file_path}` | {errors} | {warnings} |\n")
            f.write("\n")
        
        # Namespace Structure
        f.write("## Namespace Structure\n\n")
        f.write("```\n")
        f.write(str(tree_repr))
        f.write("\n```\n\n")
        
        # Namespace Collisions
        f.write("## Namespace Collisions\n\n")
        
        # Hard collisions
        f.write("### Hard Collisions\n\n")
        if not hard_collisions:
            f.write("*No hard collisions detected.*\n\n")
        else:
            f.write("These collisions can break Python imports:\n\n")
            f.write("| Name | Path 1 | Path 2 |\n")
            f.write("|------|--------|--------|\n")
            for collision in hard_collisions:
                f.write(f"| `{collision.name}` | {collision.path1} | {collision.path2} |\n")
            f.write("\n")
        
        # Soft collisions
        f.write("### Soft Collisions\n\n")
        if not soft_collisions:
            f.write("*No soft collisions detected.*\n\n") 
        else:
            f.write("These don't break Python but may confuse humans and LLMs:\n\n")
            f.write("| Name | Path 1 | Path 2 |\n")
            f.write("|------|--------|--------|\n")
            for collision in soft_collisions:
                f.write(f"| `{collision.name}` | {collision.path1} | {collision.path2} |\n")
            f.write("\n")
        
        # File Contents
        f.write("## File Contents\n\n")
        f.write("Files are ordered by their position in the namespace hierarchy.\n\n")
        
        # Get all Python files from the namespace tree in a logical order
        python_files = get_files_in_namespace_order(namespace_tree)
        
        for file_path in python_files:
            if file_path.is_file() and file_path.suffix == '.py':
                rel_path = get_relative_path(file_path, target_paths)
                f.write(f"### {rel_path}\n\n")
                try:
                    with open(file_path, 'r', encoding='utf-8') as code_file:
                        content = code_file.read()
                        f.write("```python\n")
                        f.write(content)
                        f.write("\n```\n\n")
                except Exception as e:
                    f.write(f"*Error reading file: {str(e)}*\n\n")
    
    return report_path


def get_relative_path(file_path: Path, base_paths: List[Path]) -> str:
    """
    Get the relative path of a file from the closest base path.
    
    Args:
        file_path: The file path to get the relative path for
        base_paths: List of base paths to use as reference
        
    Returns:
        The relative path as a string


vibelint/report.py
    """
    shortest_path = None
    
    for base_path in base_paths:
        try:
            rel_path = file_path.relative_to(base_path)
            if shortest_path is None or len(str(rel_path)) < len(str(shortest_path)):
                shortest_path = rel_path
        except ValueError:
            continue
            
    return str(shortest_path) if shortest_path else str(file_path)

--------------------------------------------------------------------------------


File: src/vibelint/utils.py

--------------------------------------------------------------------------------

"""
Utility functions for vibelint.


vibelint/utils.py
vibelint/utils.py
"""

from pathlib import Path
from typing import Dict, Any, Optional
import fnmatch


def find_package_root(path: Path) -> Optional[Path]:
    """
    Find the package root directory by looking for a setup.py, pyproject.toml, or __init__.py file.
    
    Args:
        path: Path to start searching from
        
    Returns:
        The package root directory, or None if not found

vibelint/utils.py
    """
    if path.is_file():
        path = path.parent
    
    current = path
    
    # First try to find setup.py or pyproject.toml
    while current.parent != current:
        if (current / "setup.py").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    # If not found, try looking for the top-level __init__.py
    current = path
    while current.parent != current:
        if not (current / "__init__.py").exists() and (current.parent / "__init__.py").exists():
            return current
        if not (current.parent / "__init__.py").exists():
            # Return the last directory that contained __init__.py
            return current
        current = current.parent
    
    # If no package structure found, return the original directory
    return path


def count_python_files(
    directory: Path, config: Dict[str, Any], include_vcs_hooks: bool = False
) -> int:
    """
    Count the number of Python files in a directory that match the configuration.

    vibelint/utils.py
    """
    count = 0

    for include_glob in config["include_globs"]:
        for file_path in directory.glob(include_glob):
            # Skip if it's not a file or not a Python file
            if not file_path.is_file() or file_path.suffix != ".py":
                continue

            # Skip VCS directories unless explicitly included
            if not include_vcs_hooks and any(
                part.startswith(".") and part in {".git", ".hg", ".svn"}
                for part in file_path.parts
            ):
                continue

            # Check exclude patterns
            if any(
                fnmatch.fnmatch(str(file_path), str(directory / exclude_glob))
                for exclude_glob in config["exclude_globs"]
            ):
                continue

            count += 1

    return count

--------------------------------------------------------------------------------


File: .gitignore

--------------------------------------------------------------------------------

__pycache__/
*.py[cod]
*.pyo
*.pyd
*.egg-info/
*.env
*$py.class

.dirfilter
o.md

--------------------------------------------------------------------------------


File: .pre-commit-hooks.yaml

--------------------------------------------------------------------------------

- id: vibelint
  name: vibelint
  description: Make Python codebases more LLM-friendly
  entry: vibelint headers
  language: python
  types: [python]
  pass_filenames: true

--------------------------------------------------------------------------------


File: LICENSE

--------------------------------------------------------------------------------

MIT License

Copyright (c) 2025 vibelint Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

--------------------------------------------------------------------------------


File: MANIFEST.in

--------------------------------------------------------------------------------

include LICENSE
include README.md
include pyproject.toml
include .pre-commit-hooks.yaml
include tox.ini

recursive-include examples *
recursive-include tests *.py

recursive-exclude * __pycache__
recursive-exclude * *.py[cod]
recursive-exclude * *.so
recursive-exclude * .*.swp
recursive-exclude * .DS_Store

--------------------------------------------------------------------------------


File: pyproject.toml

--------------------------------------------------------------------------------

[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "vibelint"
version = "0.1.1"
description = "Suite of linting tools to enhance the vibe coding process."
authors = [
  { name = "Mithran Mohanraj", email = "mithran.mohanraj@gmail.com" }
]
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Quality Assurance",
]
dependencies = [
    "click>=8.1.0",
    "tomli>=2.0.0; python_version < '3.11'",
    "colorama>=0.4.0",
    "rich>=12.0.0",
]

[project.scripts]
vibelint = "vibelint.cli:main"

[project.urls]
"Homepage" = "https://github.com/mithranm/vibelint"
"Bug Tracker" = "https://github.com/mithranm/vibelint/issues"

[tool.setuptools.packages.find]
where = ["src"]
include = ["vibelint*"]

[tool.vibelint]
package_root = ""
allowed_shebangs = ["#!/usr/bin/env python3"]
docstring_regex = "^[A-Z].+\\.$"
include_globs = ["**/*.py"]
exclude_globs = ["**/tests/**", "**/migrations/**", "**/site-packages/**", "**/dist-packages/**"]
large_dir_threshold = 500

--------------------------------------------------------------------------------


File: README.md

--------------------------------------------------------------------------------

# vibelint

**WARNING**: This entire project was almost zero-shotted by Claude 3.7 Sonnet Thinking. Bugs are expected.

A linting tool to make Python codebases more LLM-friendly while maintaining human readability.

## Installation

```bash
pip install vibelint
```

## Usage

```bash
# Check headers in the current directory
vibelint headers

# Check headers in a specific directory
vibelint headers --path /path/to/project

# Check without making changes
vibelint headers --check-only

# Force check on a large directory without confirmation
vibelint headers --yes

# Include version control system hooks
vibelint headers --include-vcs-hooks

# Show namespace representation
vibelint headers --show-namespace
```

## Configuration

vibelint can be configured via `pyproject.toml` or through a global configuration file at `~/.config/vibelint/config.toml`.

Example configuration in `pyproject.toml`:

```toml
[tool.vibelint]
package_root = "mypackage"
allowed_shebangs = ["#!/usr/bin/env python3"]
docstring_regex = "^[A-Z].+\\.$"
include_globs = ["**/*.py"]
exclude_globs = ["**/tests/**", "**/migrations/**"]
large_dir_threshold = 500
```

## Features

- **Shebang validation**: Ensures shebang lines are correct and only present when needed
- **Encoding cookie validation**: Validates UTF-8 encoding declarations
- **Module docstring validation**: Ensures all Python modules have proper docstrings
- **Auto-fix**: Automatically fixes issues when possible
- **Namespace analysis**: Detects namespace collisions across modules
- **Performance**: Processes hundreds of files quickly with parallel execution
- **Pre-commit hook**: Can be integrated into your pre-commit workflow

## License

MIT

--------------------------------------------------------------------------------


File: setup.py

--------------------------------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for vibelint.

setup.py
vibelint/setup.py (vibelint.setu)
"""
"""

from setuptools import setup

if __name__ == "__main__":
    setup()

--------------------------------------------------------------------------------


File: tox.ini

--------------------------------------------------------------------------------

[tox]
envlist = py310, py311, py312, flake8, black, isort
isolated_build = True

[gh-actions]
python =
    3.10: py310, flake8, black, isort
    3.11: py311
    3.12: py312

[testenv]
deps =
    pytest>=7.0.0
    pytest-cov>=4.0.0
commands =
    pytest {posargs:tests} --cov=vibelint --cov-report=xml

[testenv:flake8]
deps = flake8>=6.0.0
commands = flake8 src tests

[testenv:black]
deps = black>=23.0.0
commands = black --check src tests

[testenv:isort]
deps = isort>=5.12.0
commands = isort --check-only --profile black src tests

[flake8]
max-line-length = 160
exclude = .tox,*.egg,build,data
select = E,W,F
extend-ignore = 
    E203,
    W291,
    W292,
    W293,
    W391,
    E302,
    E305,
    W503,

--------------------------------------------------------------------------------
