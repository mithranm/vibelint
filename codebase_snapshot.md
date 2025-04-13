# Snapshot

## Filesystem Tree

```
vibelint/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── publish.yml
├── examples/
│   └── pre-commit-config.yaml
├── src/
│   └── vibelint/
│       ├── validators/
│       │   ├── __init__.py
│       │   ├── docstring.py
│       │   ├── encoding.py
│       │   ├── exports.py
│       │   └── shebang.py
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── discovery.py
│       ├── error_codes.py
│       ├── lint.py
│       ├── namespace.py
│       ├── report.py
│       ├── results.py
│       ├── snapshot.py
│       └── utils.py
├── tests/
│   ├── fixtures/
│   │   ├── check_success/
│   │   │   └── myproject/
│   │   │       ├── src/
│   │   │       │   └── mypkg/
│   │   │       │       ├── __init__.py
│   │   │       │       └── module.py
│   │   │       └── pyproject.toml
│   │   └── fix_missing_all/
│   │       └── fixproj/
│   │           ├── another.py
│   │           ├── needs_fix.py
│   │           └── pyproject.toml
│   ├── pleasehelpmewritethese.txt
│   └── test_cli.py
├── .dirfilter
├── .gitignore
├── LICENSE
├── MANIFEST.in
├── README.md
├── pyproject.toml
└── tox.ini
```

## File Contents

Files are ordered alphabetically by path.

### File: .dirfilter

```
[ignore]
tests/
LICENSE
README.md
o.md
out.txt
code_tape_archive.md
```

---
### File: .github/workflows/ci.yml

```yaml
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
```

---
### File: .github/workflows/publish.yml

```yaml
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
```

---
### File: .gitignore

```
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.egg-info/
*.env
*$py.class

.DS_Store
```

---
### File: LICENSE

```
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
```

---
### File: MANIFEST.in

```
# Include these specific top-level files in the sdist
include LICENSE
include README.md
include pyproject.toml
# include .pre-commit-hooks.yaml # We decided to remove this
include tox.ini

# Recursively include all files (*) found within the 'examples' directory
recursive-include examples *
# Recursively include all Python files (*.py) found within the 'tests' directory
recursive-include tests *.py

# Recursively exclude any directory named '__pycache__' anywhere in the project
recursive-exclude * __pycache__
# Recursively exclude compiled Python files anywhere
recursive-exclude * *.py[cod]
# Recursively exclude compiled C extensions anywhere
recursive-exclude * *.so
# Recursively exclude VIM swap files anywhere
recursive-exclude * .*.swp
# Recursively exclude macOS metadata files anywhere
recursive-exclude * .DS_Store
```

---
### File: README.md

```markdown
# vibelint

[![CI](https://github.com/mithranm/vibelint/actions/workflows/ci.yml/badge.svg)](https://github.com/mithranm/vibelint/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/vibelint.svg)](https://badge.fury.io/py/vibelint)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Enhance your Python codebase's "vibe" for better maintainability and LLM interaction.**

`vibelint` is a suite of tools designed to identify and help resolve common Python code smells and anti-patterns that can hinder developer understanding and confuse Large Language Models (LLMs) used in AI-assisted coding. It helps you visualize your project's structure, detect naming conflicts, and enforce coding conventions that promote clarity.

## Table of Contents

*   [Why Use vibelint?](#why-use-vibelint)
*   [Key Features](#key-features)
*   [Installation](#installation)
*   [Usage](#usage)
    *   [Checking Your Codebase (Linting & Collisions)](#checking-your-codebase-linting--collisions)
    *   [Getting a Full Report](#getting-a-full-report)
    *   [Visualizing Your Namespace](#visualizing-your-namespace)
    *   [Creating Code Snapshots](#creating-code-snapshots)
    *   [Getting Help](#getting-help)
*   [Strategies for Namespace Cleanup](#strategies-for-namespace-cleanup)
*   [Configuration](#configuration)
    *   [Disabling the Docstring Path Check](#disabling-the-docstring-path-check)
*   [Error Codes](#error-codes)
*   [Contributing](#contributing)
*   [License](#license)

## Why Use vibelint?

Modern Python codebases can become complex, leading to issues that aren't syntax errors but degrade maintainability and clarity:

1.  **Hidden Namespace Conflicts:** It's easy to accidentally define the same function or class name in multiple modules. While Python might resolve imports one way, developers (and LLMs) can be confused about which implementation is intended or active in a given context. Hard collisions (e.g., a module name clashing with a variable in `__init__.py`) can even break imports unexpectedly.
2.  **Ambiguity for LLMs & Developers:** Tools like Copilot or ChatGPT rely heavily on context. Missing `__all__` definitions obscure a module's public API. Docstrings without clear file path references make it harder for both humans and AI to know *where* that code lives within the project structure, hindering understanding and accurate code generation/analysis.
3.  **Inconsistent Code Patterns:** Issues like missing docstrings, improper `__all__` usage, or incorrect shebangs create friction during development and code reviews.

`vibelint` helps you address these by:

*   **Revealing Structure:** Clearly visualizing your project's namespace.
*   **Preventing Errors:** Catching hard namespace collisions before they cause runtime import failures.
*   **Reducing Ambiguity:** Identifying soft collisions and enforcing explicit APIs (`__all__`) and contextual docstrings.
*   **Improving Maintainability:** Promoting consistent, understandable code patterns.
*   **Enhancing AI Collaboration:** Providing clearer context (via docstring paths and snapshots) for better results from LLM coding assistants.

## Key Features

*   **Namespace Visualization (`vibelint namespace`):** Generates a tree view of your project's Python namespace (packages, modules, `__init__.py` members).
*   **Collision Detection (`vibelint check`):**
    *   **Hard Collisions:** Name conflicts likely to break Python imports.
    *   **Global Soft Collisions:** Same name defined in multiple modules (potential ambiguity).
    *   **Local Soft Collisions:** Same name exported via `__all__` in sibling modules (confusing `import *`).
*   **Targeted Linting (`vibelint check`):**
    *   **Docstring Presence & Path:** Checks for docstrings and enforces the inclusion of a standardized relative file path at the end.
    *   **`__all__` Enforcement:** Ensures modules define their public API via `__all__`.
    *   **Shebang & Encoding:** Validates script shebangs and encoding declarations.
*   **Codebase Snapshot (`vibelint snapshot`):** Creates a single Markdown file with a file tree and code contents, respecting includes/excludes – ideal for LLM context.
*   **Comprehensive Reporting (`vibelint check -o report.md`):** Generates detailed Markdown reports summarizing all findings.

*(Note: vibelint currently focuses on identifying issues, not automatically fixing them.)*

## Installation

```bash
pip install vibelint
```

`vibelint` requires Python 3.10 or higher.

*   **Note on TOML parsing:** For Python 3.10, `vibelint` requires the `tomli` package. For Python 3.11+, it uses the built-in `tomllib`. This dependency is handled automatically by `pip`.

## Usage

Run `vibelint` commands from the root of your project (the directory containing `pyproject.toml` or `.git`).

### Checking Your Codebase (Linting & Collisions)

This is the primary command to analyze your project.

```bash
vibelint check
```

This runs all configured linters and namespace collision checks, printing a summary and details of any issues found to the console. It will exit with a non-zero code if errors (like hard collisions or missing `__all__` where required) are found.

### Getting a Full Report

To get a detailed breakdown of all linting issues, the namespace structure, detected collisions, and the content of included files, use the `-o` or `--output-report` option with the `check` command:

```bash
vibelint check -o vibelint-report.md
```

This will generate a comprehensive Markdown file (e.g., `vibelint-report.md`) in your current directory. This report is useful for reviewing issues offline or sharing with your team.

### Visualizing Your Namespace

To understand your project's Python structure:

```bash
vibelint namespace
```

This prints the namespace tree directly to your terminal.

*   **Save the tree to a file:**
    ```bash
    vibelint namespace -o namespace_tree.txt
    ```

### Creating Code Snapshots

Generate a single Markdown file containing the project structure and file contents (useful for LLMs):

```bash
vibelint snapshot
```

This creates `codebase_snapshot.md` by default.

*   **Specify a different output file:**
    ```bash
    vibelint snapshot -o context_for_llm.md
    ```
The snapshot respects the `include_globs`, `exclude_globs`, and `peek_globs` defined in your configuration.

### Getting Help

```bash
vibelint --help
vibelint check --help
vibelint namespace --help
vibelint snapshot --help
```

## Strategies for Namespace Cleanup

The `vibelint check` command might report namespace collisions. Here’s a suggested strategy for addressing them:

1.  **Prioritize Hard Collisions:** These are marked `[HARD]` and are the most critical as they can break Python's import mechanism or lead to very unexpected behavior.
    *   **Cause:** Typically a clash between a submodule/subpackage name and an object (variable, function, class) defined in a parent `__init__.py`. For example, having `src/utils/` directory and defining `utils = ...` in `src/__init__.py`.
    *   **Fix:** Rename one of the conflicting items. Usually, renaming the object in the `__init__.py` is less disruptive than renaming a whole directory/package. Choose a more descriptive name.

2.  **Address Local Soft Collisions (`__all__`):** These are marked `[LOCAL_SOFT]` and occur when multiple sibling modules (files in the same directory/package) export the same name via their `__all__` list. This mainly causes issues with wildcard imports (`from package import *`).
    *   **Review:** Is it necessary for the same name to be part of the public API of multiple sibling modules?
    *   **Fix Options:**
        *   Rename the object in one of the modules.
        *   Remove the name from the `__all__` list in one or more modules if it's not truly intended to be public from that specific module.
        *   Reconsider the package structure – perhaps the conflicting objects should live elsewhere or be consolidated.

3.  **Review Global Soft Collisions (Definitions):** These are marked `[GLOBAL_SOFT]` and indicate the same name (function, class, top-level variable) is defined in multiple modules anywhere in the project. These usually don't cause runtime errors but create ambiguity for developers and LLMs.
    *   **Evaluate:** Is the duplication intentional and necessary? Sometimes utility functions might be deliberately duplicated.
    *   **Fix Options:**
        *   If the logic is identical, consolidate the definition into a single shared module and import it where needed.
        *   If the logic differs but the name causes confusion, rename the object in one or more locations to be more specific.
        *   If the duplication is intentional (e.g., different implementations of an interface), ensure clear documentation distinguishes them. You might consider ignoring specific instances if the ambiguity is acceptable (see Configuration).

4.  **Use `vibelint namespace`:** Refer to the namespace visualization (`vibelint namespace`) output while refactoring to better understand the project structure you are modifying.

5.  **Iterate:** Don't try to fix everything at once. Start with hard collisions, then local soft, then global soft. Rerun `vibelint check` after making changes.

## Configuration

Configure `vibelint` by adding a `[tool.vibelint]` section to your `pyproject.toml` file.

```toml
# pyproject.toml

[tool.vibelint]
# Globs for files to include (relative to project root)
# Files matching these patterns will be considered for linting and snapshots.
include_globs = [
    "src/**/*.py",
    "tests/**/*.py",
    "scripts/*.py",
    "*.py" # Include top-level python files
]

# Globs for files/directories to exclude
# Files matching these are ignored, even if they match include_globs.
# Defaults usually cover common virtual envs, caches, etc.
exclude_globs = [
    ".git/**",
    ".tox/**",
    "*.egg-info/**",
    "build/**",
    "dist/**",
    "**/__pycache__/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    "*.env*",
    "**/.DS_Store",
    # Add project-specific ignores:
    "docs/**",
    "data/**",
]

# List of allowed shebang lines for executable scripts (checked by VBL402)
# Only applies to files containing a `if __name__ == "__main__":` block.
allowed_shebangs = ["#!/usr/bin/env python3"]

# If true, enforce __all__ presence in __init__.py files (VBL301).
# If false (default), only issue a warning (VBL302) for missing __all__ in __init__.py.
error_on_missing_all_in_init = false

# List of VBL error/warning codes to ignore globally.
# Find codes in src/vibelint/error_codes.py or from `vibelint check` output.
ignore = ["VBL102"] # Example: Ignore missing path references in docstrings

# Threshold for confirming before processing many files during `check`.
# Set to a very large number (or use --yes flag) to disable confirmation.
large_dir_threshold = 500

# Glob patterns for files whose content should be truncated (peeked)
# instead of fully included in `vibelint snapshot` output.
# Useful for large data files, logs, etc.
# peek_globs = [
#   "data/**/*.csv",
#   "logs/*.log",
# ]
```

### Disabling the Docstring Path Check

If you disagree with the convention of including the relative file path at the end of docstrings, you can disable the specific check (`VBL102`).

Add the code `VBL102` to the `ignore` list in your `pyproject.toml` under the `[tool.vibelint]` section:

```toml
[tool.vibelint]
# ... other settings ...
ignore = ["VBL102"]
```

You can add multiple codes to the list to ignore other specific checks if needed, e.g., `ignore = ["VBL102", "VBL302"]`.

## Error Codes

`vibelint` uses specific codes (e.g., `VBL101`, `VBL301`, `VBL402`) to identify issues found by the `check` command. These codes help you understand the exact nature of the problem and allow for targeted configuration (e.g., ignoring specific codes).

You can find the definition of these codes in the source file: `src/vibelint/error_codes.py`.

*   **VBL1xx:** Docstring issues (Presence, Path reference, Format)
*   **VBL2xx:** Encoding cookie issues
*   **VBL3xx:** `__all__` export issues (Presence, Format)
*   **VBL4xx:** Shebang (`#!`) issues (Presence, Validity)
*   **VBL9xx:** Internal processing errors

## Contributing

Contributions are welcome! Please feel free to open issues for bug reports or feature requests, or submit pull requests on GitHub.

## License

`vibelint` is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

---
### File: examples/pre-commit-config.yaml

```yaml
repos:
-   repo: https://github.com/mithranm/vibelint
    rev: 0.1.0
    hooks:
    -   id: vibelint
```

---
### File: pyproject.toml

```toml
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
    "docformatter>=1.7.0",
    "libcst"
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
allowed_shebangs = ["#!/usr/bin/env python3"]
include_globs = ["**/*"]
exclude_globs = [
    ".git/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".tox/**",
    "*.egg-info/**",
    "*.env*",
    "**/__pycache__/**",
    "**/.DS_Store",
    ".DS_Store",
    "debug.txt"
]
large_dir_threshold = 500
error_on_missing_all_in_init = false
```

---
### File: src/vibelint/__init__.py

```python
"""
vibelint package initialization module.

vibelint/__init__.py
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("vibelint")
except PackageNotFoundError:
    __version__ = "unknown"
```

---
### File: src/vibelint/cli.py

```python
"""
Command-line interface for vibelint.

Provides commands to check codebase health, visualize namespaces, and create snapshots.

vibelint/cli.py
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict

import click
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler


from .results import CheckResult, NamespaceResult, SnapshotResult, CommandResult


from .config import load_config, Config
from .lint import LintRunner
from .namespace import (
    build_namespace_tree,
    detect_hard_collisions,
    detect_global_definition_collisions,
    detect_local_export_collisions,
    NamespaceCollision,
)
from .snapshot import create_snapshot
from .report import write_report_content
from .utils import get_relative_path, find_project_root


ValidationIssue = Tuple[str, str]


class VibelintContext:
    """
    Context object to store command results and shared state.

    vibelint/cli.py
    """

    def __init__(self):
        """
        Initializes VibelintContext.

        vibelint/cli.py
        """

        self.command_result: Optional[CommandResult] = None
        self.lint_runner: Optional[LintRunner] = None
        self.project_root: Optional[Path] = None


__all__ = ["snapshot", "check", "cli", "namespace", "main", "VibelintContext"]


console = Console()
logger_cli = logging.getLogger("vibelint")


def _present_check_results(result: CheckResult, runner: LintRunner, console: Console):
    """
    Presents the results of the 'check' command, including detailed lint issues with codes.

    vibelint/cli.py
    """

    runner._print_summary()

    files_with_issues = sorted(
        [lr for lr in runner.results if lr.has_issues], key=lambda r: r.file_path
    )
    if files_with_issues:
        console.print("\n[bold yellow]Files with Issues:[/bold yellow]")
        for lr in files_with_issues:
            try:
                rel_path = (
                    get_relative_path(lr.file_path, runner.config.project_root)
                    if runner.config.project_root
                    else lr.file_path
                )
                console.print(f"\n[bold cyan]{rel_path}:[/bold cyan]")
            except ValueError:
                console.print(
                    f"\n[bold cyan]{lr.file_path}:[/bold cyan] ([yellow]Outside project?[/yellow])"
                )
            except Exception as e:
                console.print(
                    f"\n[bold cyan]{lr.file_path}:[/bold cyan] ([red]Error getting relative path: {e}[/red])"
                )

            for code, error_msg in lr.errors:
                console.print(f"  [red]✗ [{code}] {error_msg}[/red]")

            for code, warning_msg in lr.warnings:
                console.print(f"  [yellow]▲ [{code}] {warning_msg}[/yellow]")

    if (
        result.hard_collisions
        or result.global_soft_collisions
        or result.local_soft_collisions
    ):
        console.print()
        _display_collisions(
            result.hard_collisions,
            result.global_soft_collisions,
            result.local_soft_collisions,
            console,
        )
    else:
        logger_cli.debug("No namespace collisions detected.")

    if result.report_path:
        console.print()
        if result.report_generated:
            console.print(f"[green]✓ Report generated at {result.report_path}[/green]")
        elif result.report_error:
            console.print(
                f"\n[bold red]Error generating report:[/bold red] {result.report_error}"
            )
        else:
            console.print(
                f"[yellow]Report status unknown for {result.report_path}[/yellow]"
            )

    console.print()
    if result.exit_code != 0:
        console.print(
            f"[bold red]Check finished with errors (exit code {result.exit_code}).[/bold red]"
        )
    elif runner.results:
        console.print("[bold green]Check finished successfully.[/bold green]")
    else:
        console.print(
            "[bold blue]Check finished. No Python files found or processed.[/bold blue]"
        )


def _present_namespace_results(result: NamespaceResult, console: Console):
    """
    Presents the results of the 'namespace' command.

    vibelint/cli.py
    """

    if not result.success:
        console.print(
            f"[bold red]Error building namespace tree:[/bold red] {result.error_message}"
        )
        return

    if result.intra_file_collisions:
        console.print("\n[bold yellow]Intra-file Collisions Found:[/bold yellow]")
        console.print("These duplicate names were found within the same file:")
        ctx = click.get_current_context(silent=True)
        project_root = (
            ctx.obj.project_root
            if ctx and hasattr(ctx.obj, "project_root")
            else Path(".")
        )

        for c in sorted(
            result.intra_file_collisions, key=lambda x: (str(x.paths[0]), x.name)
        ):
            try:
                rel_path = get_relative_path(c.paths[0], project_root)
            except ValueError:
                rel_path = c.paths[0]

            loc1 = (
                f"{rel_path}:{c.linenos[0]}"
                if c.linenos and c.linenos[0]
                else str(rel_path)
            )
            line1 = c.linenos[0] if c.linenos else "?"
            line2 = c.linenos[1] if len(c.linenos) > 1 else "?"
            console.print(
                f"- '{c.name}': Duplicate definition/import in {loc1} (lines ~{line1} and ~{line2})"
            )

    if result.output_path:
        if result.intra_file_collisions:
            console.print()
        if result.output_saved:
            console.print(
                f"\n[green]✓ Namespace tree saved to {result.output_path}[/green]"
            )
        elif result.output_error:
            console.print(
                f"[bold red]Error saving namespace tree:[/bold red] {result.output_error}"
            )
        else:
            console.print(
                f"[yellow]Namespace tree status unknown for {result.output_path}[/yellow]"
            )
    elif result.root_node:
        if result.intra_file_collisions:
            console.print()
        console.print("\n[bold blue]Namespace Structure:[/bold blue]")
        console.print(str(result.root_node))


def _present_snapshot_results(result: SnapshotResult, console: Console):
    """
    Presents the results of the 'snapshot' command.

    vibelint/cli.py
    """

    if result.success and result.output_path:
        console.print(
            f"[green]✓ Codebase snapshot created at {result.output_path}[/green]"
        )
    elif not result.success:
        console.print(
            f"[bold red]Error creating snapshot:[/bold red] {result.error_message}"
        )


def _display_collisions(
    hard_coll: List[NamespaceCollision],
    global_soft_coll: List[NamespaceCollision],
    local_soft_coll: List[NamespaceCollision],
    console: Console,
) -> int:
    """
    Displays collision results in tables and returns an exit code indicating if hard collisions were found.

    vibelint/cli.py
    """

    exit_code = 0
    total_collisions = len(hard_coll) + len(global_soft_coll) + len(local_soft_coll)

    if total_collisions == 0:
        return 0

    ctx = click.get_current_context(silent=True)
    project_root = (
        ctx.obj.project_root if ctx and hasattr(ctx.obj, "project_root") else Path(".")
    )

    def get_rel_path_display(p: Path) -> str:
        """
        Function 'get_rel_path_display'.

        vibelint/cli.py
        """

        try:
            return str(get_relative_path(p, project_root))
        except ValueError:
            return str(p)

    table = Table(title="Namespace Collision Results Summary")
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_row(
        "Hard Collisions", str(len(hard_coll)), style="red" if hard_coll else ""
    )
    table.add_row(
        "Global Soft Collisions (Definitions)",
        str(len(global_soft_coll)),
        style="yellow" if global_soft_coll else "",
    )
    table.add_row(
        "Local Soft Collisions (__all__)",
        str(len(local_soft_coll)),
        style="yellow" if local_soft_coll else "",
    )
    console.print(table)

    if hard_coll:
        console.print("\n[bold red]Hard Collisions:[/bold red]")
        console.print(
            "These collisions can break Python imports or indicate unexpected duplicates:"
        )
        grouped_hard = defaultdict(list)
        for c in hard_coll:
            grouped_hard[c.name].append(c)

        for name, collisions in sorted(grouped_hard.items()):
            locations = []
            for c in collisions:
                for i, p in enumerate(c.paths):
                    line_info = (
                        f":{c.linenos[i]}"
                        if c.linenos and i < len(c.linenos) and c.linenos[i]
                        else ""
                    )
                    locations.append(f"{get_rel_path_display(p)}{line_info}")
            unique_locations = sorted(list(set(locations)))
            is_intra_file = len(collisions[0].paths) > 1 and all(
                p == collisions[0].paths[0] for p in collisions[0].paths[1:]
            )
            if len(unique_locations) <= 2 and is_intra_file:
                console.print(
                    f"- '{name}': Duplicate definition/import in {', '.join(unique_locations)}"
                )
            else:
                console.print(
                    f"- '{name}': Conflicting definitions/imports in {', '.join(unique_locations)}"
                )
        exit_code = 1

    if local_soft_coll:
        console.print("\n[bold yellow]Local Soft Collisions (__all__):[/bold yellow]")
        console.print(
            "These names are exported via __all__ in multiple sibling modules:"
        )
        local_table = Table(show_header=True, header_style="bold yellow")
        local_table.add_column("Name", style="cyan", min_width=15)
        local_table.add_column("Exporting Files")
        grouped_local = defaultdict(list)
        for c in local_soft_coll:
            grouped_local[c.name].extend(
                p for p in c.paths if p not in grouped_local[c.name]
            )

        for name, involved_paths in sorted(grouped_local.items()):
            paths_str_list = sorted([get_rel_path_display(p) for p in involved_paths])
            local_table.add_row(name, "\n".join(paths_str_list))
        console.print(local_table)

    if global_soft_coll:
        console.print(
            "\n[bold yellow]Global Soft Collisions (Definitions):[/bold yellow]"
        )
        console.print("These names are defined in multiple modules (may confuse LLMs):")
        global_table = Table(show_header=True, header_style="bold yellow")
        global_table.add_column("Name", style="cyan", min_width=15)
        global_table.add_column("Defining Files")
        grouped_global = defaultdict(list)
        for c in global_soft_coll:
            grouped_global[c.name].extend(
                p for p in c.paths if p not in grouped_global[c.name]
            )

        for name, involved_paths in sorted(grouped_global.items()):
            paths_str_list = sorted([get_rel_path_display(p) for p in involved_paths])
            global_table.add_row(name, "\n".join(paths_str_list))
        console.print(global_table)

    return exit_code


@click.group()
@click.version_option()
@click.option("--debug", is_flag=True, help="Enable debug logging output.")
@click.pass_context
def cli(ctx: click.Context, debug: bool):
    """
    vibelint - Check, visualize, and create snapshots of Python codebases for LLM-friendliness.

    Run commands from the root of your project (where pyproject.toml or .git is located).

    vibelint/cli.py
    """

    ctx.ensure_object(VibelintContext)
    vibelint_ctx: VibelintContext = ctx.obj

    project_root = find_project_root(Path("."))
    if project_root is None:
        console.print("[bold red]Error:[/bold red] Could not find project root.")
        console.print("  vibelint must be run from within a directory that contains")
        console.print("  a 'pyproject.toml' file or a '.git' directory, or one of")
        console.print("  their subdirectories.")
        sys.exit(1)

    vibelint_ctx.project_root = project_root

    log_level = logging.DEBUG if debug else logging.INFO
    app_logger = logging.getLogger("vibelint")
    app_logger.setLevel(log_level)
    app_logger.propagate = False

    rich_handler = RichHandler(
        console=console,
        show_path=debug,
        markup=True,
        show_level=debug,
        rich_tracebacks=True,
    )
    formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    rich_handler.setFormatter(formatter)

    if not any(isinstance(h, RichHandler) for h in app_logger.handlers):
        app_logger.addHandler(rich_handler)

    logger_cli.debug(f"vibelint started. Debug mode: {'ON' if debug else 'OFF'}")
    logger_cli.debug(f"Identified project root: {project_root}")
    logger_cli.debug(f"Log level set to {logging.getLevelName(log_level)}")


@cli.command("check")
@click.option(
    "--yes", is_flag=True, help="Skip confirmation prompt for large directories."
)
@click.option(
    "-o",
    "--output-report",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Save a comprehensive Markdown report to the specified file.",
)
@click.pass_context
def check(ctx: click.Context, yes: bool, output_report: Optional[Path]):
    """
    Run lint checks and detect namespace collisions within the project.

    vibelint/cli.py
    """

    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root

    assert (
        project_root is not None
    ), "Project root must be set in context before calling check."

    logger_cli.debug(f"Running 'check' command (yes={yes}, report={output_report})")

    config: Config = load_config(project_root)
    if config.project_root is None:
        logger_cli.error("Project root became None after initial check. Aborting.")
        sys.exit(1)

    result_data = CheckResult()
    runner: Optional[LintRunner] = None

    try:
        if not config.project_root:
            raise ValueError(
                "Project root could not be definitively determined in config."
            )
        target_paths = [config.project_root]

        runner = LintRunner(config=config, skip_confirmation=yes)
        lint_exit_code = runner.run(target_paths)
        result_data.lint_results = runner.results
        vibelint_ctx.lint_runner = runner

        logger_cli.debug("Linting finished. Checking for namespace collisions...")
        result_data.hard_collisions = detect_hard_collisions(target_paths, config)
        result_data.global_soft_collisions = detect_global_definition_collisions(
            target_paths, config
        )
        result_data.local_soft_collisions = detect_local_export_collisions(
            target_paths, config
        )

        collision_exit_code = 1 if result_data.hard_collisions else 0

        report_failed = False
        if output_report:
            report_path = output_report.resolve()
            result_data.report_path = report_path
            logger_cli.info(f"Generating Markdown report to {report_path}...")
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                root_node_for_report, _ = build_namespace_tree(target_paths, config)
                with open(report_path, "w", encoding="utf-8") as f:
                    write_report_content(
                        f=f,
                        project_root=config.project_root,
                        target_paths=target_paths,
                        lint_results=result_data.lint_results,
                        hard_coll=result_data.hard_collisions,
                        soft_coll=result_data.global_soft_collisions
                        + result_data.local_soft_collisions,
                        root_node=root_node_for_report,
                        config=config,
                    )
                result_data.report_generated = True
                logger_cli.debug("Report generation successful.")
            except Exception as e:
                logger_cli.error(f"Error generating report: {e}", exc_info=True)
                result_data.report_error = str(e)
                report_failed = True

        final_exit_code = (
            lint_exit_code or collision_exit_code or (1 if report_failed else 0)
        )
        result_data.exit_code = final_exit_code
        result_data.success = final_exit_code == 0
        logger_cli.debug(f"Check command finished. Exit code: {final_exit_code}")

    except Exception as e:
        logger_cli.error(f"Critical error during 'check' execution: {e}", exc_info=True)
        result_data.success = False
        result_data.error_message = str(e)
        result_data.exit_code = 1

    vibelint_ctx.command_result = result_data

    if runner:
        _present_check_results(result_data, runner, console)
    else:
        console.print(
            "[bold red]Check command failed before linting could start.[/bold red]"
        )
        if result_data.error_message:
            console.print(f"[red]Error: {result_data.error_message}[/red]")

    sys.exit(result_data.exit_code)


@cli.command("namespace")
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Save the namespace tree visualization to the specified file.",
)
@click.pass_context
def namespace(ctx: click.Context, output: Optional[Path]):
    """
    Visualize the project's Python namespace structure as a tree.

    vibelint/cli.py
    """

    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root

    assert (
        project_root is not None
    ), "Project root must be set in context before calling namespace."

    logger_cli.debug(f"Running 'namespace' command (output={output})")

    config = load_config(project_root)
    if config.project_root is None:
        logger_cli.warning(
            "Project root missing from loaded config, forcing from context."
        )
        config._project_root = project_root

    result_data = NamespaceResult()

    try:
        if not config.project_root:
            raise ValueError(
                "Project root could not be definitively determined in config."
            )
        target_paths = [config.project_root]

        logger_cli.info("Building namespace tree...")
        root_node, intra_file_collisions = build_namespace_tree(target_paths, config)
        result_data.root_node = root_node
        result_data.intra_file_collisions = intra_file_collisions

        tree_str = str(root_node)

        if output:
            output_path = output.resolve()
            result_data.output_path = output_path
            logger_cli.info(f"Saving namespace tree to {output_path}...")
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(tree_str + "\n", encoding="utf-8")
                result_data.output_saved = True
            except Exception as e:
                logger_cli.error(f"Error saving namespace tree: {e}", exc_info=True)
                result_data.output_error = str(e)

        result_data.success = result_data.output_error is None
        result_data.exit_code = 0 if result_data.success else 1

    except Exception as e:
        logger_cli.error(f"Error building namespace tree: {e}", exc_info=True)
        result_data.success = False
        result_data.error_message = str(e)
        result_data.exit_code = 1

    vibelint_ctx.command_result = result_data
    _present_namespace_results(result_data, console)
    sys.exit(result_data.exit_code)


@cli.command("snapshot")
@click.option(
    "-o",
    "--output",
    default="codebase_snapshot.md",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output Markdown file name (default: codebase_snapshot.md)",
)
@click.pass_context
def snapshot(ctx: click.Context, output: Path):
    """
    Create a Markdown snapshot of the project files.

    vibelint/cli.py
    """

    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root

    assert (
        project_root is not None
    ), "Project root must be set in context before calling snapshot."

    logger_cli.debug(f"Running 'snapshot' command (output={output})")

    config = load_config(project_root)
    if config.project_root is None:
        logger_cli.warning(
            "Project root missing from loaded config, forcing from context."
        )
        config._project_root = project_root

    result_data = SnapshotResult()
    output_path = output.resolve()
    result_data.output_path = output_path

    try:
        if not config.project_root:
            raise ValueError(
                "Project root could not be definitively determined in config."
            )
        target_paths = [config.project_root]

        logger_cli.info(f"Creating codebase snapshot at {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        create_snapshot(
            output_path=output_path, target_paths=target_paths, config=config
        )
        result_data.success = True
        result_data.exit_code = 0

    except Exception as e:
        logger_cli.error(f"Error creating snapshot: {e}", exc_info=True)
        result_data.success = False
        result_data.error_message = str(e)
        result_data.exit_code = 1

    vibelint_ctx.command_result = result_data
    _present_snapshot_results(result_data, console)
    sys.exit(result_data.exit_code)


def main():
    """
    Main entry point for the vibelint CLI application.

    vibelint/cli.py
    """

    try:
        cli(obj=VibelintContext(), prog_name="vibelint")
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        logger_cli.error("Unhandled exception in CLI execution.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---
### File: src/vibelint/config.py

```python
"""
Configuration loading for vibelint.

Reads settings *only* from pyproject.toml under the [tool.vibelint] section.
No default values are assumed by this module. Callers must handle missing
configuration keys.

vibelint/config.py
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Mapping, List


if sys.version_info >= (3, 11):

    import tomllib
else:

    try:

        import tomli as tomllib
    except ImportError:

        print(
            "Error: vibelint requires Python 3.11+ or the 'tomli' package "
            "to parse pyproject.toml on Python 3.10."
            "\nHint: Try running: pip install tomli"
        )
        sys.exit(1)


from .utils import find_package_root

logger = logging.getLogger(__name__)


class Config:
    """
    Holds the vibelint configuration loaded *exclusively* from pyproject.toml.

    Provides access to the project root and the raw configuration dictionary.
    It does *not* provide default values for missing keys. Callers must
    check for the existence of required settings.

    Attributes:
    project_root: The detected root of the project containing pyproject.toml.
    Can be None if pyproject.toml is not found.
    settings: A read-only view of the dictionary loaded from the
    [tool.vibelint] section of pyproject.toml. Empty if the
    file or section is missing or invalid.

    vibelint/config.py
    """

    def __init__(self, project_root: Optional[Path], config_dict: Dict[str, Any]):
        """Initializes Config."""
        self._project_root = project_root
        self._config_dict = config_dict.copy()

    @property
    def project_root(self) -> Optional[Path]:
        """The detected project root directory, or None if not found."""
        return self._project_root

    @property
    def settings(self) -> Mapping[str, Any]:
        """Read-only view of the settings loaded from [tool.vibelint]."""
        return self._config_dict

    @property
    def ignore_codes(self) -> List[str]:
        """Returns the list of error codes to ignore, from config or empty list."""
        ignored = self.get("ignore", [])
        if isinstance(ignored, list) and all(isinstance(item, str) for item in ignored):
            return ignored
        elif ignored:
            logger.warning(
                "Configuration key 'ignore' in [tool.vibelint] is not a list of strings. Ignoring it."
            )
            return []
        else:
            return []

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a value from the loaded settings, returning default if not found."""
        return self._config_dict.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Gets a value, raising KeyError if the key is not found."""
        if key not in self._config_dict:
            raise KeyError(
                f"Required configuration key '{key}' not found in "
                f"[tool.vibelint] section of pyproject.toml."
            )
        return self._config_dict[key]

    def __contains__(self, key: str) -> bool:
        """Checks if a key exists in the loaded settings."""
        return key in self._config_dict

    def is_present(self) -> bool:
        """Checks if a project root was found and some settings were loaded."""
        return self._project_root is not None and bool(self._config_dict)


def load_config(start_path: Path) -> Config:
    """
    Loads vibelint configuration *only* from the nearest pyproject.toml file.

    Searches upwards from start_path. If pyproject.toml or the [tool.vibelint]
    section isn't found or is invalid, returns a Config object with project_root
    (if found) but an empty settings dictionary.

    Args:
    start_path: The directory to start searching upwards for pyproject.toml.

    Returns:
    A Config object. Check `config.project_root` and `config.settings`.

    vibelint/config.py
    """
    project_root = find_package_root(start_path)
    loaded_settings: Dict[str, Any] = {}

    if not project_root:
        logger.warning(
            f"Could not find project root (pyproject.toml) searching from '{start_path}'. "
            "No configuration will be loaded."
        )
        return Config(project_root=None, config_dict=loaded_settings)

    pyproject_path = project_root / "pyproject.toml"
    logger.debug(f"Found project root: {project_root}")
    logger.debug(f"Attempting to load config from: {pyproject_path}")

    try:
        with open(pyproject_path, "rb") as f:

            full_toml_config = tomllib.load(f)
        logger.debug("Parsed pyproject.toml")

        vibelint_config = full_toml_config.get("tool", {}).get("vibelint", {})

        if isinstance(vibelint_config, dict):
            loaded_settings = vibelint_config
            if loaded_settings:
                logger.info(f"Loaded [tool.vibelint] settings from {pyproject_path}")
                logger.debug(f"Loaded settings: {loaded_settings}")
            else:
                logger.info(
                    f"Found {pyproject_path}, but the [tool.vibelint] section is empty or missing."
                )
        else:
            logger.warning(
                f"[tool.vibelint] section in {pyproject_path} is not a valid table (dictionary). "
                "Ignoring this section."
            )

    except FileNotFoundError:

        logger.error(
            f"pyproject.toml not found at {pyproject_path} despite project root detection."
        )
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error parsing {pyproject_path}: {e}. Using empty configuration.")
    except OSError as e:
        logger.error(f"Error reading {pyproject_path}: {e}. Using empty configuration.")
    except Exception as e:

        logger.exception(f"Unexpected error loading config from {pyproject_path}: {e}")

    return Config(project_root=project_root, config_dict=loaded_settings)


__all__ = ["Config", "load_config"]
```

---
### File: src/vibelint/discovery.py

```python
"""
File discovery routines for vibelint.

Uses pathlib glob/rglob based on include patterns from pyproject.toml,
then filters results using exclude patterns.
Warns if essential configuration (like include_globs) is missing.
Warns if files inside common VCS directories are included due to missing excludes.

vibelint/discovery.py
"""

import fnmatch
import logging
from pathlib import Path
from typing import List, Set, Optional

from .config import Config
from .utils import get_relative_path

__all__ = ["discover_files"]
logger = logging.getLogger(__name__)


_VCS_DIRS = {".git", ".hg", ".svn"}


def _is_excluded(
    file_path_abs: Path,
    project_root: Path,
    exclude_globs: List[str],
    explicit_exclude_paths: Set[Path],
) -> bool:
    """
    Checks if a discovered file path should be excluded.

    Checks explicit paths first, then exclude globs.

    Args:
    file_path_abs: The absolute path of the file found by globbing.
    project_root: The absolute path of the project root.
    exclude_globs: List of glob patterns for exclusion from config.
    explicit_exclude_paths: Set of absolute paths to exclude explicitly.

    Returns:
    True if the file should be excluded, False otherwise.

    vibelint/discovery.py
    """

    if file_path_abs in explicit_exclude_paths:
        logger.debug(f"Excluding explicitly provided path: {file_path_abs}")
        return True

    try:

        rel_path_str = get_relative_path(file_path_abs, project_root)
    except ValueError:

        logger.warning(
            f"Path {file_path_abs} is outside project root {project_root}. Excluding."
        )
        return True

    for pattern in exclude_globs:

        normalized_pattern = pattern.replace("\\", "/")

        if fnmatch.fnmatch(str(rel_path_str), normalized_pattern):
            return True

    return False


def discover_files(
    paths: List[Path],
    config: Config,
    default_includes_if_missing: Optional[List[str]] = None,
    explicit_exclude_paths: Optional[Set[Path]] = None,
) -> List[Path]:
    """
    Discovers files using pathlib glob/rglob based on include patterns from
    pyproject.toml, then filters using exclude patterns.

    If `include_globs` is missing from the configuration:
    - If `default_includes_if_missing` is provided, uses those patterns and logs a warning.
    - Otherwise, logs an error and returns an empty list.

    Exclusions from `config.exclude_globs` are always applied. If missing,
    no exclusions based on globs are applied. Explicitly provided paths are
    also excluded.

    Warns if files within common VCS directories (.git, .hg, .svn) are found
    and not covered by exclude_globs.

    Args:
    paths: Initial paths (largely ignored, globs operate from project root).
    config: The vibelint configuration object (must have project_root set).
    default_includes_if_missing: Fallback include patterns if 'include_globs'
    is not in config.settings.
    explicit_exclude_paths: A set of absolute file paths to explicitly exclude
    from the results, regardless of other rules.

    Returns:
    A sorted list of unique absolute Path objects for the discovered files.

    Raises:
    ValueError: If config.project_root is None.

    vibelint/discovery.py
    """

    if config.project_root is None:
        raise ValueError(
            "Cannot discover files without a project root defined in Config."
        )

    project_root = config.project_root.resolve()
    candidate_files: Set[Path] = set()
    _explicit_excludes = explicit_exclude_paths or set()

    if "include_globs" in config.settings:
        include_globs_effective = config.settings["include_globs"]
        if not isinstance(include_globs_effective, list):
            logger.error(
                f"Configuration error: 'include_globs' in pyproject.toml must be a list. "
                f"Found type {type(include_globs_effective)}. No files will be included."
            )
            include_globs_effective = []
        elif not include_globs_effective:
            logger.warning(
                "Configuration: 'include_globs' is present but empty in pyproject.toml. "
                "No files will be included."
            )
    elif default_includes_if_missing is not None:
        logger.warning(
            "Configuration key 'include_globs' missing in [tool.vibelint] section "
            f"of pyproject.toml. Using default patterns: {default_includes_if_missing}"
        )
        include_globs_effective = default_includes_if_missing
    else:
        logger.error(
            "Configuration key 'include_globs' missing in [tool.vibelint] section "
            "of pyproject.toml. No include patterns specified. "
            "To include files, add 'include_globs = [\"**/*.py\"]' (or similar) "
            "to pyproject.toml."
        )
        return []

    normalized_includes = [p.replace("\\", "/") for p in include_globs_effective]

    exclude_globs = config.get("exclude_globs", [])
    if not isinstance(exclude_globs, list):
        logger.error(
            f"Configuration error: 'exclude_globs' in pyproject.toml must be a list. "
            f"Found type {type(exclude_globs)}. Ignoring exclusions."
        )
        exclude_globs = []
    normalized_exclude_globs = [p.replace("\\", "/") for p in exclude_globs]

    if _explicit_excludes:

        pass

    for pattern in normalized_includes:

        glob_method = project_root.rglob if "**" in pattern else project_root.glob
        try:
            matched_paths = glob_method(pattern)
            count = 0
            for p in matched_paths:
                if p.is_file():
                    candidate_files.add(p.resolve())
                    count += 1

        except Exception as e:

            pass

    vcs_warnings: Set[Path] = set()
    if candidate_files:

        for file_path in candidate_files:
            is_in_vcs_dir = any(part in _VCS_DIRS for part in file_path.parts)
            if is_in_vcs_dir:
                try:
                    rel_path_str_check = get_relative_path(file_path, project_root)
                    covered_by_exclude = False
                    for pattern in normalized_exclude_globs:
                        if fnmatch.fnmatch(str(rel_path_str_check), pattern):
                            covered_by_exclude = True
                            break
                    if not covered_by_exclude and file_path not in _explicit_excludes:
                        vcs_warnings.add(file_path)

                except ValueError:
                    pass

    discovered_files: Set[Path] = set()
    logger.debug(f"Applying exclude rules to {len(candidate_files)} candidates...")
    for file_path in candidate_files:
        if not _is_excluded(
            file_path, project_root, normalized_exclude_globs, _explicit_excludes
        ):
            discovered_files.add(file_path)
        else:
            try:
                rel_path_log = get_relative_path(file_path, project_root)
                if file_path in _explicit_excludes:
                    logger.debug(
                        f"Excluding candidate file (explicitly): {rel_path_log}"
                    )
                else:
                    matched_glob = "unknown glob"
                    for pattern in normalized_exclude_globs:
                        if fnmatch.fnmatch(str(rel_path_log), pattern):
                            matched_glob = pattern
                            break

            except ValueError:

                pass

    final_count = len(discovered_files)

    final_vcs_warnings = vcs_warnings.intersection(discovered_files)
    if final_vcs_warnings:
        logger.warning(
            f"Found {len(final_vcs_warnings)} files within potential VCS directories "
            f"({', '.join(_VCS_DIRS)}) that were included because they were not "
            f"matched by any 'exclude_globs' pattern in pyproject.toml:"
        )
        paths_to_log = sorted(
            [get_relative_path(p, project_root) for p in final_vcs_warnings]
        )[:5]
        for rel_path_warn in paths_to_log:
            logger.warning(f"  - {rel_path_warn}")
        if len(final_vcs_warnings) > 5:
            logger.warning(f"  - ... and {len(final_vcs_warnings) - 5} more.")
        logger.warning(
            "Consider adding patterns like '.git/**' to 'exclude_globs' "
            "in your [tool.vibelint] section if this was unintended."
        )

    if final_count == 0 and len(candidate_files) > 0:
        logger.warning(
            "All candidate files were excluded. Check your exclude_globs patterns."
        )
    elif final_count == 0 and not include_globs_effective:
        pass
    elif final_count == 0:
        logger.warning("No files found matching include_globs patterns.")

    return sorted(list(discovered_files))
```

---
### File: src/vibelint/error_codes.py

```python
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
VBL103 = "VBL103"


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
    VBL103: "Docstring has potential formatting or indentation issues.",
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
    "VBL103",
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
```

---
### File: src/vibelint/lint.py

```python
"""
Core linting runner for vibelint.

vibelint/lint.py
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import traceback


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

from .config import Config


from .validators.shebang import validate_shebang, file_contains_top_level_main_block
from .validators.encoding import validate_encoding_cookie
from .validators.docstring import validate_every_docstring
from .validators.exports import validate_exports
from .discovery import discover_files
from .utils import get_relative_path


from .error_codes import VBL901, VBL902, VBL903, VBL904, VBL905

__all__ = ["LintResult", "LintRunner"]

console = Console()
logger = logging.getLogger(__name__)

ValidationIssue = Tuple[str, str]


class LintResult:
    """
    Stores the result of a lint operation on a single file.

    vibelint/lint.py
    """

    def __init__(self) -> None:
        """Initializes a LintResult instance."""
        self.file_path: Path = Path()
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []

    @property
    def has_issues(self) -> bool:
        """Returns True if there are any errors or warnings."""
        return bool(self.errors or self.warnings)


class LintRunner:
    """
    Runner for linting operations. No longer supports fixing.

    vibelint/lint.py
    """

    def __init__(self, config: Config, skip_confirmation: bool = False) -> None:
        """
        Initializes the LintRunner.

        Args:
            config: The vibelint configuration object.
            skip_confirmation: If True, bypass the confirmation prompt for large directories.

        vibelint/lint.py
        """
        self.config = config

        self.skip_confirmation = skip_confirmation
        self.results: List[LintResult] = []
        self._final_exit_code: int = 0

    def run(self, paths: List[Path]) -> int:
        """
        Runs the linting process, returns the exit code.

        vibelint/lint.py
        """
        logger.debug("LintRunner.run: Starting file discovery...")
        if not self.config.project_root:
            logger.error("Project root not found in config. Cannot run.")
            return 1

        ignore_codes_set = set(self.config.ignore_codes)
        if ignore_codes_set:
            logger.info(f"Ignoring codes: {sorted(list(ignore_codes_set))}")

        all_discovered_files: List[Path] = discover_files(
            paths=paths, config=self.config, explicit_exclude_paths=set()
        )
        python_files: List[Path] = [
            f for f in all_discovered_files if f.is_file() and f.suffix == ".py"
        ]
        logger.debug(f"LintRunner.run: Discovered {len(python_files)} Python files.")

        if not python_files:
            logger.info("No Python files found matching includes/excludes.")
            return 0

        large_dir_threshold = self.config.get("large_dir_threshold", 500)
        if len(python_files) > large_dir_threshold and not self.skip_confirmation:
            logger.debug(
                f"File count {len(python_files)} > threshold {large_dir_threshold}. Requesting confirmation."
            )
            if not self._confirm_large_directory(len(python_files)):
                logger.info("User aborted due to large file count.")
                return 1

        MAX_WORKERS = self.config.get("max_workers")
        logger.debug(f"LintRunner.run: Processing files with max_workers={MAX_WORKERS}")
        progress_console = Console(stderr=True)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=progress_console,
            transient=True,
        ) as progress:

            task_desc = f"Checking {len(python_files)} Python files..."
            task_id = progress.add_task(task_desc, total=len(python_files))
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

                futures = {
                    executor.submit(self._process_file, f, ignore_codes_set): f
                    for f in python_files
                }
                temp_results = []
                for future in futures:
                    file_proc = futures[future]
                    try:
                        res = future.result()
                        temp_results.append(res)
                    except Exception as exc:
                        rel_path_log_err = file_proc.name
                        try:
                            rel_path_log_err = (
                                str(
                                    get_relative_path(
                                        file_proc, self.config.project_root
                                    )
                                )
                                if self.config.project_root
                                else file_proc.name
                            )
                        except ValueError:
                            rel_path_log_err = str(file_proc.resolve())
                        logger.error(
                            f"Exception processing {rel_path_log_err}: {exc}",
                            exc_info=True,
                        )
                        lr_err = LintResult()
                        lr_err.file_path = file_proc

                        lr_err.errors.append(
                            (VBL904, f"Processing thread error: {exc}")
                        )
                        temp_results.append(lr_err)
                    finally:
                        progress.update(task_id, advance=1)

                self.results = sorted(temp_results, key=lambda r: r.file_path)

        files_with_errors = sum(1 for r in self.results if r.errors)
        self._final_exit_code = 1 if files_with_errors > 0 else 0
        logger.debug(
            f"LintRunner.run: Finished processing. Final exit code: {self._final_exit_code}"
        )
        return self._final_exit_code

    def _process_file(self, file_path: Path, ignore_codes_set: set[str]) -> LintResult:
        """
        Processes a single file for linting issues, filtering by ignored codes.

        vibelint/lint.py
        """
        lr = LintResult()
        lr.file_path = file_path
        relative_path_str = file_path.name
        log_prefix = f"[{file_path.name}]"
        original_content: Optional[str] = None
        collected_errors: List[ValidationIssue] = []
        collected_warnings: List[ValidationIssue] = []

        try:

            if self.config.project_root:
                try:
                    relative_path = get_relative_path(
                        file_path, self.config.project_root
                    )
                    relative_path_str = str(relative_path).replace("\\", "/")
                    log_prefix = f"[{relative_path_str}]"
                except ValueError:
                    relative_path_str = str(file_path.resolve())
            else:
                relative_path_str = str(file_path.resolve())

            logger.debug(f"{log_prefix} --- Starting validation ---")

            try:
                original_content = file_path.read_text(encoding="utf-8")
                logger.debug(f"{log_prefix} Read {len(original_content)} bytes.")
            except Exception as read_e:
                logger.error(
                    f"{log_prefix} Error reading file: {read_e}", exc_info=True
                )

                lr.errors.append((VBL901, f"Error reading file: {read_e}"))
                return lr

            try:

                doc_res, _ = validate_every_docstring(
                    original_content, relative_path_str
                )
                if doc_res:
                    collected_errors.extend(doc_res.errors)
                    collected_warnings.extend(doc_res.warnings)

                allowed_sb: List[str] = self.config.get(
                    "allowed_shebangs", ["#!/usr/bin/env python3"]
                )
                is_script = file_contains_top_level_main_block(
                    file_path, original_content
                )
                sb_res = validate_shebang(original_content, is_script, allowed_sb)
                collected_errors.extend(sb_res.errors)
                collected_warnings.extend(sb_res.warnings)

                enc_res = validate_encoding_cookie(original_content)
                collected_errors.extend(enc_res.errors)
                collected_warnings.extend(enc_res.warnings)

                export_res = validate_exports(
                    original_content, relative_path_str, self.config
                )
                collected_errors.extend(export_res.errors)
                collected_warnings.extend(export_res.warnings)

                logger.debug(
                    f"{log_prefix} Validation Complete. Found E={len(collected_errors)}, W={len(collected_warnings)} (before filtering)"
                )

            except SyntaxError as se:
                line = f"line {se.lineno}" if se.lineno else "unk line"
                col = f", col {se.offset}" if se.offset is not None else ""
                err_msg = f"SyntaxError parsing file: {se.msg} ({relative_path_str}, {line}{col})"
                logger.error(f"{log_prefix} {err_msg}")

                collected_errors.append((VBL902, err_msg))

            except Exception as val_e:
                logger.error(
                    f"{log_prefix} Error during validation phase: {val_e}",
                    exc_info=True,
                )

                collected_errors.append((VBL903, f"Internal validation error: {val_e}"))

            lr.errors = [
                (code, msg)
                for code, msg in collected_errors
                if code not in ignore_codes_set
            ]
            lr.warnings = [
                (code, msg)
                for code, msg in collected_warnings
                if code not in ignore_codes_set
            ]

            if len(collected_errors) != len(lr.errors) or len(
                collected_warnings
            ) != len(lr.warnings):
                logger.debug(
                    f"{log_prefix} Filtered issues based on ignore config. Final E={len(lr.errors)}, W={len(lr.warnings)}"
                )

        except Exception as e:
            logger.error(
                f"{log_prefix} Critical unhandled error in _process_file: {e}\n{traceback.format_exc()}"
            )

            lr.errors.append((VBL905, f"Critical processing error: {e}"))

        logger.debug(f"{log_prefix} --- Finished validation ---")
        return lr

    def _confirm_large_directory(self, file_count: int) -> bool:
        """
        Asks user for confirmation if many files are found.

        vibelint/lint.py
        """

        prompt_console = Console(stderr=True)
        prompt_console.print(
            f"[yellow]WARNING:[/yellow] Found {file_count} Python files. This might take time."
        )
        try:
            return click.confirm("Proceed?", default=False, err=True)
        except click.Abort:
            prompt_console.print("[yellow]Aborted.[/yellow]")
            return False
        except RuntimeError as e:
            if "Cannot prompt" in str(e) or "tty" in str(e).lower():
                prompt_console.print(
                    "[yellow]Non-interactive session detected. Use --yes to bypass confirmation. Aborting.[/yellow]"
                )
            else:
                logger.error(f"RuntimeError during confirm: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Error during confirm: {e}", exc_info=True)
            return False

    def _print_summary(self) -> None:
        """
        Prints a summary table of the linting results.

        vibelint/lint.py
        """
        summary_console = Console()
        table = Table(title="vibelint Results Summary")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="magenta")
        total = len(self.results)

        errors = sum(1 for r in self.results if r.errors)
        warns = sum(1 for r in self.results if r.warnings and not r.errors)
        ok = total - errors - warns

        table.add_row("Files Scanned", str(total))
        table.add_row(
            "Files OK", str(ok), style="green" if ok == total and total > 0 else ""
        )
        table.add_row("Files with Errors", str(errors), style="red" if errors else "")
        table.add_row(
            "Files with Warnings only",
            str(warns),
            style="yellow" if warns else "",
        )

        summary_console.print(table)
```

---
### File: src/vibelint/namespace.py

```python
"""
Namespace representation & collision detection for Python code.

vibelint/namespace.py
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .config import Config
from .utils import get_relative_path, find_project_root
from .discovery import discover_files


__all__ = [
    "CollisionType",
    "NamespaceCollision",
    "NamespaceNode",
    "detect_hard_collisions",
    "detect_global_definition_collisions",
    "detect_local_export_collisions",
    "build_namespace_tree",
    "get_namespace_collisions_str",
]

logger = logging.getLogger(__name__)


class CollisionType:
    """
    Enum-like class for collision types.

    vibelint/namespace.py
    """

    HARD = "hard"
    LOCAL_SOFT = "local_soft"
    GLOBAL_SOFT = "global_soft"


class NamespaceCollision:
    """
    Represents a collision between two or more same-named entities.

    vibelint/namespace.py
    """

    def __init__(
        self,
        name: str,
        collision_type: str,
        paths: List[Path],
        linenos: Optional[List[Optional[int]]] = None,
    ) -> None:
        """
        Initializes a NamespaceCollision instance.

        Args:
        name: The name of the colliding entity.
        collision_type: The type of collision (HARD, LOCAL_SOFT, GLOBAL_SOFT).
        paths: A list of Path objects for all files involved in the collision.
        linenos: An optional list of line numbers corresponding to each path.

        vibelint/namespace.py
        """

        if not paths:
            raise ValueError("At least one path must be provided for a collision.")

        self.name = name
        self.collision_type = collision_type

        self.paths = sorted(list(set(paths)), key=str)

        self.linenos = (
            linenos
            if linenos and len(linenos) == len(self.paths)
            else [None] * len(self.paths)
        )

        self.path1: Path = self.paths[0]
        self.path2: Path = self.paths[1] if len(self.paths) > 1 else self.paths[0]
        self.lineno1: Optional[int] = self.linenos[0] if self.linenos else None
        self.lineno2: Optional[int] = (
            self.linenos[1] if len(self.linenos) > 1 else self.lineno1
        )

        self.definition_paths: List[Path] = (
            self.paths
            if self.collision_type
            in [CollisionType.GLOBAL_SOFT, CollisionType.LOCAL_SOFT]
            else []
        )

    def __repr__(self) -> str:
        """
        Provides a detailed string representation for debugging.

        vibelint/namespace.py
        """

        return (
            f"NamespaceCollision(name='{self.name}', type='{self.collision_type}', "
            f"paths={self.paths}, linenos={self.linenos})"
        )

    def __str__(self) -> str:
        """
        Provides a user-friendly string representation of the collision.

        vibelint/namespace.py
        """

        proj_root = find_project_root(Path(".").resolve())
        base_path = proj_root if proj_root else Path(".")

        paths_str_list = []
        for i, p in enumerate(self.paths):
            loc = (
                f":{self.linenos[i]}"
                if self.linenos and self.linenos[i] is not None
                else ""
            )
            try:
                paths_str_list.append(f"{get_relative_path(p, base_path)}{loc}")
            except ValueError:
                paths_str_list.append(f"{p}{loc}")
        paths_str = ", ".join(paths_str_list)

        if self.collision_type == CollisionType.HARD:
            if len(self.paths) == 2 and self.paths[0] == self.paths[1]:

                line_info = ""
                if self.lineno1 is not None and self.lineno2 is not None:
                    line_info = f" (lines ~{self.lineno1} and ~{self.lineno2})"
                elif self.lineno1 is not None:
                    line_info = f" (line ~{self.lineno1})"

                return (
                    f"{self.collision_type.upper()} Collision: Duplicate definition/import of '{self.name}' in "
                    f"{paths_str_list[0]}{line_info}"
                )
            else:
                return f"{self.collision_type.upper()} Collision: Name '{self.name}' used by conflicting entities in: {paths_str}"
        elif self.collision_type == CollisionType.LOCAL_SOFT:
            return f"{self.collision_type.upper()} Collision: '{self.name}' exported via __all__ in multiple sibling modules: {paths_str}"
        elif self.collision_type == CollisionType.GLOBAL_SOFT:
            return f"{self.collision_type.upper()} Collision: '{self.name}' defined in multiple modules: {paths_str}"
        else:
            return f"Unknown Collision: '{self.name}' involving paths: {paths_str}"


def detect_hard_collisions(
    paths: List[Path],
    config: Config,
) -> List[NamespaceCollision]:
    """
    Detect HARD collisions: member vs. submodule, or duplicate definitions within a file.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns:
    A list of detected HARD NamespaceCollision objects.

    vibelint/namespace.py
    """

    root_node, intra_file_collisions = build_namespace_tree(paths, config)

    inter_file_collisions = root_node.get_hard_collisions()

    all_collisions = intra_file_collisions + inter_file_collisions
    for c in all_collisions:
        c.collision_type = CollisionType.HARD
    return all_collisions


def detect_global_definition_collisions(
    paths: List[Path],
    config: Config,
) -> List[NamespaceCollision]:
    """
    Detect GLOBAL SOFT collisions: the same name defined/assigned at the top level
    in multiple different modules across the project.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns:
    A list of detected GLOBAL_SOFT NamespaceCollision objects.

    vibelint/namespace.py
    """

    root_node, _ = build_namespace_tree(paths, config)

    definition_collisions = root_node.detect_global_definition_collisions()

    return definition_collisions


def detect_local_export_collisions(
    paths: List[Path],
    config: Config,
) -> List[NamespaceCollision]:
    """
    Detect LOCAL SOFT collisions: the same name exported via __all__ by multiple
    sibling modules within the same package.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns:
    A list of detected LOCAL_SOFT NamespaceCollision objects.

    vibelint/namespace.py
    """

    root_node, _ = build_namespace_tree(paths, config)
    collisions: List[NamespaceCollision] = []
    root_node.find_local_export_collisions(collisions)
    return collisions


def get_namespace_collisions_str(
    paths: List[Path],
    config: Config,
    console=None,
) -> str:
    """
    Return a string representation of all collision types for quick debugging.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.
    console: Optional console object (unused).

    Returns:
    A string summarizing all detected collisions.

    vibelint/namespace.py
    """

    from io import StringIO

    buf = StringIO()

    hard_collisions = detect_hard_collisions(paths, config)
    global_soft_collisions = detect_global_definition_collisions(paths, config)
    local_soft_collisions = detect_local_export_collisions(paths, config)

    proj_root = find_project_root(Path(".").resolve())
    base_path = proj_root if proj_root else Path(".")

    if hard_collisions:
        buf.write("Hard Collisions:\n")
        for c in sorted(hard_collisions, key=lambda x: (x.name, str(x.paths[0]))):
            buf.write(f"- {str(c)}\n")

    if local_soft_collisions:
        buf.write("\nLocal Soft Collisions (__all__):\n")

        grouped = defaultdict(list)
        for c in local_soft_collisions:
            grouped[c.name].extend(c.paths)
        for name, involved_paths in sorted(grouped.items()):
            try:
                paths_str = ", ".join(
                    sorted(
                        str(get_relative_path(p, base_path))
                        for p in set(involved_paths)
                    )
                )
            except ValueError:
                paths_str = ", ".join(sorted(str(p) for p in set(involved_paths)))
            buf.write(f"- '{name}': exported by {paths_str}\n")

    if global_soft_collisions:
        buf.write("\nGlobal Soft Collisions (Definitions):\n")

        grouped = defaultdict(list)
        for c in global_soft_collisions:
            grouped[c.name].extend(c.paths)
        for name, involved_paths in sorted(grouped.items()):
            try:
                paths_str = ", ".join(
                    sorted(
                        str(get_relative_path(p, base_path))
                        for p in set(involved_paths)
                    )
                )
            except ValueError:
                paths_str = ", ".join(sorted(str(p) for p in set(involved_paths)))
            buf.write(f"- '{name}': defined in {paths_str}\n")

    return buf.getvalue()


class NamespaceNode:
    """
    A node in the "module" hierarchy (like package/subpackage, or file-level).
    Holds child nodes and top-level members (functions/classes).

    vibelint/namespace.py
    """

    def __init__(
        self, name: str, path: Optional[Path] = None, is_package: bool = False
    ) -> None:
        """
        Initializes a NamespaceNode.

        Args:
        name: The name of the node (e.g., module name, package name).
        path: The filesystem path associated with this node (optional).
        is_package: True if this node represents a package (directory).

        vibelint/namespace.py
        """

        self.name = name
        self.path = path
        self.is_package = is_package
        self.children: Dict[str, "NamespaceNode"] = {}

        self.members: Dict[str, Tuple[Path, Optional[int]]] = {}

        self.member_collisions: List[NamespaceCollision] = []

        self.exported_names: Optional[List[str]] = None

    def set_exported_names(self, names: List[str]):
        """
        Sets the list of names found in __all__.

        vibelint/namespace.py
        """

        self.exported_names = names

    def add_child(
        self, name: str, path: Path, is_package: bool = False
    ) -> "NamespaceNode":
        """
        Adds a child node, creating if necessary.

        vibelint/namespace.py
        """

        if name not in self.children:
            self.children[name] = NamespaceNode(name, path, is_package)

        elif path:

            if not (self.children[name].is_package and not is_package):
                self.children[name].path = path
            self.children[name].is_package = (
                is_package or self.children[name].is_package
            )
        return self.children[name]

    def get_hard_collisions(self) -> List[NamespaceCollision]:
        """
        Detect HARD collisions recursively: members vs. child modules.

        vibelint/namespace.py
        """

        collisions: List[NamespaceCollision] = []

        member_names_with_info = {}
        if self.is_package and self.path:
            init_path = (self.path / "__init__.py").resolve()
            member_names_with_info = {
                name: (def_path, lineno)
                for name, (def_path, lineno) in self.members.items()
                if def_path.resolve() == init_path
            }

        child_names = set(self.children.keys())
        common_names = set(member_names_with_info.keys()).intersection(child_names)

        for name in common_names:

            member_def_path, member_lineno = member_names_with_info.get(
                name, (None, None)
            )
            cnode = self.children[name]
            child_path = cnode.path

            if member_def_path and child_path:

                collisions.append(
                    NamespaceCollision(
                        name=name,
                        collision_type=CollisionType.HARD,
                        paths=[member_def_path, child_path],
                        linenos=[member_lineno, None],
                    )
                )

        for cnode in self.children.values():
            collisions.extend(cnode.get_hard_collisions())
        return collisions

    def collect_defined_members(
        self, all_dict: Dict[str, List[Tuple[Path, Optional[int]]]]
    ):
        """
        Recursively collects defined members (path, lineno) for global definition collision check.

        vibelint/namespace.py
        """

        if self.path and self.members:

            for mname, (mpath, mlineno) in self.members.items():
                all_dict.setdefault(mname, []).append((mpath, mlineno))

        for cnode in self.children.values():
            cnode.collect_defined_members(all_dict)

    def detect_global_definition_collisions(self) -> List[NamespaceCollision]:
        """
        Detects GLOBAL SOFT collisions across the whole tree starting from this node.

        vibelint/namespace.py
        """

        all_defined_members: Dict[str, List[Tuple[Path, Optional[int]]]] = defaultdict(
            list
        )
        self.collect_defined_members(all_defined_members)

        collisions: List[NamespaceCollision] = []
        for name, path_lineno_list in all_defined_members.items():

            unique_paths_map: Dict[Path, Optional[int]] = {}
            for path, lineno in path_lineno_list:
                resolved_p = path.resolve()

                if resolved_p not in unique_paths_map:
                    unique_paths_map[resolved_p] = lineno

            if len(unique_paths_map) > 1:

                sorted_paths = sorted(unique_paths_map.keys(), key=str)

                sorted_linenos = [unique_paths_map[p] for p in sorted_paths]

                collisions.append(
                    NamespaceCollision(
                        name=name,
                        collision_type=CollisionType.GLOBAL_SOFT,
                        paths=sorted_paths,
                        linenos=sorted_linenos,
                    )
                )
        return collisions

    def find_local_export_collisions(self, collisions_list: List[NamespaceCollision]):
        """
        Recursively finds LOCAL SOFT collisions (__all__) within packages.

        Args:
        collisions_list: A list to append found collisions to.

        vibelint/namespace.py
        """

        if self.is_package:
            exports_in_package: Dict[str, List[Path]] = defaultdict(list)

            if self.path and self.path.is_dir() and self.exported_names:

                init_path = (self.path / "__init__.py").resolve()

                if init_path.exists() and any(
                    p.resolve() == init_path for p, _ in self.members.values()
                ):
                    for name in self.exported_names:
                        exports_in_package[name].append(init_path)

            for child in self.children.values():

                if (
                    child.path
                    and child.path.is_file()
                    and not child.is_package
                    and child.name != "__init__"
                    and child.exported_names
                ):
                    for name in child.exported_names:
                        exports_in_package[name].append(child.path.resolve())

            for name, paths in exports_in_package.items():
                unique_paths = sorted(list(set(paths)), key=str)
                if len(unique_paths) > 1:
                    collisions_list.append(
                        NamespaceCollision(
                            name=name,
                            collision_type=CollisionType.LOCAL_SOFT,
                            paths=unique_paths,
                            linenos=[None for _ in unique_paths],
                        )
                    )

        for child in self.children.values():
            if child.is_package:
                child.find_local_export_collisions(collisions_list)

    def __str__(self) -> str:
        """
        Provides a string representation of the node and its subtree, including members.
        Uses a revised formatting approach for better clarity relative to project root.

        vibelint/namespace.py
        """

        lines = []

        proj_root = find_project_root(Path(".").resolve())
        base_path_for_display = proj_root if proj_root else Path(".")

        def build_tree_lines(
            node: "NamespaceNode", prefix: str = "", base: Path = Path(".")
        ):
            """
            Docstring for function 'build_tree_lines'.

            vibelint/namespace.py
            """

            child_items = sorted(node.children.items())

            direct_members = []
            if node.path and node.members:

                expected_def_path = None
                node_path_resolved = node.path.resolve()
                if node.is_package and node_path_resolved.is_dir():
                    expected_def_path = (node_path_resolved / "__init__.py").resolve()
                elif node_path_resolved.is_file():
                    expected_def_path = node_path_resolved

                if expected_def_path:
                    direct_members = sorted(
                        [
                            name
                            for name, (def_path, _) in node.members.items()
                            if def_path.resolve() == expected_def_path
                        ]
                    )

            all_items = child_items + [(name, "member") for name in direct_members]
            total_items = len(all_items)

            for i, (name, item) in enumerate(all_items):
                is_last = i == total_items - 1
                connector = "└── " if is_last else "├── "
                next_level_prefix = prefix + ("    " if is_last else "│   ")

                if item == "member":

                    lines.append(f"{prefix}{connector}{name} (member)")
                else:

                    child: "NamespaceNode" = item
                    child_path_str = ""
                    indicator = ""
                    if child.path:
                        try:
                            rel_p = get_relative_path(child.path, base)

                            if child.is_package:
                                indicator = " (P)"
                            elif child.name == "__init__":
                                indicator = " (I)"
                            else:
                                indicator = " (M)"
                            child_path_str = f"  [{rel_p}{indicator}]"
                        except ValueError:
                            indicator = (
                                " (P)"
                                if child.is_package
                                else (" (I)" if child.name == "__init__" else " (M)")
                            )
                            child_path_str = f"  [{child.path.resolve()}{indicator}]"
                    else:
                        child_path_str = "  [No Path]"

                    lines.append(f"{prefix}{connector}{name}{child_path_str}")

                    if child.children or (
                        child.members
                        and any(
                            m_path.resolve()
                            == (child.path.resolve() if child.path else None)
                            for m, (m_path, _) in child.members.items()
                        )
                    ):
                        build_tree_lines(child, next_level_prefix, base)

        root_path_str = ""
        root_indicator = ""

        if self.path:
            root_path_resolved = self.path.resolve()
            try:

                rel_p = get_relative_path(
                    root_path_resolved, base_path_for_display.parent
                )

                if rel_p == Path("."):
                    rel_p = Path(self.name)

                root_indicator = (
                    " (P)"
                    if self.is_package
                    else (" (M)" if root_path_resolved.is_file() else "")
                )
                root_path_str = f"  [{rel_p}{root_indicator}]"
            except ValueError:
                root_indicator = (
                    " (P)"
                    if self.is_package
                    else (" (M)" if root_path_resolved.is_file() else "")
                )
                root_path_str = f"  [{root_path_resolved}{root_indicator}]"
        else:
            root_path_str = "  [No Path]"

        lines.append(f"{self.name}{root_path_str}")
        build_tree_lines(self, prefix="", base=base_path_for_display)
        return "\n".join(lines)


def _extract_module_members(
    file_path: Path,
) -> Tuple[
    Dict[str, Tuple[Path, Optional[int]]], List[NamespaceCollision], Optional[List[str]]
]:
    """
    Parses a Python file and extracts top-level member definitions/assignments,
    intra-file hard collisions, and the contents of __all__ if present.

    Returns:
    - A dictionary mapping defined/assigned names to a tuple of (file path, line number).
    - A list of intra-file hard collisions (NamespaceCollision objects).
    - A list of names in __all__, or None if __all__ is not found or invalid.

    vibelint/namespace.py
    """

    try:
        source = file_path.read_text(encoding="utf-8")

        tree = ast.parse(source, filename=str(file_path))
    except Exception as e:
        logger.warning(f"Could not parse file {file_path} for namespace analysis: {e}")

        return {}, [], None

    defined_members_map: Dict[str, Tuple[Path, Optional[int]]] = {}
    collisions: List[NamespaceCollision] = []
    exported_names: Optional[List[str]] = None

    defined_names_nodes: Dict[str, ast.AST] = {}

    for node in tree.body:
        current_node = node
        name: Optional[str] = None
        is_definition = False
        is_all_assignment = False
        lineno = getattr(current_node, "lineno", None)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            is_definition = True
        elif isinstance(node, ast.Assign):

            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
            ):
                is_all_assignment = True

                if isinstance(node.value, (ast.List, ast.Tuple)):
                    exported_names = []
                    for elt in node.value.elts:

                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exported_names.append(elt.value)

                if "__all__" not in defined_names_nodes:
                    defined_names_nodes["__all__"] = current_node
                else:
                    first_node = defined_names_nodes["__all__"]
                    collisions.append(
                        NamespaceCollision(
                            name="__all__",
                            collision_type=CollisionType.HARD,
                            paths=[file_path, file_path],
                            linenos=[getattr(first_node, "lineno", None), lineno],
                        )
                    )

            else:

                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        is_definition = True

                        if name:
                            if name in defined_names_nodes:

                                first_node = defined_names_nodes[name]
                                collisions.append(
                                    NamespaceCollision(
                                        name=name,
                                        collision_type=CollisionType.HARD,
                                        paths=[file_path, file_path],
                                        linenos=[
                                            getattr(first_node, "lineno", None),
                                            lineno,
                                        ],
                                    )
                                )
                            else:

                                defined_names_nodes[name] = current_node
                                defined_members_map[name] = (
                                    file_path,
                                    lineno,
                                )
                            name = None

        if name and is_definition and not is_all_assignment:
            if name in defined_names_nodes:

                first_node = defined_names_nodes[name]
                collisions.append(
                    NamespaceCollision(
                        name=name,
                        collision_type=CollisionType.HARD,
                        paths=[file_path, file_path],
                        linenos=[getattr(first_node, "lineno", None), lineno],
                    )
                )
            else:

                defined_names_nodes[name] = current_node
                defined_members_map[name] = (file_path, lineno)

    return defined_members_map, collisions, exported_names


def build_namespace_tree(
    paths: List[Path], config: Config
) -> Tuple[NamespaceNode, List[NamespaceCollision]]:
    """
    Builds the namespace tree, collects intra-file collisions, and stores members/__all__.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns a tuple: (root_node, all_intra_file_collisions)

    vibelint/namespace.py
    """

    project_root_found = config.project_root or find_project_root(
        paths[0].resolve() if paths else Path(".")
    )
    if not project_root_found:

        project_root_found = Path(".")
        root_node_name = "root"
        logger.warning(
            "Could not determine project root. Using '.' as root for namespace analysis."
        )
    else:
        root_node_name = project_root_found.name

    root = NamespaceNode(
        root_node_name, path=project_root_found.resolve(), is_package=True
    )
    root_path_for_rel = project_root_found.resolve()
    all_intra_file_collisions: List[NamespaceCollision] = []

    python_files = [
        f
        for f in discover_files(
            paths,
            config,
        )
        if f.suffix == ".py"
    ]

    if not python_files:
        logger.info(
            "No Python files found for namespace analysis based on configuration."
        )
        return root, all_intra_file_collisions

    for f in python_files:
        try:

            rel_path = f.relative_to(root_path_for_rel)
            rel_parts = list(rel_path.parts)
        except ValueError:

            rel_parts = [f.name]
            logger.warning(
                f"File {f} is outside the determined project root {root_path_for_rel}. Adding directly under root."
            )

        current = root

        for i, part in enumerate(rel_parts[:-1]):

            dir_path = root_path_for_rel.joinpath(*rel_parts[: i + 1])
            current = current.add_child(part, dir_path, is_package=True)

        file_name = rel_parts[-1]
        mod_name = Path(file_name).stem
        file_abs_path = f

        members, intra_collisions, exported_names = _extract_module_members(
            file_abs_path
        )
        all_intra_file_collisions.extend(intra_collisions)

        if mod_name == "__init__":

            package_node = current
            package_node.is_package = True
            package_node.path = file_abs_path.parent

            for m_name, m_info in members.items():
                if m_name not in package_node.members:
                    package_node.members[m_name] = m_info

            if exported_names is not None:
                package_node.set_exported_names(exported_names)

        else:

            module_node = current.add_child(mod_name, file_abs_path, is_package=False)
            module_node.members = members
            if exported_names is not None:
                module_node.set_exported_names(exported_names)
            module_node.member_collisions.extend(intra_collisions)

    return root, all_intra_file_collisions
```

---
### File: src/vibelint/report.py

```python
"""
Report generation functionality for vibelint.

vibelint/report.py
"""

from pathlib import Path
from typing import List, TextIO, Set
from collections import defaultdict
from datetime import datetime
import logging


from .lint import LintResult
from .namespace import NamespaceNode, NamespaceCollision
from .config import Config
from .utils import get_relative_path

__all__ = ["write_report_content"]
logger = logging.getLogger(__name__)


def _get_files_in_namespace_order(
    node: NamespaceNode, collected_files: Set[Path], project_root: Path
) -> None:
    """
    Recursively collects file paths from the namespace tree in DFS order,
    including __init__.py files for packages. Populates the collected_files set.

    Args:
        node: The current NamespaceNode.
        collected_files: A set to store the absolute paths of collected files.
        project_root: The project root path for checking containment.

    vibelint/report.py (Modified)
    """

    if node.is_package and node.path and node.path.is_dir():
        try:

            node.path.relative_to(project_root)
            init_file = node.path / "__init__.py"

            if init_file.is_file() and init_file not in collected_files:

                init_file.relative_to(project_root)
                logger.debug(f"Report: Adding package init file: {init_file}")
                collected_files.add(init_file)
        except ValueError:
            logger.warning(
                f"Report: Skipping package node outside project root: {node.path}"
            )
        except Exception as e:
            logger.error(
                f"Report: Error checking package init file for {node.path}: {e}"
            )

    for child_name in sorted(node.children.keys()):
        child_node = node.children[child_name]

        if child_node.path and child_node.path.is_file() and not child_node.is_package:
            try:

                child_node.path.relative_to(project_root)
                if child_node.path not in collected_files:
                    logger.debug(f"Report: Adding module file: {child_node.path}")
                    collected_files.add(child_node.path)
            except ValueError:
                logger.warning(
                    f"Report: Skipping module file outside project root: {child_node.path}"
                )
            except Exception as e:
                logger.error(
                    f"Report: Error checking module file {child_node.path}: {e}"
                )

        _get_files_in_namespace_order(child_node, collected_files, project_root)

    if not node.children and node.path and node.path.is_file():
        try:
            node.path.relative_to(project_root)
            if node.path not in collected_files:
                logger.debug(f"Report: Adding root file node: {node.path}")
                collected_files.add(node.path)
        except ValueError:
            logger.warning(
                f"Report: Skipping root file node outside project root: {node.path}"
            )
        except Exception as e:
            logger.error(f"Report: Error checking root file node {node.path}: {e}")


def write_report_content(
    f: TextIO,
    project_root: Path,
    target_paths: List[Path],
    lint_results: List[LintResult],
    hard_coll: List[NamespaceCollision],
    soft_coll: List[NamespaceCollision],
    root_node: NamespaceNode,
    config: Config,
) -> None:
    """
    Writes the comprehensive markdown report content to the given file handle.

    Args:
    f: The text file handle to write the report to.
    project_root: The root directory of the project.
    target_paths: List of paths that were analyzed.
    lint_results: List of LintResult objects from the linting phase.
    hard_coll: List of hard NamespaceCollision objects.
    soft_coll: List of definition/export (soft) NamespaceCollision objects.
    root_node: The root NamespaceNode of the project structure.
    config: Configuration object.

    vibelint/report.py (Modified - calling corrected helper)
    """

    package_name = project_root.name if project_root else "Unknown"

    f.write("# vibelint Report\n\n")
    f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    f.write(f"**Project:** {package_name}\n")

    f.write(f"**Project Root:** `{str(project_root.resolve())}`\n\n")
    f.write(f"**Paths analyzed:** {', '.join(str(p) for p in target_paths)}\n\n")

    f.write("## Table of Contents\n\n")
    f.write("1. [Summary](#summary)\n")
    f.write("2. [Linting Results](#linting-results)\n")
    f.write("3. [Namespace Structure](#namespace-structure)\n")
    f.write("4. [Namespace Collisions](#namespace-collisions)\n")
    f.write("5. [File Contents](#file-contents)\n\n")

    f.write("## Summary\n\n")
    f.write("| Metric | Count |\n")
    f.write("|--------|-------|\n")

    files_analyzed_count = len(lint_results)
    f.write(f"| Files analyzed | {files_analyzed_count} |\n")
    f.write(f"| Files with errors | {sum(1 for r in lint_results if r.errors)} |\n")
    f.write(
        f"| Files with warnings only | {sum(1 for r in lint_results if r.warnings and not r.errors)} |\n"
    )
    f.write(f"| Hard namespace collisions | {len(hard_coll)} |\n")
    total_soft_collisions = len(soft_coll)
    f.write(f"| Definition/Export namespace collisions | {total_soft_collisions} |\n\n")

    f.write("## Linting Results\n\n")

    sorted_lint_results = sorted(lint_results, key=lambda r: r.file_path)
    files_with_issues = [r for r in sorted_lint_results if r.has_issues]

    if not files_with_issues:
        f.write("*No linting issues found.*\n\n")
    else:
        f.write("| File | Errors | Warnings |\n")
        f.write("|------|--------|----------|\n")
        for result in files_with_issues:

            errors_str = (
                "; ".join(f"`[{code}]` {msg}" for code, msg in result.errors)
                if result.errors
                else "None"
            )
            warnings_str = (
                "; ".join(f"`[{code}]` {msg}" for code, msg in result.warnings)
                if result.warnings
                else "None"
            )
            try:

                rel_path = get_relative_path(
                    result.file_path.resolve(), project_root.resolve()
                )
            except ValueError:
                rel_path = result.file_path

            f.write(f"| `{rel_path}` | {errors_str} | {warnings_str} |\n")
        f.write("\n")

    f.write("## Namespace Structure\n\n")
    f.write("```\n")
    try:

        tree_str = root_node.__str__()
        f.write(tree_str)
    except Exception as e:
        logger.error(f"Report: Error generating namespace tree string: {e}")
        f.write(f"[Error generating namespace tree: {e}]\n")
    f.write("\n```\n\n")

    f.write("## Namespace Collisions\n\n")
    f.write("### Hard Collisions\n\n")
    if not hard_coll:
        f.write("*No hard collisions detected.*\n\n")
    else:
        f.write(
            "These collisions can break Python imports or indicate duplicate definitions:\n\n"
        )
        f.write("| Name | Path 1 | Path 2 | Details |\n")
        f.write("|------|--------|--------|---------|\n")
        for collision in sorted(hard_coll, key=lambda c: (c.name, str(c.path1))):
            try:
                p1_rel = (
                    get_relative_path(collision.path1.resolve(), project_root.resolve())
                    if collision.path1
                    else "N/A"
                )
                p2_rel = (
                    get_relative_path(collision.path2.resolve(), project_root.resolve())
                    if collision.path2
                    else "N/A"
                )
            except ValueError:
                p1_rel = collision.path1 or "N/A"
                p2_rel = collision.path2 or "N/A"
            loc1 = f":{collision.lineno1}" if collision.lineno1 else ""
            loc2 = f":{collision.lineno2}" if collision.lineno2 else ""
            details = (
                "Intra-file duplicate"
                if str(p1_rel) == str(p2_rel)
                else "Module/Member clash"
            )
            f.write(
                f"| `{collision.name}` | `{p1_rel}{loc1}` | `{p2_rel}{loc2}` | {details} |\n"
            )
        f.write("\n")

    f.write("### Definition & Export Collisions (Soft)\n\n")
    if not soft_coll:
        f.write("*No definition or export collisions detected.*\n\n")
    else:
        f.write(
            "These names are defined/exported in multiple files, which may confuse humans and LLMs:\n\n"
        )
        f.write("| Name | Type | Files Involved |\n")
        f.write("|------|------|----------------|\n")
        grouped_soft = defaultdict(lambda: {"paths": set(), "types": set()})
        for collision in soft_coll:
            all_paths = collision.definition_paths or [collision.path1, collision.path2]
            grouped_soft[collision.name]["paths"].update(p for p in all_paths if p)
            grouped_soft[collision.name]["types"].add(collision.collision_type)

        for name, data in sorted(grouped_soft.items()):
            paths_str_list = []
            for p in sorted(list(data["paths"]), key=str):
                try:
                    paths_str_list.append(
                        f"`{get_relative_path(p.resolve(), project_root.resolve())}`"
                    )
                except ValueError:
                    paths_str_list.append(f"`{p}`")
            type_str = (
                " & ".join(
                    sorted([t.replace("_soft", "").upper() for t in data["types"]])
                )
                or "Unknown"
            )
            f.write(f"| `{name}` | {type_str} | {', '.join(paths_str_list)} |\n")
        f.write("\n")

    f.write("## File Contents\n\n")
    f.write("Files are ordered alphabetically by path.\n\n")

    collected_files_set: Set[Path] = set()
    try:
        _get_files_in_namespace_order(
            root_node, collected_files_set, project_root.resolve()
        )

        python_files_abs = sorted(list(collected_files_set), key=lambda p: str(p))
        logger.info(f"Report: Found {len(python_files_abs)} files for content section.")
    except Exception as e:
        logger.error(
            f"Report: Error collecting files for content section: {e}", exc_info=True
        )
        python_files_abs = []

    if not python_files_abs:
        f.write("*No Python files found in the namespace tree to display.*\n\n")
    else:
        for abs_file_path in python_files_abs:

            if abs_file_path and abs_file_path.is_file():
                try:

                    rel_path = get_relative_path(abs_file_path, project_root.resolve())
                    f.write(f"### {rel_path}\n\n")

                    try:
                        lang = "python"
                        content = abs_file_path.read_text(
                            encoding="utf-8", errors="ignore"
                        )
                        f.write(f"```{lang}\n")
                        f.write(content)

                        if not content.endswith("\n"):
                            f.write("\n")
                        f.write("```\n\n")
                    except Exception as read_e:
                        logger.warning(
                            f"Report: Error reading file content for {rel_path}: {read_e}"
                        )
                        f.write(f"*Error reading file content: {read_e}*\n\n")

                except ValueError:

                    logger.warning(
                        f"Report: Skipping file outside project root in content section: {abs_file_path}"
                    )
                    f.write(f"### {abs_file_path} (Outside Project Root)\n\n")
                    f.write(
                        "*Skipping content as file is outside the detected project root.*\n\n"
                    )
                except Exception as e_outer:
                    logger.error(
                        f"Report: Error processing file entry for {abs_file_path}: {e_outer}",
                        exc_info=True,
                    )
                    f.write(f"### Error Processing Entry for {abs_file_path}\n\n")
                    f.write(f"*An unexpected error occurred: {e_outer}*\n\n")
            elif abs_file_path:
                logger.warning(
                    f"Report: Skipping non-file path found during content writing: {abs_file_path}"
                )
                f.write(f"### {abs_file_path} (Not a File)\n\n")
                f.write("*Skipping entry as it is not a file.*\n\n")

            f.write("---\n\n")
```

---
### File: src/vibelint/results.py

```python
"""
Module for vibelint/results.py.

vibelint/results.py
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


from .lint import LintResult
from .namespace import NamespaceNode, NamespaceCollision


__all__ = ["CheckResult", "CommandResult", "NamespaceResult", "SnapshotResult"]


@dataclass
class CommandResult:
    """
    Base class for command results.

    vibelint/results.py
    """

    success: bool = True
    error_message: Optional[str] = None
    exit_code: int = 0

    def __post_init__(self):
        """
        Set exit code based on success if not explicitly set.

        vibelint/results.py
        """

        if not self.success and self.exit_code == 0:
            self.exit_code = 1


@dataclass
class CheckResult(CommandResult):
    """
    Result data from the 'check' command.

    vibelint/results.py
    """

    lint_results: List[LintResult] = field(default_factory=list)
    hard_collisions: List[NamespaceCollision] = field(default_factory=list)
    global_soft_collisions: List[NamespaceCollision] = field(default_factory=list)
    local_soft_collisions: List[NamespaceCollision] = field(default_factory=list)
    report_path: Optional[Path] = None
    report_generated: bool = False
    report_error: Optional[str] = None


@dataclass
class NamespaceResult(CommandResult):
    """
    Result data from the 'namespace' command.

    vibelint/results.py
    """

    root_node: Optional[NamespaceNode] = None
    intra_file_collisions: List[NamespaceCollision] = field(default_factory=list)
    output_path: Optional[Path] = None
    output_saved: bool = False
    output_error: Optional[str] = None


@dataclass
class SnapshotResult(CommandResult):
    """
    Result data from the 'snapshot' command.

    vibelint/results.py
    """

    output_path: Optional[Path] = None
```

---
### File: src/vibelint/snapshot.py

```python
"""
Codebase snapshot generation in markdown format.

vibelint/snapshot.py
"""

import fnmatch
import logging
from pathlib import Path
from typing import List, Dict, Tuple

from .config import Config
from .discovery import discover_files
from .utils import get_relative_path, is_binary

__all__ = ["create_snapshot"]

logger = logging.getLogger(__name__)


def create_snapshot(
    output_path: Path,
    target_paths: List[Path],
    config: Config,
):
    """
    Creates a Markdown snapshot file containing the project structure and file contents,
    respecting the include/exclude rules defined in pyproject.toml.

    Args:
    output_path: The path where the Markdown file will be saved.
    target_paths: List of initial paths (files or directories) to discover from.
    config: The vibelint configuration object.

    vibelint/snapshot.py
    """

    assert (
        config.project_root is not None
    ), "Project root must be set before creating snapshot."
    project_root = config.project_root.resolve()

    absolute_output_path = output_path.resolve()

    logger.debug("create_snapshot: Running discovery based on pyproject.toml config...")

    discovered_files = discover_files(
        paths=target_paths,
        config=config,
        explicit_exclude_paths={absolute_output_path},
    )

    logger.debug(f"create_snapshot: Discovery finished, count: {len(discovered_files)}")

    for excluded_pattern_root in [".pytest_cache", ".ruff_cache", ".git"]:
        present = any(excluded_pattern_root in str(f) for f in discovered_files)

        logger.debug(
            "!!! Check @ start of create_snapshot: '{}' presence in list: {}".format(
                excluded_pattern_root, present
            )
        )

    file_infos: List[Tuple[Path, str]] = []

    peek_globs = config.get("peek_globs", [])
    if not isinstance(peek_globs, list):
        logger.warning("Configuration 'peek_globs' is not a list. Ignoring peek rules.")
        peek_globs = []

    for abs_file_path in discovered_files:
        try:

            rel_path_str = get_relative_path(abs_file_path, project_root)
        except ValueError:

            logger.warning(
                f"Skipping file outside project root during snapshot categorization: {abs_file_path}"
            )
            continue

        if is_binary(abs_file_path):
            cat = "BINARY"
        else:
            cat = "FULL"
            for pk in peek_globs:

                normalized_rel_path = str(rel_path_str).replace("\\", "/")

                normalized_peek_glob = pk.replace("\\", "/")
                if fnmatch.fnmatch(normalized_rel_path, normalized_peek_glob):
                    cat = "PEEK"
                    break
        file_infos.append((abs_file_path, cat))
        logger.debug(f"Categorized {rel_path_str} as {cat}")

    file_infos.sort(key=lambda x: x[0])

    logger.debug(f"Sorted {len(file_infos)} files for snapshot.")

    tree: Dict = {}
    for f_path, f_cat in file_infos:
        try:

            relative_parts = str(get_relative_path(f_path, project_root)).split("/")
        except ValueError:

            logger.warning(
                f"Skipping file outside project root during snapshot tree build: {f_path}"
            )
            continue

        node = tree
        for i, part in enumerate(relative_parts):
            if not part:
                continue
            if i == len(relative_parts) - 1:

                if "__FILES__" not in node:
                    node["__FILES__"] = []

                node["__FILES__"].append((f_path, f_cat))
            else:

                if part not in node:
                    node[part] = {}
                node = node[part]

    logger.info(f"Writing snapshot to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:

        with open(absolute_output_path, "w", encoding="utf-8") as outfile:

            outfile.write("# Snapshot\n\n")

            outfile.write("## Filesystem Tree\n\n```\n")
            tree_root_name = (
                project_root.name if project_root.name else str(project_root)
            )
            outfile.write(f"{tree_root_name}/\n")
            _write_tree(outfile, tree, "")
            outfile.write("```\n\n")

            outfile.write("## File Contents\n\n")
            outfile.write("Files are ordered alphabetically by path.\n\n")
            for f, cat in file_infos:
                try:
                    relpath_header = get_relative_path(f, project_root)
                    outfile.write(f"### File: {relpath_header}\n\n")
                    logger.debug(
                        f"Writing content for {relpath_header} (Category: {cat})"
                    )

                    if cat == "BINARY":
                        outfile.write("```\n")
                        outfile.write("[Binary File - Content not displayed]\n")
                        outfile.write("```\n\n---\n")
                    elif cat == "PEEK":
                        outfile.write("```\n")
                        outfile.write("[PEEK - Content truncated]\n")
                        try:
                            with open(
                                f, "r", encoding="utf-8", errors="ignore"
                            ) as infile:
                                lines_read = 0
                                for line in infile:
                                    if lines_read >= 10:
                                        outfile.write("...\n")
                                        break
                                    outfile.write(line)
                                    lines_read += 1
                        except Exception as e:
                            logger.warning(
                                f"Error reading file for peek {relpath_header}: {e}"
                            )
                            outfile.write(f"[Error reading file for peek: {e}]\n")
                        outfile.write("```\n\n---\n")
                    else:
                        lang = _get_language(f)
                        outfile.write(f"```{lang}\n")
                        try:
                            with open(
                                f, "r", encoding="utf-8", errors="ignore"
                            ) as infile:
                                content = infile.read()
                                if not content.endswith("\n"):
                                    content += "\n"
                                outfile.write(content)
                        except Exception as e:
                            logger.warning(
                                f"Error reading file content {relpath_header}: {e}"
                            )
                            outfile.write(f"[Error reading file: {e}]\n")
                        outfile.write("```\n\n---\n")

                except Exception as e:
                    try:
                        relpath_header = get_relative_path(f, project_root)
                    except Exception:
                        relpath_header = str(f)

                    logger.error(
                        f"Error processing file entry for {relpath_header} in snapshot: {e}",
                        exc_info=True,
                    )
                    outfile.write(f"### File: {relpath_header} (Error)\n\n")
                    outfile.write(f"[Error processing file entry: {e}]\n\n---\n")

            outfile.write("\n")

    except IOError as e:

        logger.error(
            f"Failed to write snapshot file {absolute_output_path}: {e}", exc_info=True
        )
        raise
    except Exception as e:

        logger.error(
            f"An unexpected error occurred during snapshot writing: {e}", exc_info=True
        )
        raise


def _write_tree(outfile, node: Dict, prefix=""):
    """
    Helper function to recursively write the directory tree.

    vibelint/snapshot.py
    """

    dirs = sorted([k for k in node if k != "__FILES__"])

    files_data: List[Tuple[Path, str]] = sorted(
        node.get("__FILES__", []), key=lambda x: x[0].name
    )

    entries = dirs + [f[0].name for f in files_data]

    for i, name in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        outfile.write(f"{prefix}{connector}")

        if name in dirs:

            outfile.write(f"{name}/\n")
            new_prefix = prefix + ("    " if is_last else "│   ")
            _write_tree(outfile, node[name], new_prefix)
        else:

            file_info_tuple = next(
                (info for info in files_data if info[0].name == name), None
            )
            file_cat = "FULL"
            if file_info_tuple:
                file_cat = file_info_tuple[1]

            peek_indicator = " (PEEK)" if file_cat == "PEEK" else ""
            binary_indicator = " (BINARY)" if file_cat == "BINARY" else ""
            outfile.write(f"{name}{peek_indicator}{binary_indicator}\n")


def _get_language(file_path: Path) -> str:
    """
    Guess language for syntax highlighting based on extension.

    vibelint/snapshot.py
    """

    ext = file_path.suffix.lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".less": "less",
        ".json": "json",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".sh": "bash",
        ".ps1": "powershell",
        ".bat": "batch",
        ".sql": "sql",
        ".dockerfile": "dockerfile",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".gitignore": "gitignore",
    }
    return mapping.get(ext, "")
```

---
### File: src/vibelint/utils.py

```python
"""
Utility functions for vibelint.

vibelint/utils.py
"""

import os
from pathlib import Path
from typing import Optional, List


__all__ = [
    "ensure_directory",
    "find_files_by_extension",
    "find_package_root",
    "get_import_path",
    "get_module_name",
    "get_relative_path",
    "is_python_file",
    "read_file_safe",
    "write_file_safe",
    "find_project_root",
    "is_binary",
]


def find_project_root(start_path: Path) -> Optional[Path]:
    """
    Find the root directory of a project containing the given path.

    A project root is identified by containing either:
    1. A pyproject.toml file
    2. A .git directory

    Args:
    start_path: Path to start the search from

    Returns:
    Path to project root, or None if not found

    vibelint/utils.py
    """

    current_path = start_path.resolve()
    while True:
        if (current_path / "pyproject.toml").is_file():
            return current_path
        if (current_path / ".git").is_dir():
            return current_path
        if current_path.parent == current_path:
            return None
        current_path = current_path.parent


def find_package_root(start_path: Path) -> Optional[Path]:
    """
    Find the root directory of a Python package containing the given path.

    A package root is identified by containing either:
    1. A pyproject.toml file
    2. A  file
    3. An  file at the top level with no parent

    Args:
    start_path: Path to start the search from

    Returns:
    Path to package root, or None if not found

    vibelint/utils.py
    """

    current_path = start_path.resolve()
    if current_path.is_file():
        current_path = current_path.parent

    while True:
        if (current_path / "__init__.py").is_file():
            project_root_marker = find_project_root(current_path)
            if project_root_marker and current_path.is_relative_to(project_root_marker):
                pass

        if (current_path / "pyproject.toml").is_file() or (
            current_path / ".git"
        ).is_dir():
            src_dir = current_path / "src"
            if src_dir.is_dir():
                if start_path.resolve().is_relative_to(src_dir):
                    for item in src_dir.iterdir():
                        if item.is_dir() and (item / "__init__.py").is_file():
                            return item
                    return src_dir
                else:
                    if (current_path / "__init__.py").is_file():
                        return current_path

            if (current_path / "__init__.py").is_file():
                return current_path
            return current_path

        if current_path.parent == current_path:
            return start_path.parent if start_path.is_file() else start_path

        current_path = current_path.parent


def is_python_file(path: Path) -> bool:
    """
    Check if a path represents a Python file.

    Args:
    path: Path to check

    Returns:
    True if the path is a Python file, False otherwise

    vibelint/utils.py
    """

    return path.is_file() and path.suffix == ".py"


def get_relative_path(path: Path, base: Path) -> Path:
    """
    Safely compute a relative path, falling back to the original path.

    vibelint/utils.py
    """

    try:

        return path.resolve().relative_to(base.resolve())
    except ValueError:

        return path.resolve()


def get_import_path(file_path: Path, package_root: Optional[Path] = None) -> str:
    """
    Get the import path for a Python file.

    Args:
    file_path: Path to the Python file
    package_root: Optional path to the package root

    Returns:
    Import path (e.g., "vibelint.utils")

    vibelint/utils.py
    """

    if package_root is None:
        package_root = find_package_root(file_path)

    if package_root is None:
        return file_path.stem

    try:
        rel_path = file_path.relative_to(package_root)
        import_path = str(rel_path).replace(os.sep, ".").replace("/", ".")
        if import_path.endswith(".py"):
            import_path = import_path[:-3]
        return import_path
    except ValueError:
        return file_path.stem


def get_module_name(file_path: Path) -> str:
    """
    Extract module name from a Python file path.

    Args:
    file_path: Path to a Python file

    Returns:
    Module name

    vibelint/utils.py
    """

    return file_path.stem


def find_files_by_extension(
    root_path: Path,
    extension: str = ".py",
    exclude_globs: List[str] = [],
    include_vcs_hooks: bool = False,
) -> List[Path]:
    """
    Find all files with a specific extension in a directory and its subdirectories.

    Args:
    root_path: Root path to search in
    extension: File extension to look for (including the dot)
    exclude_globs: Glob patterns to exclude
    include_vcs_hooks: Whether to include version control directories

    Returns:
    List of paths to files with the specified extension

    vibelint/utils.py
    """

    import fnmatch

    if exclude_globs is None:
        exclude_globs = []

    result = []

    for file_path in root_path.glob(f"**/*{extension}"):
        if not include_vcs_hooks:
            if any(
                part.startswith(".") and part in {".git", ".hg", ".svn"}
                for part in file_path.parts
            ):
                continue

        if any(fnmatch.fnmatch(str(file_path), pattern) for pattern in exclude_globs):
            continue

        result.append(file_path)

    return result


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
    path: Path to directory

    Returns:
    Path to the directory

    vibelint/utils.py
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


def read_file_safe(file_path: Path, encoding: str = "utf-8") -> Optional[str]:
    """
    Safely read a file, returning None if any errors occur.

    Args:
    file_path: Path to file
    encoding: File encoding

    Returns:
    File contents or None if error

    vibelint/utils.py
    """

    try:
        return file_path.read_text(encoding=encoding)
    except Exception:
        return None


def write_file_safe(file_path: Path, content: str, encoding: str = "utf-8") -> bool:
    """
    Safely write content to a file, returning success status.

    Args:
    file_path: Path to file
    content: Content to write
    encoding: File encoding

    Returns:
    True if successful, False otherwise

    vibelint/utils.py
    """

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)
        return True
    except Exception:
        return False


def is_binary(file_path: Path, chunk_size: int = 1024) -> bool:
    """
    Check if a file appears to be binary by looking for null bytes
    or a high proportion of non-text bytes in the first chunk.

    Args:
    file_path: The path to the file.
    chunk_size: The number of bytes to read from the beginning.

    Returns:
    True if the file seems binary, False otherwise.

    vibelint/utils.py
    """

    try:
        with open(file_path, "rb") as f:
            chunk = f.read(chunk_size)
        if not chunk:
            return False

        if b"\x00" in chunk:
            return True

        text_characters = bytes(range(32, 127)) + b"\n\r\t\f\b"
        non_text_count = sum(
            1 for byte in chunk if bytes([byte]) not in text_characters
        )

        if len(chunk) > 0 and (non_text_count / len(chunk)) > 0.3:
            return True

        return False
    except OSError:

        return True
    except Exception:

        return True
```

---
### File: src/vibelint/validators/__init__.py

```python
"""
vibelint validators sub-package.

Re-exports key classes and functions for easier access.

vibelint/validators/__init__.py
"""

from .docstring import (
    DocstringValidationResult,
    validate_every_docstring,
    get_normalized_filepath,
)
from .encoding import (
    EncodingValidationResult,
    validate_encoding_cookie,
)
from .exports import (
    ExportValidationResult,
    validate_exports,
)
from .shebang import (
    ShebangValidationResult,
    validate_shebang,
    file_contains_top_level_main_block,
)

__all__ = [
    "DocstringValidationResult",
    "validate_every_docstring",
    "get_normalized_filepath",
    "EncodingValidationResult",
    "validate_encoding_cookie",
    "ExportValidationResult",
    "validate_exports",
    "ShebangValidationResult",
    "validate_shebang",
    "file_contains_top_level_main_block",
]
```

---
### File: src/vibelint/validators/docstring.py

```python
"""
Validator for Python docstrings. Checks for presence and path reference.

vibelint/validators/docstring.py
"""

import logging
import re
import sys
from typing import List, Optional, Dict, Tuple, Any, Union, Sequence, cast, Mapping

import libcst as cst
from libcst import (
    BaseStatement,
    IndentedBlock,
    Module,
    SimpleStatementLine,
    ClassDef,
    FunctionDef,
    Expr,
    SimpleString,
    BaseSuite,
    EmptyLine,
    Newline,
    Comment,
    Pass,
    TrailingWhitespace,
    CSTNode,
    BaseCompoundStatement,
    RemovalSentinel,
    BaseSmallStatement,
    MaybeSentinel,
    Name,
)
from libcst.metadata import (
    PositionProvider,
    CodeRange,
    MetadataWrapper,
    ProviderT,
    WhitespaceInclusivePositionProvider,
    ParentNodeProvider,
    CodePosition,
)


from ..error_codes import VBL101, VBL102, VBL103

logger = logging.getLogger(__name__)


__all__ = [
    "DocstringValidationResult",
    "get_normalized_filepath",
    "validate_every_docstring",
]

IssueKey = int
BodyItem = Union[BaseStatement, BaseSmallStatement, EmptyLine, Comment]
ValidationIssue = Tuple[str, str]


def _get_docstring_node(body_stmts: Sequence[CSTNode]) -> Optional[SimpleStatementLine]:
    """
    Attempts to get the CST node representing the docstring from a sequence of body statements.
    Searches for the first non-comment/empty statement and checks if it's a SimpleString expression.

    vibelint/validators/docstring.py
    """

    first_real_stmt = None
    for stmt in body_stmts:
        if not isinstance(stmt, (EmptyLine, Comment)):
            first_real_stmt = stmt
            break

    if (
        first_real_stmt
        and isinstance(first_real_stmt, SimpleStatementLine)
        and len(first_real_stmt.body) == 1
        and isinstance(first_real_stmt.body[0], Expr)
        and isinstance(first_real_stmt.body[0].value, SimpleString)
    ):
        return first_real_stmt
    return None


def _get_simple_string_node(body_stmts: Sequence[CSTNode]) -> Optional[SimpleString]:
    """
    Gets the SimpleString node if it's the first statement.

    vibelint/validators/docstring.py
    """

    doc_stmt_line = _get_docstring_node(body_stmts)
    if doc_stmt_line:
        try:
            expr_node = doc_stmt_line.body[0]
            if isinstance(expr_node, Expr) and isinstance(
                expr_node.value, SimpleString
            ):
                return expr_node.value
        except (IndexError, AttributeError):
            pass
    return None


def _extract_docstring_text(node: Optional[SimpleStatementLine]) -> Optional[str]:
    """
    Extracts the interpreted string value from a docstring node.

    vibelint/validators/docstring.py
    """

    if node:
        try:
            expr_node = node.body[0]
            if isinstance(expr_node, Expr):
                str_node = expr_node.value
                if isinstance(str_node, SimpleString):

                    evaluated = str_node.evaluated_value
                    return evaluated if isinstance(evaluated, str) else None
        except (IndexError, AttributeError, Exception) as e:

            logger.debug(f"Failed to extract/evaluate SimpleString: {e}", exc_info=True)
            return None
    return None


def _get_docstring_node_index(body_stmts: Sequence[CSTNode]) -> Optional[int]:
    """
    Gets the index of the docstring node in a body list.

    vibelint/validators/docstring.py
    """

    for i, stmt in enumerate(body_stmts):

        if isinstance(stmt, (EmptyLine, Comment)):
            continue

        if (
            isinstance(stmt, SimpleStatementLine)
            and len(stmt.body) == 1
            and isinstance(stmt.body[0], Expr)
            and isinstance(stmt.body[0].value, SimpleString)
        ):
            return i
        else:

            return None

    return None


class DocstringValidationResult:
    """
    Stores the result of docstring validation.

    vibelint/validators/docstring.py
    """

    def __init__(self) -> None:
        """
        Initializes DocstringValidationResult.

        vibelint/validators/docstring.py
        """
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []

    def has_issues(self) -> bool:
        """
        Checks if there are any errors or warnings.

        vibelint/validators/docstring.py
        """
        return bool(self.errors or self.warnings)

    def add_error(self, code: str, message: str):
        """
        Adds an error with its code.

        vibelint/validators/docstring.py
        """
        self.errors.append((code, message))

    def add_warning(self, code: str, message: str):
        """
        Adds a warning with its code.

        vibelint/validators/docstring.py
        """
        self.warnings.append((code, message))


def get_normalized_filepath(relative_path: str) -> str:
    """
    Normalizes a path for docstring references.
    Removes './', converts '' to '/', and removes leading 'src/'.

    vibelint/validators/docstring.py
    """

    path = relative_path.replace("\\", "/").lstrip("./")
    if path.startswith("src/"):
        return path[len("src/") :]
    return path


def get_node_start_line(
    node: CSTNode, metadata: Mapping[ProviderT, Mapping[CSTNode, object]]
) -> int:
    """
    Gets the 1-based start line number of a CST node using metadata.
    Returns 0 if position info is unavailable.

    vibelint/validators/docstring.py
    """

    try:
        pos_info = metadata.get(PositionProvider, {}).get(node)
        return pos_info.start.line if isinstance(pos_info, CodeRange) else 0
    except Exception:
        logger.debug(f"Failed to get start line for node {type(node)}", exc_info=True)
        return 0


def _get_node_base_indent_str(
    node: Union[FunctionDef, ClassDef, Module],
    metadata: Mapping[ProviderT, Mapping[CSTNode, object]],
) -> str:
    """
    Get the base indentation string of the node definition line.
    Returns empty string for Module or if indent info is unavailable.

    vibelint/validators/docstring.py
    """

    if isinstance(node, Module):
        return ""
    try:
        pos_info = metadata.get(WhitespaceInclusivePositionProvider, {}).get(node)
        return " " * pos_info.start.column if isinstance(pos_info, CodeRange) else ""
    except Exception:
        logger.debug(f"Failed to get indentation for node {type(node)}", exc_info=True)
        return ""


class DocstringInfoExtractor(cst.CSTVisitor):
    """
    Visits CST nodes to extract docstring info and validate.

    vibelint/validators/docstring.py
    """

    METADATA_DEPENDENCIES = (
        PositionProvider,
        WhitespaceInclusivePositionProvider,
        ParentNodeProvider,
    )

    def __init__(self, relative_path: str):
        """
        Initializes DocstringInfoExtractor.

        vibelint/validators/docstring.py
        """
        super().__init__()
        self.relative_path = relative_path
        self.path_ref = get_normalized_filepath(relative_path)
        self.result = DocstringValidationResult()

        logger.debug(
            f"[Validator:{self.relative_path}] Initialized. Expecting path ref: '{self.path_ref}'"
        )

    def visit_Module(self, node: Module) -> None:
        """Visits Module node."""
        doc_node = _get_docstring_node(node.body)
        doc_text = _extract_docstring_text(doc_node)
        self._validate_docstring(node, doc_node, doc_text, "module", "module")

    def leave_Module(self, node: Module) -> None:
        """Leaves Module node."""
        pass

    def visit_ClassDef(self, node: ClassDef) -> bool:
        """Visits ClassDef node."""
        if isinstance(node.body, IndentedBlock):
            doc_node = _get_docstring_node(node.body.body)
            doc_text = _extract_docstring_text(doc_node)
            self._validate_docstring(node, doc_node, doc_text, "class", node.name.value)
        else:
            self._validate_docstring(node, None, None, "class", node.name.value)
        return True

    def leave_ClassDef(self, node: ClassDef) -> None:
        """Leaves ClassDef node."""
        pass

    def visit_FunctionDef(self, node: FunctionDef) -> bool:
        """Visits FunctionDef node."""
        parent = self.get_metadata(ParentNodeProvider, node)
        is_method = isinstance(parent, IndentedBlock) and isinstance(
            self.get_metadata(ParentNodeProvider, parent), ClassDef
        )
        node_type = "method" if is_method else "function"

        if isinstance(node.body, IndentedBlock):
            doc_node = _get_docstring_node(node.body.body)
            doc_text = _extract_docstring_text(doc_node)
            self._validate_docstring(
                node, doc_node, doc_text, node_type, node.name.value
            )
        else:
            self._validate_docstring(node, None, None, node_type, node.name.value)
        return True

    def leave_FunctionDef(self, node: FunctionDef) -> None:
        """Leaves FunctionDef node."""
        pass

    def _validate_docstring(
        self,
        node: Union[Module, ClassDef, FunctionDef],
        node_doc: Optional[SimpleStatementLine],
        text_doc: Optional[str],
        n_type: str,
        n_name: str,
    ) -> None:
        """
        Performs the validation logic, reporting issues with codes.

        vibelint/validators/docstring.py
        """
        is_module = isinstance(node, Module)
        start_line = get_node_start_line(node, self.metadata)
        if start_line == 0:
            logger.warning(
                f"Could not get start line for {n_type} '{n_name}', skipping validation."
            )
            return

        doc_present = node_doc is not None

        is_simple_init = False
        if (
            n_name == "__init__"
            and n_type == "method"
            and isinstance(node, FunctionDef)
            and isinstance(node.body, IndentedBlock)
        ):
            non_empty_stmts = [
                s for s in node.body.body if not isinstance(s, (EmptyLine, Comment))
            ]
            doc_node_in_body = _get_docstring_node(node.body.body)
            actual_code_stmts = [
                s for s in non_empty_stmts if s is not doc_node_in_body
            ]
            if (
                len(actual_code_stmts) == 1
                and isinstance(actual_code_stmts[0], SimpleStatementLine)
                and len(actual_code_stmts[0].body) == 1
                and isinstance(actual_code_stmts[0].body[0], Pass)
            ):
                is_simple_init = True

        if not doc_present:
            if not (n_type == "method" and is_simple_init):
                msg = f"Missing docstring for {n_type} '{n_name}'."
                self.result.add_error(VBL101, msg)
                logger.debug(
                    f"[Validator:{self.relative_path}] Added issue {VBL101} for line {start_line}: Missing docstring"
                )
            else:
                logger.debug(
                    f"[Validator:{self.relative_path}] Validation OK (Suppressed simple __init__) for {n_type} '{n_name}' line {start_line}"
                )
            return

        path_issue = False
        if text_doc is not None:
            stripped_text = text_doc.rstrip()
            if not stripped_text.endswith(self.path_ref):
                path_issue = True
        else:
            path_issue = True

        if path_issue:
            msg = f"Docstring for {n_type} '{n_name}' missing/incorrect path reference (expected '{self.path_ref}')."
            self.result.add_warning(VBL102, msg)
            logger.debug(
                f"[Validator:{self.relative_path}] Added issue {VBL102} for line {start_line}: Path reference"
            )

        format_issue = False
        raw_value = None
        try:
            if node_doc:
                expr_node = node_doc.body[0]
                if isinstance(expr_node, Expr):
                    str_node = expr_node.value
                    if isinstance(str_node, SimpleString):
                        raw_value = str_node.value

                        is_multiline = "\n" in raw_value
                        if is_multiline:

                            base_indent = _get_node_base_indent_str(node, self.metadata)
                            docstring_stmt_indent = (
                                "" if is_module else base_indent + "    "
                            )
                            expected_ending = f'\n{docstring_stmt_indent}"""'

                            if not raw_value.endswith(expected_ending):
                                format_issue = True
                                logger.debug(
                                    f"Format issue L{start_line}: Closing quote mismatch. Raw: {repr(raw_value)}"
                                )

        except Exception as e:
            logger.debug(
                f"Format check failed for {n_name} L{start_line}: {e}", exc_info=True
            )

        if format_issue:
            msg = f"Docstring for {n_type} '{n_name}' has potential format/indentation issues."
            self.result.add_warning(VBL103, msg)
            logger.debug(
                f"[Validator:{self.relative_path}] Added issue {VBL103} for line {start_line}: Format/Indent"
            )

        if not path_issue and not format_issue and doc_present:
            logger.debug(
                f"[Validator:{self.relative_path}] Validation OK for {n_type} '{n_name}' line {start_line}"
            )


def validate_every_docstring(
    content: str, relative_path: str
) -> Tuple[DocstringValidationResult, Optional[Module]]:
    """
    Parse source code and run the DocstringInfoExtractor visitor to validate all docstrings.

    Args:
    content: The source code as a string.
    relative_path: The relative path of the file (used for path refs).

    Returns:
    A tuple containing:
    - DocstringValidationResult object with found issues.
    - The parsed CST Module node (or None if parsing failed).

    Raises:
    SyntaxError: If LibCST encounters a parsing error, it's converted and re-raised.

    vibelint/validators/docstring.py
    """
    result = DocstringValidationResult()
    module = None
    try:
        module = cst.parse_module(content)
        wrapper = MetadataWrapper(module, unsafe_skip_copy=True)

        wrapper.resolve(PositionProvider)
        wrapper.resolve(WhitespaceInclusivePositionProvider)
        wrapper.resolve(ParentNodeProvider)

        extractor = DocstringInfoExtractor(relative_path)
        wrapper.visit(extractor)
        logger.debug(
            f"[Validator:{relative_path}] Validation complete. Issues found: E={len(extractor.result.errors)}, W={len(extractor.result.warnings)}"
        )
        return extractor.result, module
    except cst.ParserSyntaxError as e:

        logger.warning(
            f"CST ParserSyntaxError in {relative_path} L{e.raw_line}:{e.raw_column}: {e.message}"
        )
        err = SyntaxError(e.message)
        err.lineno = e.raw_line
        err.offset = e.raw_column + 1 if e.raw_column is not None else None
        err.filename = relative_path
        try:
            err.text = content.splitlines()[e.raw_line - 1]
        except IndexError:
            err.text = None
        raise err from e
    except Exception as e:
        logger.error(
            f"Unexpected CST validation error {relative_path}: {e}", exc_info=True
        )

        result.add_error("VBL903", f"Internal validation error: {e}")
        return result, None
```

---
### File: src/vibelint/validators/encoding.py

```python
"""
Validator for Python encoding cookies.

vibelint/validators/encoding.py
"""

import re
from typing import List, Tuple


from ..error_codes import VBL201

__all__ = [
    "EncodingValidationResult",
    "validate_encoding_cookie",
]

ValidationIssue = Tuple[str, str]


class EncodingValidationResult:
    """
    Result of a validation for encoding cookies.

    vibelint/validators/encoding.py
    """

    def __init__(self) -> None:
        """Initializes EncodingValidationResult."""
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.line_number: int = -1

    def has_issues(self) -> bool:
        """Check if there are any issues."""
        return bool(self.errors or self.warnings)

    def add_error(self, code: str, message: str):
        """Adds an error with its code."""
        self.errors.append((code, message))

    def add_warning(self, code: str, message: str):
        """Adds a warning with its code."""
        self.warnings.append((code, message))


def validate_encoding_cookie(content: str) -> EncodingValidationResult:
    """
    Validate the encoding cookie in a Python file.

    vibelint/validators/encoding.py
    """
    result = EncodingValidationResult()
    lines = content.splitlines()
    pattern = r"^# -\*- coding: (.+) -\*-$"

    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1

    if idx < len(lines):
        m = re.match(pattern, lines[idx])
        if m:
            enc = m.group(1).lower()
            result.line_number = idx
            if enc != "utf-8":
                msg = f"Invalid encoding cookie: '{enc}' found on line {idx + 1}, must be 'utf-8'."
                result.add_error(VBL201, msg)

    return result
```

---
### File: src/vibelint/validators/exports.py

```python
"""
Validator for __all__ exports in Python modules.

Checks for presence (optional for __init__.py) and correct format.

vibelint/validators/exports.py
"""

import ast
from pathlib import Path
from typing import List, Optional, Tuple
from ..config import Config


from ..error_codes import VBL301, VBL302, VBL303, VBL304

__all__ = ["ExportValidationResult", "validate_exports"]

ValidationIssue = Tuple[str, str]


class ExportValidationResult:
    """
    Stores the result of __all__ validation for a single file.

    vibelint/validators/exports.py
    """

    def __init__(self, file_path: str) -> None:
        """Initializes ExportValidationResult."""
        self.file_path = file_path
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.has_all: bool = False

        self.all_lineno: Optional[int] = None

    def has_issues(self) -> bool:
        """Returns True if there are any errors or warnings."""
        return bool(self.errors or self.warnings)

    def add_error(self, code: str, message: str):
        """Adds an error with its code."""
        self.errors.append((code, message))

    def add_warning(self, code: str, message: str):
        """Adds a warning with its code."""
        self.warnings.append((code, message))


def validate_exports(
    source_code: str, file_path_str: str, config: Config
) -> ExportValidationResult:
    """
    Validates the presence and format of __all__ in the source code.

    Args:
    source_code: The source code of the Python file as a string.
    file_path_str: The path to the file (used for context, e.g., __init__.py).
    config: The loaded vibelint configuration.

    Returns:
    An ExportValidationResult object.

    vibelint/validators/exports.py
    """
    result = ExportValidationResult(file_path_str)
    file_path = Path(file_path_str)
    is_init_py = file_path.name == "__init__.py"

    try:
        tree = ast.parse(source_code, filename=file_path_str)
    except SyntaxError as e:
        result.add_error(VBL304, f"SyntaxError parsing file: {e}")
        return result

    found_all = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
            ):
                found_all = True
                result.has_all = True
                result.all_lineno = node.lineno

                if not isinstance(node.value, (ast.List, ast.Tuple)):
                    msg = f"__all__ is not assigned a List or Tuple (assigned {type(node.value).__name__}) near line {node.lineno}."
                    result.add_error(VBL303, msg)

                break

    if not found_all:
        error_on_init = config.get("error_on_missing_all_in_init", False)
        if is_init_py and not error_on_init:
            msg = f"Optional: __all__ definition not found in {file_path.name}."
            result.add_warning(VBL302, msg)
        elif not is_init_py:
            msg = f"__all__ definition not found in {file_path.name}."
            result.add_error(VBL301, msg)
        elif is_init_py and error_on_init:
            msg = f"__all__ definition not found in {file_path.name} (required by config)."
            result.add_error(VBL301, msg)

    return result
```

---
### File: src/vibelint/validators/shebang.py

```python
"""
Validator for Python shebang lines.

vibelint/validators/shebang.py
"""

from typing import List, Tuple
import ast
from pathlib import Path


from ..error_codes import VBL401, VBL402, VBL403

__all__ = [
    "ShebangValidationResult",
    "validate_shebang",
    "file_contains_top_level_main_block",
]

ValidationIssue = Tuple[str, str]


class ShebangValidationResult:
    """
    Result of a shebang validation.

    vibelint/validators/shebang.py
    """

    def __init__(self) -> None:
        """Initializes ShebangValidationResult."""
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.line_number: int = 0

    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return bool(self.errors or self.warnings)

    def add_error(self, code: str, message: str):
        """Adds an error with its code."""
        self.errors.append((code, message))

    def add_warning(self, code: str, message: str):
        """Adds a warning with its code."""
        self.warnings.append((code, message))


def validate_shebang(
    content: str, is_script: bool, allowed_shebangs: List[str]
) -> ShebangValidationResult:
    """
    Validate the shebang line if present; ensure it's correct for scripts with __main__.

    vibelint/validators/shebang.py
    """
    res = ShebangValidationResult()
    lines = content.splitlines()

    if lines and lines[0].startswith("#!"):
        sb = lines[0]
        res.line_number = 0
        if not is_script:
            msg = f"File has shebang '{sb}' but no 'if __name__ == \"__main__\":' block found."
            res.add_error(VBL401, msg)
        elif sb not in allowed_shebangs:
            allowed_str = ", ".join(f"'{s}'" for s in allowed_shebangs)
            msg = f"Invalid shebang '{sb}'. Allowed: {allowed_str}."
            res.add_error(VBL402, msg)
    else:
        if is_script:
            res.line_number = 0
            msg = "Script contains 'if __name__ == \"__main__\":' block but lacks a shebang line ('#!...')."
            res.add_warning(VBL403, msg)

    return res


def file_contains_top_level_main_block(file_path: Path, content: str) -> bool:
    """
    Checks if a Python file contains a top-level 'if __name__ == "__main__":' block using AST.
    Returns True if found, False otherwise (including syntax errors).

    vibelint/validators/shebang.py
    """

    try:
        tree = ast.parse(content, filename=str(file_path))
        for node in tree.body:
            if isinstance(node, ast.If):
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq)
                    and len(test.comparators) == 1
                    and (
                        (
                            isinstance(test.comparators[0], ast.Constant)
                            and test.comparators[0].value == "__main__"
                        )
                        or (
                            isinstance(test.comparators[0], ast.Str)
                            and test.comparators[0].s == "__main__"
                        )
                    )
                ):
                    return True
    except (SyntaxError, Exception):

        return False
    return False
```

---
### File: tests/fixtures/check_success/myproject/pyproject.toml

```toml
[project]
name = "myproject"
version = "0.1.0"

[tool.vibelint]
include_globs = ["src/**/*.py"]
error_on_missing_all_in_init = false
large_dir_threshold = 500
```

---
### File: tests/fixtures/check_success/myproject/src/mypkg/__init__.py

```python
"""
Package init.

mypkg/__init__.py
"""

__all__ = []
```

---
### File: tests/fixtures/check_success/myproject/src/mypkg/module.py

```python
"""
A sample module.

mypkg/module.py
"""

__all__ = ["hello"]


def hello():
    """
    Prints hello.

    mypkg/module.py
    """
    print("Hello, world!")
```

---
### File: tests/fixtures/fix_missing_all/fixproj/another.py

```python
"""Another file. MISSING PATH"""

__all__ = []


def something():
    pass
```

---
### File: tests/fixtures/fix_missing_all/fixproj/needs_fix.py

```python
"""Module needing a fix. MISSING PATH"""


def func_one():
    pass


def _internal_func():
    """Internal func doc. MISSING PATH"""
    pass


def func_two():
    pass
```

---
### File: tests/fixtures/fix_missing_all/fixproj/pyproject.toml

```toml
[project]
name = "fixproj"
version = "0.1.0"

[tool.vibelint]
include_globs = ["*.py"]
large_dir_threshold = 500
```

---
### File: tests/pleasehelpmewritethese.txt

```
These tests are far from comprehensive... and were generated from the LLM generated code.

However, they serve as a baseline for communicating the intended behavior of vibelint to the greater open source community.
```

---
### File: tests/test_cli.py

```python
import pytest
from click.testing import CliRunner
from pathlib import Path
import shutil
import sys
import os
import io
import re


if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

try:
    import tomli_w
except ImportError:
    tomli_w = None


from vibelint.cli import cli
from vibelint import __version__


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def runner():
    """Provides a Click CliRunner instance."""
    return CliRunner()


@pytest.fixture
def setup_test_project(tmp_path, request):
    """
    Copies a fixture directory into a temporary directory and changes
    the current working directory to the *actual project root* within
    that temporary directory for the duration of the test.

    Yields the Path to the temporary project root.
    """
    fixture_name = request.param
    source_fixture_path = FIXTURES_DIR / fixture_name
    if not source_fixture_path.is_dir():
        raise ValueError(f"Fixture directory not found: {source_fixture_path}")

    project_dirs = [d for d in source_fixture_path.iterdir() if d.is_dir()]
    if len(project_dirs) != 1:
        if (source_fixture_path / "pyproject.toml").exists():
            project_dir_name = None
            target_project_root = tmp_path / fixture_name
            shutil.copytree(
                source_fixture_path, target_project_root, dirs_exist_ok=True
            )
        else:
            raise ValueError(
                f"Fixture '{fixture_name}' must contain exactly one project subdirectory "
                f"or have a pyproject.toml in its root."
            )
    else:
        project_dir_name = project_dirs[0].name
        target_fixture_path = tmp_path / fixture_name
        shutil.copytree(source_fixture_path, target_fixture_path, dirs_exist_ok=True)
        target_project_root = target_fixture_path / project_dir_name

    original_cwd = Path.cwd()
    os.chdir(target_project_root)
    print(f"DEBUG: Changed CWD to: {Path.cwd()}")
    try:
        yield target_project_root
    finally:
        os.chdir(original_cwd)
        print(f"DEBUG: Restored CWD to: {Path.cwd()}")


def modify_pyproject(project_path: Path, updates: dict):
    """Modifies the [tool.vibelint] section of pyproject.toml."""
    if tomllib is None:
        pytest.fail("TOML reading library (tomllib/tomli) could not be imported.")
    if tomli_w is None:
        pytest.fail(
            "TOML writing library (tomli_w) could not be imported. Is it installed?"
        )

    pyproject_file = project_path / "pyproject.toml"
    print(f"DEBUG: Attempting to modify pyproject.toml at: {pyproject_file}")
    if not pyproject_file.is_file():
        print(f"DEBUG: Contents of {project_path}: {list(project_path.iterdir())}")
        raise FileNotFoundError(f"pyproject.toml not found in {project_path}")

    with open(pyproject_file, "rb") as f:
        data = tomllib.load(f)

    if "tool" not in data:
        data["tool"] = {}
    if "vibelint" not in data["tool"]:
        data["tool"]["vibelint"] = {}
    data["tool"]["vibelint"].update(updates)

    with open(pyproject_file, "wb") as f:
        tomli_w.dump(data, f)
    print(f"DEBUG: Successfully modified {pyproject_file}")


def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"], prog_name="vibelint")
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Usage: vibelint [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "check" in result.output
    assert "namespace" in result.output
    assert "snapshot" in result.output


def test_cli_no_project_root(runner, tmp_path):
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["check"], prog_name="vibelint")
        assert result.exit_code == 1
        assert "Error: Could not find project root." in result.output
        assert "a 'pyproject.toml' file or a '.git' directory" in result.output
    finally:
        os.chdir(original_cwd)


def assert_in_output(substring: str, full_output: str, msg: str = ""):
    """Asserts substring is in full_output, handling potential formatting chars."""
    if substring in full_output:
        return

    escaped_substring = re.escape(substring)

    pattern_basic = rf"\x1b\[.*?m{escaped_substring}\x1b\[.*?m|{escaped_substring}"
    if re.search(pattern_basic, full_output):
        return

    cleaned_output = re.sub(r"\x1b\[.*?m", "", full_output)
    cleaned_output = cleaned_output.replace("\r", "")

    cleaned_output = (
        cleaned_output.replace("│", "|")
        .replace("─", "-")
        .replace("┌", "+")
        .replace("┐", "+")
        .replace("└", "+")
        .replace("┘", "+")
        .replace("├", "|")
        .replace("┤", "|")
        .replace("┬", "+")
        .replace("┴", "+")
    )
    cleaned_output = re.sub(r"\s+", " ", cleaned_output)

    cleaned_substring = (
        substring.replace("│", "|")
        .replace("─", "-")
        .replace("┌", "+")
        .replace("┐", "+")
        .replace("└", "+")
        .replace("┘", "+")
        .replace("├", "|")
        .replace("┤", "|")
        .replace("┬", "+")
        .replace("┴", "+")
    )
    cleaned_substring = re.sub(r"\s+", " ", cleaned_substring).strip()

    assert (
        cleaned_substring in cleaned_output
    ), f"{msg} Substring '{substring}' not found in cleaned output:\n{cleaned_output}\nOriginal output:\n{full_output}"


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_check_success(runner, setup_test_project):
    result = runner.invoke(cli, ["check"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert_in_output("vibelint Results Summary", result.output)
    assert_in_output("Files Scanned", result.output)
    assert_in_output(" 2 ", result.output)
    assert_in_output("Files OK", result.output)
    assert_in_output("Files with Errors", result.output)
    assert_in_output(" 0 ", result.output)
    assert_in_output("Files with Warnings only", result.output)
    assert "Check finished successfully" in result.output
    assert "Namespace Collision Results Summary" not in result.output


@pytest.mark.parametrize("setup_test_project", ["fix_missing_all"], indirect=True)
def test_check_errors_missing_all(runner, setup_test_project):
    result = runner.invoke(cli, ["check"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 1, got {result.exit_code}. Output:\n{result.output}"
    assert_in_output("vibelint Results Summary", result.output)
    assert_in_output("Files Scanned", result.output)
    assert_in_output(" 2 ", result.output)
    assert_in_output("Files OK", result.output)
    assert_in_output(" 0 ", result.output)
    assert_in_output("Files with Errors", result.output)
    assert_in_output(" 2 ", result.output)
    assert "[VBL301] __all__ definition not found" in result.output
    assert "[VBL101] Missing docstring for function 'func_one'" in result.output
    assert "[VBL101] Missing docstring for function 'something'" in result.output
    assert (
        "[VBL102] Docstring for module 'module' missing/incorrect path reference"
        in result.output
    )
    assert "Check finished with errors (exit code 1)" in result.output


@pytest.mark.parametrize("setup_test_project", ["fix_missing_all"], indirect=True)
def test_check_ignore_codes(runner, setup_test_project):
    modify_pyproject(setup_test_project, {"ignore": ["VBL301"]})
    result = runner.invoke(cli, ["check"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 1, got {result.exit_code}. Output:\n{result.output}"
    assert_in_output("vibelint Results Summary", result.output)
    assert_in_output("Files Scanned", result.output)
    assert_in_output(" 2 ", result.output)
    assert_in_output("Files with Errors", result.output)
    assert_in_output(" 2 ", result.output)
    assert "[VBL301]" not in result.output
    assert "[VBL101] Missing docstring for function 'func_one'" in result.output
    assert "[VBL101] Missing docstring for function 'something'" in result.output
    assert "Check finished with errors (exit code 1)" in result.output


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_check_output_report(runner, setup_test_project):
    """Test `vibelint check -o report.md`."""
    report_file = setup_test_project / "vibelint_report.md"
    assert not report_file.exists()

    result = runner.invoke(
        cli, ["check", "-o", "vibelint_report.md"], prog_name="vibelint"
    )
    print(f"Output:\n{result.output}")

    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert report_file.exists()
    assert "Report generated at" in result.output
    assert str(report_file.resolve()) in result.output.replace("\n", "")

    report_content = report_file.read_text()
    assert "# vibelint Report" in report_content
    assert "## Summary" in report_content
    assert "## Linting Results" in report_content
    assert "*No linting issues found.*" in report_content
    assert "## Namespace Structure" in report_content

    assert "myproject" in report_content
    assert "src" in report_content
    assert "mypkg" in report_content
    assert "module" in report_content
    assert "hello (member)" in report_content
    assert "## Namespace Collisions" in report_content
    assert "*No hard collisions detected.*" in report_content
    assert "## File Contents" in report_content
    assert "### src/mypkg/__init__.py" in report_content
    assert "### src/mypkg/module.py" in report_content


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_check_with_hard_collision(runner, setup_test_project):
    """Create and test a hard collision: object in parent __init__ vs child dir."""
    src_dir = setup_test_project / "src"
    src_init_file = src_dir / "__init__.py"
    mypkg_dir = src_dir / "mypkg"

    print(f"DEBUG: Creating file: {src_init_file}")
    src_init_file.touch()
    assert src_init_file.is_file(), f"{src_init_file} not created!"

    print(f"DEBUG: Modifying file: {src_init_file} to add collision")
    src_init_file.write_text("mypkg = 123 # This clashes with the mypkg directory\n")
    print(f"DEBUG: Successfully modified {src_init_file}")

    result = runner.invoke(cli, ["check"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 1 (hard collision), got {result.exit_code}. Output:\n{result.output}"
    assert_in_output("Namespace Collision Results Summary", result.output)
    assert_in_output("Hard Collisions", result.output)

    collision_summary_match = re.search(r"Hard Collisions\s*│\s*1\s*│", result.output)
    assert (
        collision_summary_match is not None
    ), "Could not find 'Hard Collisions | 1 |' row in summary table"

    assert_in_output("Hard Collisions:", result.output)
    assert "- 'mypkg': Conflicting definitions/imports" in result.output
    assert str(src_init_file.relative_to(setup_test_project)) in result.output
    assert str(mypkg_dir.relative_to(setup_test_project)) in result.output
    assert "Check finished with errors (exit code 1)" in result.output


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_namespace_basic(runner, setup_test_project):
    result = runner.invoke(cli, ["namespace"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert "Namespace Structure:" in result.output
    assert "myproject" in result.output
    assert "└── src" in result.output
    assert "└── mypkg" in result.output
    assert "└── module" in result.output
    assert "hello (member)" in result.output


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_namespace_output_file(runner, setup_test_project):
    tree_file = setup_test_project / "namespace_tree.txt"
    assert not tree_file.exists()

    result = runner.invoke(
        cli, ["namespace", "-o", "namespace_tree.txt"], prog_name="vibelint"
    )
    print(f"Output:\n{result.output}")

    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert tree_file.exists()
    assert "Namespace tree saved to" in result.output
    assert str(tree_file.resolve()) in result.output.replace("\n", "")

    tree_content = tree_file.read_text()
    assert "myproject" in tree_content
    assert "└── src" in tree_content
    assert "└── mypkg" in tree_content
    assert "hello (member)" in tree_content


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_namespace_intra_file_collision(runner, setup_test_project):
    module_file = setup_test_project / "src" / "mypkg" / "module.py"
    print(f"DEBUG: Modifying file: {module_file}")
    assert module_file.is_file(), f"{module_file} not found!"
    content = module_file.read_text()
    content += "\nhello = 123 # Duplicate definition\n"
    module_file.write_text(content)
    print(f"DEBUG: Successfully modified {module_file}")

    result = runner.invoke(cli, ["namespace"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert "Intra-file Collisions Found:" in result.output
    assert (
        "- 'hello': Duplicate definition/import in src/mypkg/module.py" in result.output
    )
    assert "Namespace Structure:" in result.output
    assert "myproject" in result.output


@pytest.fixture
def setup_snapshot_project(tmp_path):
    """Fixture specifically for snapshot tests, ensuring pyproject.toml is included."""
    fixture_name = "check_success"
    source_fixture_path = FIXTURES_DIR / fixture_name
    project_dir_name = "myproject"
    target_fixture_path = tmp_path / fixture_name
    shutil.copytree(source_fixture_path, target_fixture_path, dirs_exist_ok=True)
    target_project_root = target_fixture_path / project_dir_name

    pyproject_file = target_project_root / "pyproject.toml"
    if tomllib and tomli_w and pyproject_file.is_file():
        with open(pyproject_file, "rb") as f:
            data = tomllib.load(f)
        if "tool" not in data:
            data["tool"] = {}
        if "vibelint" not in data["tool"]:
            data["tool"]["vibelint"] = {}
        data["tool"]["vibelint"]["include_globs"] = ["src/**/*.py", "pyproject.toml"]
        with open(pyproject_file, "wb") as f:
            tomli_w.dump(data, f)
    else:
        print("WARN: Could not modify pyproject.toml for snapshot include test.")

    original_cwd = Path.cwd()
    os.chdir(target_project_root)
    print(f"DEBUG: Snapshot Test Changed CWD to: {Path.cwd()}")
    try:
        yield target_project_root
    finally:
        os.chdir(original_cwd)
        print(f"DEBUG: Snapshot Test Restored CWD to: {Path.cwd()}")


def test_snapshot_basic(runner, setup_snapshot_project):
    """Test `vibelint snapshot` default behavior (using modified fixture)."""
    snapshot_file = setup_snapshot_project / "codebase_snapshot.md"
    assert not snapshot_file.exists()

    result = runner.invoke(cli, ["snapshot"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert snapshot_file.exists()
    assert "Codebase snapshot created at" in result.output
    assert str(snapshot_file.resolve()) in result.output.replace("\n", "")

    snapshot_content = snapshot_file.read_text()

    tree_match = re.search(
        r"## Filesystem Tree\s*```\s*(.*?)\s*```", snapshot_content, re.DOTALL
    )
    assert tree_match, "Filesystem Tree section not found in snapshot"
    tree_block = tree_match.group(1)

    assert "# Snapshot" in snapshot_content
    assert "## Filesystem Tree" in snapshot_content
    assert "myproject/" in tree_block

    assert "pyproject.toml" in tree_block
    assert "src/" in tree_block
    assert "mypkg/" in tree_block
    assert "__init__.py" in tree_block
    assert "module.py" in tree_block

    assert "## File Contents" in snapshot_content
    assert "### File: pyproject.toml" in snapshot_content
    assert "[tool.vibelint]" in snapshot_content
    assert "### File: src/mypkg/__init__.py" in snapshot_content
    assert "### File: src/mypkg/module.py" in snapshot_content


def test_snapshot_output_file(runner, setup_snapshot_project):
    snapshot_file = setup_snapshot_project / "custom_snapshot.md"
    assert not snapshot_file.exists()

    result = runner.invoke(
        cli, ["snapshot", "-o", "custom_snapshot.md"], prog_name="vibelint"
    )
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert snapshot_file.exists()
    assert "Codebase snapshot created at" in result.output
    assert str(snapshot_file.resolve()) in result.output.replace("\n", "")


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_snapshot_exclude(runner, setup_test_project):
    """Test snapshot respects exclude_globs from config ('check_success' fixture)."""
    modify_pyproject(setup_test_project, {"exclude_globs": ["src/mypkg/module.py"]})

    snapshot_file = setup_test_project / "codebase_snapshot.md"
    result = runner.invoke(cli, ["snapshot"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert snapshot_file.exists()

    snapshot_content = snapshot_file.read_text()
    tree_match = re.search(
        r"## Filesystem Tree\s*```\s*(.*?)\s*```", snapshot_content, re.DOTALL
    )
    assert tree_match, "Filesystem Tree section not found in snapshot"
    tree_block = tree_match.group(1)

    assert "module.py" not in tree_block
    assert "### File: src/mypkg/module.py" not in snapshot_content

    assert "__init__.py" in tree_block
    assert "### File: src/mypkg/__init__.py" in snapshot_content

    assert "pyproject.toml" not in tree_block
    assert "### File: pyproject.toml" not in snapshot_content


def test_snapshot_exclude_output_file(runner, setup_snapshot_project):
    """Test snapshot doesn't include its own output file (using modified fixture)."""
    snapshot_file = setup_snapshot_project / "mysnapshot.md"

    result1 = runner.invoke(
        cli, ["snapshot", "-o", "mysnapshot.md"], prog_name="vibelint"
    )
    assert (
        result1.exit_code == 0
    ), f"Snapshot creation failed (1st run). Output:\n{result1.output}"
    assert snapshot_file.exists()

    result2 = runner.invoke(
        cli, ["snapshot", "-o", "mysnapshot.md"], prog_name="vibelint"
    )
    assert (
        result2.exit_code == 0
    ), f"Snapshot creation failed (2nd run). Output:\n{result2.output}"

    snapshot_content = snapshot_file.read_text()
    assert "mysnapshot.md" not in snapshot_content
    assert "### File: pyproject.toml" in snapshot_content
```

---
### File: tox.ini

```ini
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
```

---

