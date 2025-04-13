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