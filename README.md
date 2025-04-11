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