#!/usr/bin/env python3
"""
Convert all relative imports to absolute imports in vibelint.

This script systematically finds and replaces relative imports with absolute ones.
"""

import re
import sys
from pathlib import Path


def convert_relative_to_absolute(file_path: Path, content: str) -> str:
    """Convert relative imports to absolute imports."""

    # Get the module path relative to src/vibelint
    relative_path = file_path.relative_to(file_path.parents[2] / "src")
    module_parts = list(relative_path.parts[:-1])  # Remove the .py file part

    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        # Match relative imports like "from .module import something"
        rel_import_match = re.match(r'^(\s*)from (\.+)([^.\s]*) import (.+)$', line)
        if rel_import_match:
            indent, dots, module_name, imports = rel_import_match.groups()

            # Calculate the absolute module path
            levels_up = len(dots) - 1  # Number of parent directories to go up

            if levels_up == 0:
                # Same directory: from .module import something
                if module_name:
                    abs_module = ".".join(module_parts + [module_name])
                else:
                    # from . import something (importing from __init__.py)
                    abs_module = ".".join(module_parts)
            else:
                # Parent directories: from ..parent.module import something
                if len(module_parts) >= levels_up:
                    base_parts = module_parts[:-levels_up] if levels_up > 0 else module_parts
                    if module_name:
                        abs_module = ".".join(base_parts + [module_name])
                    else:
                        abs_module = ".".join(base_parts)
                else:
                    # Too many levels up, keep as is
                    fixed_lines.append(line)
                    continue

            # Create the absolute import
            fixed_line = f"{indent}from {abs_module} import {imports}"
            fixed_lines.append(fixed_line)
            print(f"  {file_path.name}: {line.strip()} -> {fixed_line.strip()}")

        else:
            # Not a relative import, keep as is
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def main():
    """Convert all relative imports in the vibelint source."""

    vibelint_src = Path("src/vibelint")

    if not vibelint_src.exists():
        print("Error: src/vibelint directory not found")
        sys.exit(1)

    print("Converting relative imports to absolute imports...")

    # Find all Python files
    python_files = list(vibelint_src.rglob("*.py"))

    for py_file in python_files:
        try:
            content = py_file.read_text(encoding='utf-8')

            # Check if file has relative imports
            if 'from .' in content:
                print(f"\nProcessing {py_file}")
                fixed_content = convert_relative_to_absolute(py_file, content)
                py_file.write_text(fixed_content, encoding='utf-8')

        except Exception as e:
            print(f"Error processing {py_file}: {e}")

    print("\nDone! All relative imports converted to absolute imports.")


if __name__ == "__main__":
    main()