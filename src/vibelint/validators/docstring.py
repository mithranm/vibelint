"""
Validator for Python module docstrings.

vibelint/validators/docstring.py
"""

import re
import os
from typing import List, Optional


class ValidationResult:
    """
    Class to store the result of a validation.

    vibelint/validators/docstring.py
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.line_number: int = -1
        self.needs_fix: bool = False
        self.module_docstring: Optional[str] = None

    def has_issues(self) -> bool:
        """Check if there are any issues."""
        return len(self.errors) > 0 or len(self.warnings) > 0


def validate_module_docstring(
    content: str, relative_path: str, docstring_regex: str
) -> ValidationResult:
    """
    Validate the module docstring in a Python file.

    vibelint/validators/docstring.py
    """
    result = ValidationResult()
    lines = content.splitlines()

    # Skip shebang and encoding cookie if present
    line_index = 0
    if len(lines) > line_index and lines[line_index].startswith("#!"):
        line_index += 1
    if len(lines) > line_index and lines[line_index].startswith("# -*-"):
        line_index += 1

    # Skip blank lines
    while line_index < len(lines) and not lines[line_index].strip():
        line_index += 1

    # Check for docstring
    docstring_start = None
    docstring_end = None
    docstring_lines = []

    # Try to find the docstring
    for i in range(line_index, min(line_index + 10, len(lines))):
        line = lines[i].strip()
        if line.startswith('"""'):
            docstring_start = i
            # Single line docstring
            if line.endswith('"""') and len(line) > 6:
                docstring_end = i
                docstring_lines = [line[3:-3].strip()]
                break

            # Handle multi-line docstring
            first_content_line = None
            if line == '"""':
                # Triple quotes on their own line - content starts on next line
                first_content_line = i + 1
            else:
                # Content starts on same line as opening quotes
                first_content_line = i
                docstring_lines.append(line[3:].strip())

            # Find the end of the docstring
            for j in range(first_content_line, len(lines)):
                current_line = lines[j].strip()
                # Look for a line that ends with triple quotes or is just triple quotes
                if current_line == '"""' or current_line.endswith('"""'):
                    docstring_end = j

                    # If we didn't already add the first line (when quotes were on their own line)
                    if first_content_line > i:
                        # Add lines between opening quote and closing quote
                        docstring_lines.extend(
                            lines[k].strip()
                            for k in range(first_content_line, j)
                            if lines[k].strip()
                        )
                    else:
                        # Add lines after the first content line
                        docstring_lines.extend(
                            lines[k].strip()
                            for k in range(i + 1, j)
                            if lines[k].strip()
                        )

                    # Add content from the last line if it has content before the closing quotes
                    if current_line != '"""':
                        content_part = current_line.split('"""')[0].strip()
                        if content_part:
                            docstring_lines.append(content_part)
                    break
            break

    # If no docstring found or incomplete docstring
    if docstring_start is None or docstring_end is None:
        result.errors.append("Module docstring missing or incomplete")
        result.line_number = line_index
        result.needs_fix = True
        return result

    # Store the docstring for potential fixes
    result.module_docstring = "\n".join(docstring_lines)
    result.line_number = docstring_start

    # Validate docstring content
    if not docstring_lines:
        result.errors.append("Empty module docstring")
        result.needs_fix = True
        return result

    # Check first line format (capitalized sentence ending in period)
    # Get the first non-empty line
    first_content = next((line for line in docstring_lines if line), "")
    if not re.match(docstring_regex, first_content):
        result.errors.append(
            f"First line of docstring should match regex: {docstring_regex}"
        )
        result.needs_fix = True

    # Check for relative path in docstring
    path_found = False

    # Extract the package-relative path from the full path
    package_path = relative_path

    # Handle files in project root differently
    if os.path.basename(relative_path) == relative_path:
        # It's already just a filename with no directories, so use as is
        package_path = relative_path
    elif "/src/" in relative_path:
        package_path = relative_path.split("/src/")[-1]
    elif "/Users/" in relative_path:
        # Extract just the project path
        parts = relative_path.split("/")
        if "vibelint" in parts:
            idx = parts.index("vibelint")
            # Handle setup.py and other files in project root
            if idx + 1 >= len(parts) or parts[idx + 1] in [
                "setup.py",
                "README.md",
                "pyproject.toml",
            ]:
                package_path = parts[-1]  # Just use the filename
            elif idx + 1 < len(parts) and parts[idx + 1] == "src":
                package_path = "/".join(parts[idx + 2 :])
            else:
                package_path = "/".join(
                    parts[idx + 1 :]
                )  # Don't include vibelint itself

    # Check for either the full path or the package-relative path
    for line in docstring_lines:
        if relative_path in line or package_path in line:
            path_found = True
            break

    if not path_found:
        result.errors.append(
            f"Docstring should include the relative path: {package_path}"
        )
        result.needs_fix = True

    return result


def fix_module_docstring(
    content: str, result: ValidationResult, relative_path: str
) -> str:
    """
    Fix module docstring issues in a Python file.

    vibelint/validators/docstring.py
    """
    # Extract the package-relative path with the same logic as in validate_module_docstring
    package_path = relative_path

    # Handle files in project root differently
    if os.path.basename(relative_path) == relative_path:
        # It's already just a filename with no directories, so use as is
        package_path = relative_path
    elif "/src/" in relative_path:
        package_path = relative_path.split("/src/")[-1]
    elif "/Users/" in relative_path:
        parts = relative_path.split("/")
        if "vibelint" in parts:
            idx = parts.index("vibelint")
            # Handle setup.py and other files in project root
            if idx + 1 >= len(parts) or parts[idx + 1] in [
                "setup.py",
                "README.md",
                "pyproject.toml",
            ]:
                package_path = parts[-1]  # Just use the filename
            elif idx + 1 < len(parts) and parts[idx + 1] == "src":
                package_path = "/".join(parts[idx + 2 :])
            else:
                package_path = "/".join(
                    parts[idx + 1 :]
                )  # Don't include vibelint itself

    if not result.needs_fix:
        return content

    lines = content.splitlines()

    # If there's no docstring, create a new one
    if result.module_docstring is None:
        # Get the module name from the relative path
        module_name = os.path.basename(relative_path).replace(".py", "")

        # Create a docstring with preferred style (quotes on their own lines)
        docstring = [
            '"""',
            f"{module_name.replace('_', ' ').title()} module.",
            "",
            f"{package_path}",
            '"""',
        ]

        # Insert the docstring at the appropriate position
        for i, line in enumerate(docstring):
            lines.insert(result.line_number + i, line)
    else:
        # Modify the existing docstring
        existing_docstring = result.module_docstring.splitlines()

        # Fix the first line if needed
        if existing_docstring:
            first_content = existing_docstring[0]
            if not re.match(r"^[A-Z].+\.$", first_content):
                # Capitalize first letter and ensure it ends with a period
                if first_content:
                    first_content = first_content[0].upper() + first_content[1:]
                    if not first_content.endswith("."):
                        first_content += "."
                    existing_docstring[0] = first_content

        # Add relative path if missing
        path_found = False
        for i, line in enumerate(existing_docstring):
            if relative_path in line or package_path in line:
                path_found = True
                break

        if not path_found:
            # Add an empty line before the path if there isn't one already
            if existing_docstring and existing_docstring[-1]:
                existing_docstring.append("")
            existing_docstring.append(package_path)

        # Reconstruct the docstring
        docstring_text = "\n".join(existing_docstring)

        # Replace the old docstring
        start_idx = result.line_number
        end_idx = start_idx

        # Find the end of the old docstring
        in_docstring = False
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if line.startswith('"""') and not in_docstring:
                in_docstring = True
                if line.endswith('"""') and len(line) > 6:
                    # Single line docstring
                    end_idx = i
                    break
            elif (line == '"""' or line.endswith('"""')) and in_docstring:
                end_idx = i
                break

        # If it's a single-line docstring, convert to multi-line with preferred style
        new_docstring_lines = ['"""']
        docstring_lines = docstring_text.splitlines()
        new_docstring_lines.extend(docstring_lines)
        new_docstring_lines.append('"""')

        # Replace the old docstring lines with the new ones
        lines = lines[:start_idx] + new_docstring_lines + lines[end_idx + 1 :]

    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")
