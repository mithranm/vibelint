"""
Validator for Python module docstrings.

vibelint/validators/docstring.py
"""

import re
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


def validate_module_docstring(content: str, relative_path: str, docstring_regex: str) -> ValidationResult:
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
        if lines[i].strip().startswith('"""'):
            docstring_start = i
            # Single line docstring
            if lines[i].strip().endswith('"""') and len(lines[i].strip()) > 6:
                docstring_end = i
                docstring_lines = [lines[i].strip()[3:-3].strip()]
                break
            # Multi-line docstring
            for j in range(i + 1, len(lines)):
                if '"""' in lines[j]:
                    docstring_lines = [lines[i].strip()[3:].strip()]
                    docstring_lines.extend(line.strip() for line in lines[i+1:j])
                    if lines[j].strip().startswith('"""'):
                        # The closing triple quotes are on their own line
                        pass
                    else:
                        # The closing triple quotes are at the end of content
                        docstring_lines.append(lines[j].split('"""')[0].strip())
                    break
            break
            
    # If no docstring found
    if docstring_start is None:
        result.errors.append("Module docstring missing")
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
    if not re.match(docstring_regex, docstring_lines[0]):
        result.errors.append(
            f"First line of docstring should match regex: {docstring_regex}"
        )
        result.needs_fix = True
    
    # Check for relative path in docstring
    path_found = False
    for line in docstring_lines:
        if relative_path in line:
            path_found = True
            break
    
    if not path_found:
        result.errors.append(
            f"Docstring should include the relative path: {relative_path}"
        )
        result.needs_fix = True
    
    return result


def fix_module_docstring(content: str, result: ValidationResult, relative_path: str) -> str:
    """
    Fix module docstring issues in a Python file.

    vibelint/validators/docstring.py
    """
    if not result.needs_fix:
        return content
        
    lines = content.splitlines()
    
    # If there's no docstring, create a new one
    if result.module_docstring is None:
        # Get the module name from the relative path
        module_name = relative_path.split("/")[-1].replace(".py", "")
        
        # Create a docstring
        docstring = [
            '"""',
            f"{module_name.replace('_', ' ').title()} module.",
            "",
            f"{relative_path}",
            '"""'
        ]
        
        # Insert the docstring at the appropriate position
        for i, line in enumerate(docstring):
            lines.insert(result.line_number + i, line)
    else:
        # Modify the existing docstring
        existing_docstring = result.module_docstring.splitlines()
        
        # Fix the first line if needed
        if not re.match(r"^[A-Z].+\.$", existing_docstring[0]):
            # Capitalize first letter and ensure it ends with a period
            first_line = existing_docstring[0]
            if first_line:
                first_line = first_line[0].upper() + first_line[1:]
                if not first_line.endswith("."):
                    first_line += "."
                existing_docstring[0] = first_line
        
        # Add relative path if missing
        path_found = False
        for i, line in enumerate(existing_docstring):
            if relative_path in line:
                path_found = True
                break
        
        if not path_found:
            # Add an empty line before the path if there isn't one already
            if len(existing_docstring) > 1 and existing_docstring[-1]:
                existing_docstring.append("")
            existing_docstring.append(relative_path)
        
        # Reconstruct the docstring
        docstring_text = "\n".join(existing_docstring)
        
        # Replace the old docstring
        start_idx = result.line_number
        end_idx = start_idx
        
        # Find the end of the old docstring
        in_docstring = False
        for i in range(start_idx, len(lines)):
            if lines[i].strip().startswith('"""') and not in_docstring:
                in_docstring = True
                if lines[i].strip().endswith('"""') and len(lines[i].strip()) > 6:
                    # Single line docstring
                    end_idx = i
                    break
            elif '"""' in lines[i] and in_docstring:
                end_idx = i
                break
        
        # If it's a single-line docstring
        if start_idx == end_idx:
            lines[start_idx] = f'"""{docstring_text}"""'
        else:
            # Create a multi-line docstring
            docstring_lines = docstring_text.splitlines()
            if docstring_lines:
                # Combine the opening quotes with the first content line
                new_docstring_lines = [f'"""{docstring_lines[0]}']
                new_docstring_lines.extend(docstring_lines[1:])
            else:
                new_docstring_lines = ['"""']  # Empty docstring content
            
            # Replace the old docstring lines with the new ones
            lines = lines[:start_idx] + new_docstring_lines + lines[end_idx+1:]
    
    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")