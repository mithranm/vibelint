"""
Validator for Python docstrings at module/class/function/method level.

src/vibelint/validators/docstring.py
"""

import ast
from typing import List, Optional, Dict, Tuple, Any

MISSING_DOCSTRING = object()


class DocstringValidationResult:
    """
    Stores the result of docstring validation.

    src/vibelint/validators/docstring.py
    """
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.line_number: int = -1
        self.needs_fix: bool = False
        self.docstring_issues: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def has_issues(self) -> bool:
        return bool(self.errors or self.warnings or self.docstring_issues)


def get_normalized_filepath(relative_path: str) -> str:
    """
    Return a normalized path for docstring references:
    If file is in 'tests/' or 'src/', we return from there, else as-is.

    src/vibelint/validators/docstring.py
    """
    path = relative_path.replace("\\", "/")
    if "/tests/" in path:
        return path.split("/tests/", 1)[-1].rpartition("/")[0].join(["tests/", ""]) if "/tests/" in path else path
    if "/src/" in path:
        return path.split("/src/", 1)[-1]
    return relative_path


def extract_all_docstrings(content: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Parse the file content, get docstrings for module/class/function/method.

    src/vibelint/validators/docstring.py
    """
    results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return results

    # Module docstring
    if tree.body and isinstance(tree.body[0], ast.Expr):
        expr = tree.body[0]
        maybe_doc = None
        if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
            maybe_doc = expr.value.value
        elif isinstance(expr.value, ast.Str):
            maybe_doc = expr.value.s
        if maybe_doc is not None:
            lineno = expr.lineno
            end_lineno = lineno + len(maybe_doc.splitlines()) - 1
            results[(lineno, end_lineno)] = {"type": "module", "name": "module", "docstring": maybe_doc}
        else:
            results[(1, 1)] = {"type": "module", "name": "module", "docstring": None}
    else:
        results[(1, 1)] = {"type": "module", "name": "module", "docstring": None}

    def record_doc(node, node_type: str, parent_name: str = ""):
        doc_text = None
        if node.body and isinstance(node.body[0], ast.Expr):
            val = node.body[0].value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                doc_text = val.value
            elif isinstance(val, ast.Str):
                doc_text = val.s

        name = getattr(node, "name", node_type)
        if parent_name and node_type in ("method", "function"):
            name = f"{parent_name}.{name}"

        if doc_text is None:
            results[(node.lineno, node.lineno)] = {
                "type": node_type,
                "name": name,
                "docstring": None
            }
        else:
            lines = doc_text.splitlines()
            start_line = node.body[0].lineno
            end_line = start_line + len(lines) - 1
            results[(start_line, end_line)] = {
                "type": node_type,
                "name": name,
                "docstring": doc_text
            }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            record_doc(node, "class")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    record_doc(child, "method", node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node in tree.body:
                record_doc(node, "function")

    return results


def _docstring_includes_path(doc_lines: List[str], package_path: str) -> bool:
    """
    Return True if package_path is in any doc line.

    src/vibelint/validators/docstring.py
    """
    for ln in doc_lines:
        if package_path in ln:
            return True
    return False


def _split_docstring_lines(docstring: Optional[str]) -> List[str]:
    """
    Split docstring into stripped lines ignoring empties.

    src/vibelint/validators/docstring.py
    """
    if not docstring:
        return []
    return [x.strip() for x in docstring.splitlines() if x.strip()]


def validate_every_docstring(content: str, relative_path: str) -> DocstringValidationResult:
    """
    Validate that each docstring is present and includes the normalized file path.

    src/vibelint/validators/docstring.py
    """
    result = DocstringValidationResult()
    path_ref = get_normalized_filepath(relative_path)
    doc_map = extract_all_docstrings(content)
    if not doc_map:
        result.errors.append("No docstrings found in file (all missing?).")
        result.needs_fix = True
        return result

    for (start_line, end_line), info in doc_map.items():
        d_type = info["type"]
        d_name = info["name"]
        text = info["docstring"]
        lines = _split_docstring_lines(text)

        if text is None:
            msg = f"Missing docstring for {d_type} '{d_name}'."
            result.errors.append(msg)
            result.docstring_issues[(start_line, end_line)] = {
                "type": d_type,
                "name": d_name,
                "missing": True,
                "message": msg
            }
            result.needs_fix = True
        else:
            if not _docstring_includes_path(lines, path_ref):
                msg = f"Docstring for {d_type} '{d_name}' must include file path: {path_ref}"
                result.errors.append(msg)
                result.docstring_issues[(start_line, end_line)] = {
                    "type": d_type,
                    "name": d_name,
                    "missing": False,
                    "message": msg
                }
                result.needs_fix = True

    return result


def fix_every_docstring(content: str, result: DocstringValidationResult, relative_path: str) -> str:
    """
    Fix missing docstrings or missing path references.

    src/vibelint/validators/docstring.py
    """
    if not result.needs_fix:
        return content

    lines = content.split("\n")
    path_ref = get_normalized_filepath(relative_path)

    # Sort docstring issues from bottom to top
    doc_issues_sorted = sorted(
        result.docstring_issues.items(),
        key=lambda x: x[0][0],
        reverse=True
    )

    def create_block(indent: str, d_type: str, d_name: str) -> List[str]:
        block = []
        block.append(f'{indent}"""')
        block.append(f"{indent}Docstring for {d_type} '{d_name}'.")
        block.append(f"{indent}") # Add an empty line for spacing
        block.append(f"{indent}{path_ref}")
        block.append(f'{indent}"""')
        return block

    def fix_block(original_lines: List[str]) -> List[str]:
        if not original_lines:
            return original_lines
        first_line = original_lines[0].rstrip()
        if first_line.endswith('"""'):
            triple_quote = '"""'
        elif first_line.endswith("'''"):
            triple_quote = "'''"
        else:
            # Handle cases where docstring might not start with triple quotes (though unlikely from ast)
            # Or if the block passed is not actually a docstring
             return original_lines # Cannot reliably fix if not standard docstring format

        # indentation
        indent = ""
        for ch in first_line:
            if ch in (" ", "\t"):
                indent += ch
            else:
                break

        joined = "\n".join(original_lines).strip()
        if joined.startswith(triple_quote):
            joined = joined[len(triple_quote):]
        if joined.endswith(triple_quote):
            joined = joined[: -len(triple_quote)]

        # Split carefully, preserving relative indentation within the docstring
        lines_inner = joined.splitlines()
        # Strip leading/trailing empty lines and common indent from the original inner content
        while lines_inner and not lines_inner[0].strip():
            lines_inner.pop(0)
        while lines_inner and not lines_inner[-1].strip():
            lines_inner.pop(-1)

        # Remove old path references if they exist
        new_body_lines = [ln for ln in lines_inner if path_ref not in ln.strip()]

        # Add the required path reference, ensuring it's on its own line
        if path_ref not in [ln.strip() for ln in new_body_lines]:
             # Add an empty line before the path if the body isn't empty
            if new_body_lines and new_body_lines[-1].strip():
                 new_body_lines.append("")
            new_body_lines.append(path_ref)

        # Reconstruct the docstring block
        updated = [f"{indent}{triple_quote}"]
        for nb in new_body_lines:
             # Re-apply original indent + relative indent from original docstring line
             # This part is tricky without full dedent/reindent logic.
             # Simplification: just apply the base indent.
             updated.append(f"{indent}{nb.strip()}") # Apply base indent, strip original relative indent
        updated.append(f"{indent}{triple_quote}")
        return updated

    for ((start_line, end_line), info) in doc_issues_sorted:
        missing = info["missing"]
        d_type = info["type"]
        d_name = info["name"]

        if missing:
            # Find the actual end of the function/method signature (line ending with ':')
            def_line_idx = start_line - 1
            if def_line_idx < 0 or def_line_idx >= len(lines):
                # Log error or skip if definition line is out of bounds
                print(f"Warning: Could not find definition line for {d_name} at {start_line}")
                continue

            signature_end_idx = -1
            # Search from the 'def' line downwards
            for i in range(def_line_idx, len(lines)):
                # Check if line stripped of trailing whitespace ends with ':'
                # This handles comments after the colon as well
                if lines[i].rstrip().endswith(":"):
                    signature_end_idx = i
                    break

            if signature_end_idx == -1:
                # Log error or skip if signature end ':' is not found
                print(f"Warning: Could not find signature end for {d_name} starting at {start_line}")
                continue

            insertion_index = signature_end_idx + 1 # Insert *after* the signature line

            # Calculate indentation based on the 'def' or 'class' line
            def_line = lines[def_line_idx]
            base_indent = ""
            for ch in def_line:
                if ch in (" ", "\t"):
                    base_indent += ch
                else:
                    break
            # Standard Python indent is 4 spaces deeper than the definition line
            doc_indent = base_indent + "    "

            block = create_block(doc_indent, d_type, d_name)
            # Insert the new docstring block at the correct position
            lines = lines[:insertion_index] + block + lines[insertion_index:]
        else:
            # Fix existing docstring (add missing path)
            slice_start = start_line - 1
            slice_end = end_line # end_line from ast is the last line of the docstring
            if slice_start < 0 or slice_end > len(lines):
                 print(f"Warning: Invalid slice [{slice_start}:{slice_end}] for {d_name}")
                 continue

            existing_block = lines[slice_start:slice_end]
            fixed = fix_block(existing_block)
            # Replace the old block with the fixed one
            # Need to adjust line counts if fix_block changed the number of lines
            lines = lines[:slice_start] + fixed + lines[slice_end:]


    new_content = "\n".join(lines)
    # Preserve trailing newline if original content had one
    if content.endswith("\n") and not new_content.endswith("\n"):
        new_content += "\n"
    elif not content.endswith("\n") and new_content.endswith("\n"):
         # Avoid adding a newline if the original didn't have one (less common)
         new_content = new_content.rstrip("\n")

    return new_content
