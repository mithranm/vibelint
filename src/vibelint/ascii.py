"""
ASCII Art Scaling and Loading Utilities
This module provides functions to scale ASCII art to fit within the terminal's
dimensions while preserving the original aspect ratio. It also includes a function
to load ASCII art from a file.

src/vibelint/ascii.py
"""

import shutil


def _get_terminal_size():
    """
    Returns the terminal size as a tuple (width, height) of characters.
    Falls back to (80, 24) if the dimensions cannot be determined.

    src/vibelint/ascii.py
    """
    try:
        size = shutil.get_terminal_size(fallback=(80, 24))
        return size.columns, size.lines
    except Exception:
        return 80, 24


def scale_ascii_art_by_height(ascii_art: str, target_height: int) -> str:
    """
    Scales the ASCII art to have a specified target height (in characters)
    while preserving the original aspect ratio. The target width is
    automatically computed based on the scaling factor.

    src/vibelint/ascii.py
    """
    # Split into lines and remove any fully blank lines.
    lines = [line for line in ascii_art.splitlines() if line.strip()]
    if not lines:
        return ""

    orig_height = len(lines)
    orig_width = max(len(line) for line in lines)

    # Pad all lines to the same length (for a rectangular grid)
    normalized_lines = [line.ljust(orig_width) for line in lines]

    # Compute the vertical scale factor and derive the target width.
    scale_factor = target_height / orig_height
    target_width = max(1, int(orig_width * scale_factor))

    # Calculate step sizes for sampling
    row_step = orig_height / target_height
    col_step = orig_width / target_width if target_width > 0 else 1

    result_lines = []
    for r in range(target_height):
        orig_r = min(int(r * row_step), orig_height - 1)
        new_line = []
        for c in range(target_width):
            orig_c = min(int(c * col_step), orig_width - 1)
            new_line.append(normalized_lines[orig_r][orig_c])
        result_lines.append("".join(new_line))

    return "\n".join(result_lines)


def scale_to_terminal_by_height(ascii_art: str) -> str:
    """
    Scales the provided ASCII art to fit based on the terminal's available height.
    The width is computed automatically to maintain the art's original aspect ratio.

    src/vibelint/ascii.py
    """
    _, term_height = _get_terminal_size()
    # Optionally, leave a margin (here, using 90% of available height)
    target_height = max(1, int(term_height * 0.9))
    return scale_ascii_art_by_height(ascii_art, target_height)


def load_ascii_from_file(filepath: str) -> str:
    """
    Reads an ASCII art file from disk and returns its content as a string.
    Using a file is generally wise because it separates your art data from your code,
    making it easier to update without modifying the code.

    src/vibelint/ascii.py
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading {filepath}: {e}")


# Public API
__all__ = ["scale_ascii_art_by_height", "scale_to_terminal_by_height", "load_ascii_from_file"]
