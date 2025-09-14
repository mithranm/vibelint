# vibelint/src/vibelint/snapshot.py
"""
Codebase snapshot generation in markdown format.

vibelint/src/vibelint/snapshot.py
"""

import fnmatch
import logging
from pathlib import Path

from .config import Config
from .discovery import discover_files
from .utils import get_relative_path, is_binary

__all__ = ["create_snapshot"]

logger = logging.getLogger(__name__)


def create_snapshot(
    output_path: Path,
    target_paths: list[Path],
    config: Config,
):
    """
    Creates a Markdown snapshot file containing the project structure and file contents,
    respecting the include/exclude rules defined in pyproject.toml.

    Args:
    output_path: The path where the Markdown file will be saved.
    target_paths: List of initial paths (files or directories) to discover from.
    config: The vibelint configuration object.

    vibelint/src/vibelint/snapshot.py
    """

    assert config.project_root is not None, "Project root must be set before creating snapshot."
    project_root = config.project_root.resolve()

    absolute_output_path = output_path.resolve()

    logger.debug("create_snapshot: Running discovery based on pyproject.toml config...")

    discovered_files = discover_files(
        paths=target_paths,
        config=config,
        explicit_exclude_paths={absolute_output_path},
    )

    logger.debug(f"create_snapshot: Discovery finished, count: {len(discovered_files)}")

    # Debugging check (can be removed later)
    for excluded_pattern_root in [".pytest_cache", ".ruff_cache", ".git"]:
        present = any(excluded_pattern_root in str(f) for f in discovered_files)
        logger.debug(
            f"!!! Check @ start of create_snapshot: '{excluded_pattern_root}' presence in list: {present}"
        )

    file_infos: list[tuple[Path, str]] = []

    peek_globs = config.get("peek_globs", [])
    if not isinstance(peek_globs, list):
        logger.warning("Configuration 'peek_globs' is not a list. Ignoring peek rules.")
        peek_globs = []

    for abs_file_path in discovered_files:
        try:
            rel_path_obj = get_relative_path(abs_file_path, project_root)
            rel_path_str = str(rel_path_obj)  # Still useful for fnmatch below
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
                normalized_rel_path = rel_path_str.replace("\\", "/")
                normalized_peek_glob = pk.replace("\\", "/")
                if fnmatch.fnmatch(normalized_rel_path, normalized_peek_glob):
                    cat = "PEEK"
                    break
        file_infos.append((abs_file_path, cat))
        logger.debug(f"Categorized {rel_path_str} as {cat}")

    file_infos.sort(key=lambda x: x[0])

    logger.debug(f"Sorted {len(file_infos)} files for snapshot.")

    # Build the tree structure using a dictionary
    tree: dict = {}
    for f_path, f_cat in file_infos:
        try:
            # --- FIX START ---
            # Get the relative path object
            relative_path_obj = get_relative_path(f_path, project_root)
            # Use the .parts attribute which is OS-independent
            relative_parts = relative_path_obj.parts
            # --- FIX END ---
        except ValueError:
            # Handle files outside the project root if they somehow got here
            logger.warning(
                f"Skipping file outside project root during snapshot tree build: {f_path}"
            )
            continue

        node = tree
        # Iterate through the path components tuple
        for i, part in enumerate(relative_parts):
            # Skip empty parts if any somehow occur (unlikely with .parts)
            if not part:
                continue

            is_last_part = i == len(relative_parts) - 1

            if is_last_part:
                # This is the filename part
                if "__FILES__" not in node:
                    node["__FILES__"] = []
                # Add the tuple (absolute path, category)
                node["__FILES__"].append((f_path, f_cat))
            else:
                # This is a directory part
                if part not in node:
                    node[part] = {}  # Create a new dictionary for the subdirectory
                # Move deeper into the tree structure
                node = node[part]

    logger.info(f"Writing snapshot to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(absolute_output_path, "w", encoding="utf-8") as outfile:

            outfile.write("# Snapshot\n\n")

            # Write Filesystem Tree section
            outfile.write("## Filesystem Tree\n\n```\n")
            # Use project root name for the tree root display
            tree_root_name = project_root.name if project_root.name else str(project_root)
            outfile.write(f"{tree_root_name}/\n")
            _write_tree(outfile, tree, "")  # Pass the populated tree dictionary
            outfile.write("```\n\n")

            # Write File Contents section
            outfile.write("## File Contents\n\n")
            outfile.write("Files are ordered alphabetically by path.\n\n")
            for f, cat in file_infos:  # Iterate through the sorted list again
                try:
                    relpath_header = get_relative_path(f, project_root)
                    outfile.write(f"### File: {relpath_header}\n\n")
                    logger.debug(f"Writing content for {relpath_header} (Category: {cat})")

                    if cat == "BINARY":
                        outfile.write("```\n")
                        outfile.write("[Binary File - Content not displayed]\n")
                        outfile.write("```\n\n---\n")
                    elif cat == "PEEK":
                        outfile.write("```\n")
                        outfile.write("[PEEK - Content truncated]\n")
                        try:
                            with open(f, encoding="utf-8", errors="ignore") as infile:
                                lines_read = 0
                                for line in infile:
                                    if lines_read >= 10:  # Peek limit (e.g., 10 lines)
                                        outfile.write("...\n")
                                        break
                                    outfile.write(line)
                                    lines_read += 1
                        except Exception as e:
                            logger.warning(f"Error reading file for peek {relpath_header}: {e}")
                            outfile.write(f"[Error reading file for peek: {e}]\n")
                        outfile.write("```\n\n---\n")
                    else:  # cat == "FULL"
                        lang = _get_language(f)
                        outfile.write(f"```{lang}\n")
                        try:
                            with open(f, encoding="utf-8", errors="ignore") as infile:
                                content = infile.read()
                                # Ensure final newline for cleaner markdown rendering
                                if not content.endswith("\n"):
                                    content += "\n"
                                outfile.write(content)
                        except Exception as e:
                            logger.warning(f"Error reading file content {relpath_header}: {e}")
                            outfile.write(f"[Error reading file: {e}]\n")
                        outfile.write("```\n\n---\n")

                except Exception as e:
                    # General error handling for processing a single file entry
                    try:
                        relpath_header_err = get_relative_path(f, project_root)
                    except Exception:
                        relpath_header_err = str(f)  # Fallback to absolute path if rel path fails

                    logger.error(
                        f"Error processing file entry for {relpath_header_err} in snapshot: {e}",
                        exc_info=True,
                    )
                    outfile.write(f"### File: {relpath_header_err} (Error)\n\n")
                    outfile.write(f"[Error processing file entry: {e}]\n\n---\n")

            # Add a final newline for good measure
            outfile.write("\n")

    except OSError as e:
        # Error writing the main output file
        logger.error(f"Failed to write snapshot file {absolute_output_path}: {e}", exc_info=True)
        raise  # Re-raise IOErrors
    except Exception as e:
        # Catch-all for other unexpected errors during writing
        logger.error(f"An unexpected error occurred during snapshot writing: {e}", exc_info=True)
        raise  # Re-raise other critical exceptions


def _write_tree(outfile, node: dict, prefix=""):
    """
    Helper function to recursively write the directory tree structure
    from the prepared dictionary.

    Args:
        outfile: The file object to write to.
        node: The current dictionary node representing a directory.
        prefix: The string prefix for drawing tree lines.

    vibelint/src/vibelint/snapshot.py
    """
    # Separate directories (keys other than '__FILES__') from files (items in '__FILES__')
    dirs = sorted([k for k in node if k != "__FILES__"])
    files_data: list[tuple[Path, str]] = sorted(node.get("__FILES__", []), key=lambda x: x[0].name)

    # Combine directory names and file names for iteration order
    entries = dirs + [f_info[0].name for f_info in files_data]

    for i, name in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        outfile.write(f"{prefix}{connector}")

        if name in dirs:
            # It's a directory - write its name and recurse
            outfile.write(f"{name}/\n")
            new_prefix = prefix + ("    " if is_last else "│   ")
            _write_tree(outfile, node[name], new_prefix)  # Recurse into the sub-dictionary
        else:
            # It's a file - find its category and write name with indicators
            file_info_tuple = next((info for info in files_data if info[0].name == name), None)
            file_cat = "FULL"  # Default category
            if file_info_tuple:
                file_cat = file_info_tuple[1]  # Get category ('FULL', 'PEEK', 'BINARY')

            # Add indicators for non-full content files
            peek_indicator = " (PEEK)" if file_cat == "PEEK" else ""
            binary_indicator = " (BINARY)" if file_cat == "BINARY" else ""
            outfile.write(f"{name}{peek_indicator}{binary_indicator}\n")


def _get_language(file_path: Path) -> str:
    """
    Guess language for syntax highlighting based on extension.
    Returns an empty string if no specific language is known.

    Args:
        file_path: The path to the file.

    Returns:
        A string representing the language identifier for markdown code blocks,
        or an empty string.

    vibelint/src/vibelint/snapshot.py
    """
    ext = file_path.suffix.lower()
    # Mapping from file extension to markdown language identifier
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
        ".env": "bash",  # Treat .env like bash for highlighting often
        ".tf": "terraform",
        ".hcl": "terraform",
        ".lua": "lua",
        ".perl": "perl",
        ".pl": "perl",
        ".r": "r",
        ".ex": "elixir",
        ".exs": "elixir",
        ".dart": "dart",
        ".groovy": "groovy",
        ".gradle": "groovy",  # Gradle files often use groovy
        ".vb": "vbnet",
        ".fs": "fsharp",
        ".fsi": "fsharp",
        ".fsx": "fsharp",
        ".fsscript": "fsharp",
    }
    return mapping.get(ext, "")  # Return the mapped language or empty string
