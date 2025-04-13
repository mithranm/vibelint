"""
Codebase snapshot generation in markdown format.

vibelint/snapshot.py
"""

import fnmatch
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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
            tree_root_name = project_root.name if project_root.name else str(project_root)
            outfile.write(f"{tree_root_name}/\n")
            _write_tree(outfile, tree, "")
            outfile.write("```\n\n")

            outfile.write("## File Contents\n\n")
            outfile.write("Files are ordered alphabetically by path.\n\n")
            for f, cat in file_infos:
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
                            with open(f, "r", encoding="utf-8", errors="ignore") as infile:
                                lines_read = 0
                                for line in infile:
                                    if lines_read >= 10:
                                        outfile.write("...\n")
                                        break
                                    outfile.write(line)
                                    lines_read += 1
                        except Exception as e:
                            logger.warning(f"Error reading file for peek {relpath_header}: {e}")
                            outfile.write(f"[Error reading file for peek: {e}]\n")
                        outfile.write("```\n\n---\n")
                    else:
                        lang = _get_language(f)
                        outfile.write(f"```{lang}\n")
                        try:
                            with open(f, "r", encoding="utf-8", errors="ignore") as infile:
                                content = infile.read()
                                if not content.endswith("\n"):
                                    content += "\n"
                                outfile.write(content)
                        except Exception as e:
                            logger.warning(f"Error reading file content {relpath_header}: {e}")
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

        logger.error(f"Failed to write snapshot file {absolute_output_path}: {e}", exc_info=True)
        raise
    except Exception as e:

        logger.error(f"An unexpected error occurred during snapshot writing: {e}", exc_info=True)
        raise


def _write_tree(outfile, node: Dict, prefix=""):
    """
    Helper function to recursively write the directory tree.

    vibelint/snapshot.py
    """

    dirs = sorted([k for k in node if k != "__FILES__"])

    files_data: List[Tuple[Path, str]] = sorted(node.get("__FILES__", []), key=lambda x: x[0].name)

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

            file_info_tuple = next((info for info in files_data if info[0].name == name), None)
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
