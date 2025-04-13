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
