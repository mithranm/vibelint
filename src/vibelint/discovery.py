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
import time
from pathlib import Path
from typing import List, Optional, Set

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
        logger.warning(f"Path {file_path_abs} is outside project root {project_root}. Excluding.")
        return True

    for pattern in exclude_globs:
        normalized_pattern = pattern.replace("\\", "/")
        if fnmatch.fnmatch(str(rel_path_str), normalized_pattern):
            logger.debug(f"Excluding '{rel_path_str}' due to pattern '{pattern}'")
            return True

    logger.debug(f"Path '{rel_path_str}' not excluded by any pattern.")
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
        raise ValueError("Cannot discover files without a project root defined in Config.")

    project_root = config.project_root.resolve()
    candidate_files: Set[Path] = set()
    _explicit_excludes = explicit_exclude_paths or set()

    # --- Load include/exclude globs ---
    include_globs_config = config.get("include_globs")
    if include_globs_config is None:
        if default_includes_if_missing is not None:
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
    elif not isinstance(include_globs_config, list):
        logger.error(
            f"Configuration error: 'include_globs' in pyproject.toml must be a list. "
            f"Found type {type(include_globs_config)}. No files will be included."
        )
        return []
    elif not include_globs_config:
        logger.warning(
            "Configuration: 'include_globs' is present but empty in pyproject.toml. "
            "No files will be included."
        )
        include_globs_effective = []
    else:
        include_globs_effective = include_globs_config

    normalized_includes = [p.replace("\\", "/") for p in include_globs_effective]

    exclude_globs = config.get("exclude_globs", [])
    if not isinstance(exclude_globs, list):
        logger.error(
            f"Configuration error: 'exclude_globs' in pyproject.toml must be a list. "
            f"Found type {type(exclude_globs)}. Ignoring exclusions."
        )
        exclude_globs = []
    normalized_exclude_globs = [p.replace("\\", "/") for p in exclude_globs]

    # --- Identify common exclude patterns for early checking ---
    _common_exclude_identifiers = [
        ".tox",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".venv",
        "venv",
        ".env",
        "env",
        "node_modules",
    ]
    common_exclude_patterns = [
        pat
        for pat in normalized_exclude_globs
        if any(
            f"/{ident}/" in f"/{pat}/" or pat.startswith(f"{ident}/") or pat == ident
            for ident in _common_exclude_identifiers
        )
    ]

    logger.debug(f"Starting file discovery from project root: {project_root}")
    logger.debug(f"Effective Include globs: {normalized_includes}")
    logger.debug(f"Exclude globs: {normalized_exclude_globs}")
    logger.debug(f"Explicit excludes: {_explicit_excludes}")
    logger.debug(f"Pre-checking against common exclude patterns: {common_exclude_patterns}")

    start_time = time.time()
    total_glob_yield_count = 0

    for pattern in normalized_includes:
        pattern_start_time = time.time()
        logger.debug(f"Processing include pattern: '{pattern}'")
        glob_method = project_root.rglob if "**" in pattern else project_root.glob
        pattern_yield_count = 0
        pattern_added_count = 0
        try:
            logger.debug(f"Running {glob_method.__name__}('{pattern}')...")
            for p in glob_method(pattern):
                pattern_yield_count += 1
                total_glob_yield_count += 1
                is_symlink = p.is_symlink()

                logger.debug(
                    f"  Glob yielded (from '{pattern}'): {p} (is_file: {p.is_file()}, is_dir: {p.is_dir()}, is_symlink: {is_symlink})"
                )

                if is_symlink:
                    logger.debug(f"    -> Skipping discovered symlink: {p}")
                    continue

                if common_exclude_patterns:
                    try:
                        rel_path_check = p.relative_to(project_root)
                        rel_path_check_str = str(rel_path_check).replace("\\", "/")

                        should_exclude_early = False
                        for exclude_pattern in common_exclude_patterns:
                            if fnmatch.fnmatch(rel_path_check_str, exclude_pattern):
                                logger.debug(
                                    f"    -> EARLY EXCLUDE: Skipping {p} due to pre-check against '{exclude_pattern}'"
                                )
                                should_exclude_early = True
                                break
                        if should_exclude_early:
                            continue

                    except ValueError:
                        logger.warning(
                            f"    -> EARLY SKIP: Path {p} outside project root {project_root}"
                        )
                        continue
                    except Exception as e_rel:
                        logger.error(
                            f"    -> ERROR getting relative path for early check on {p}: {e_rel}"
                        )
                        continue

                if p.is_file():
                    resolved_p = p.resolve()
                    logger.debug(
                        f"    -> Adding candidate: {resolved_p} (from pattern '{pattern}')"
                    )
                    candidate_files.add(resolved_p)
                    pattern_added_count += 1

        except PermissionError as e:
            logger.warning(
                f"Permission denied accessing path during glob for pattern '{pattern}': {e}. Skipping."
            )
        except Exception as e:
            logger.error(f"Error during glob execution for pattern '{pattern}': {e}", exc_info=True)

        pattern_time = time.time() - pattern_start_time
        logger.debug(
            f"Pattern '{pattern}' yielded {pattern_yield_count} paths, added {pattern_added_count} candidates in {pattern_time:.4f} seconds."
        )

    discovery_time = time.time() - start_time
    logger.debug(
        f"Initial globbing finished in {discovery_time:.4f} seconds. Total yielded paths: {total_glob_yield_count}. Total candidates: {len(candidate_files)}"
    )

    logger.debug(f"Applying *remaining* exclude rules to {len(candidate_files)} candidates...")
    final_files_set: Set[Path] = set()
    exclusion_start_time = time.time()

    sorted_candidates = sorted(list(candidate_files), key=lambda x: str(x))

    for file_path in sorted_candidates:
        if not _is_excluded(file_path, project_root, normalized_exclude_globs, _explicit_excludes):
            logger.debug(f"Including file: {file_path}")
            final_files_set.add(file_path)

    exclusion_time = time.time() - exclusion_start_time
    logger.debug(f"Exclusion phase finished in {exclusion_time:.4f} seconds.")

    discovered_files = final_files_set

    vcs_warnings: Set[Path] = set()
    potential_vcs_candidates = candidate_files
    if potential_vcs_candidates:
        for file_path in potential_vcs_candidates:
            is_in_vcs_dir = any(part in _VCS_DIRS for part in file_path.parts)
            if is_in_vcs_dir:
                if not _is_excluded(
                    file_path, project_root, normalized_exclude_globs, _explicit_excludes
                ):
                    if file_path in discovered_files:
                        vcs_warnings.add(file_path)

    final_count = len(discovered_files)

    if vcs_warnings:
        logger.warning(
            f"Found {len(vcs_warnings)} files within potential VCS directories "
            f"({', '.join(_VCS_DIRS)}) that were included because they were not "
            f"matched by any 'exclude_globs' pattern in pyproject.toml:"
        )
        sorted_warnings = sorted(list(vcs_warnings), key=lambda x: str(x))
        paths_to_log = []
        try:
            paths_to_log = [get_relative_path(p, project_root) for p in sorted_warnings[:5]]
        except ValueError:
            paths_to_log = [p for p in sorted_warnings[:5]]

        for rel_path_warn in paths_to_log:
            logger.warning(f"  - {rel_path_warn}")
        if len(vcs_warnings) > 5:
            logger.warning(f"  - ... and {len(vcs_warnings) - 5} more.")
        logger.warning(
            "Consider adding patterns like '.git/**' to 'exclude_globs' "
            "in your [tool.vibelint] section if this was unintended."
        )

    if final_count == 0 and len(candidate_files) > 0 and include_globs_effective:
        logger.warning("All candidate files were excluded. Check your exclude_globs patterns.")
    elif final_count == 0 and not include_globs_effective:
        pass
    elif final_count == 0:
        if include_globs_effective and total_glob_yield_count == 0:
            logger.warning("No files found matching include_globs patterns.")

    logger.debug(f"Discovery complete. Returning {len(discovered_files)} files.")
    return sorted(list(discovered_files))
