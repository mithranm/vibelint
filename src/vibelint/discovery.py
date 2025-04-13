"""
Discovers files using pathlib glob/rglob based on include patterns from
pyproject.toml, respecting the pattern's implied scope, then filters
using exclude patterns.

If `include_globs` is missing from the configuration:
- If `default_includes_if_missing` is provided, uses those patterns and logs a warning.
- Otherwise, logs an error and returns an empty list.

Exclusions from `config.exclude_globs` are always applied. Explicitly
provided paths are also excluded.

Warns if files within common VCS directories (.git, .hg, .svn) are found
and not covered by exclude_globs.

src/vibelint/discovery.py
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

_VCS_DIRS = {".git", ".hg", ".svn"}  # Keep this if needed for VCS warnings later


def _is_excluded(  # Keep this helper function as is
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
        # Use resolve() for consistent comparison base
        rel_path = file_path_abs.resolve().relative_to(project_root.resolve())
        rel_path_str = str(rel_path).replace("\\", "/")  # Normalize for fnmatch
    except ValueError:
        logger.warning(f"Path {file_path_abs} is outside project root {project_root}. Excluding.")
        return True
    except Exception as e:
        logger.error(f"Error getting relative path for exclusion check on {file_path_abs}: {e}")
        return True  # Exclude if relative path fails

    for pattern in exclude_globs:
        normalized_pattern = pattern.replace("\\", "/")
        # Use fnmatch which matches based on Unix glob rules
        if fnmatch.fnmatch(rel_path_str, normalized_pattern):
            logger.debug(f"Excluding '{rel_path_str}' due to exclude pattern '{pattern}'")
            return True
        # Also check if the pattern matches any parent directory part
        # Example: exclude "build/" should match "build/lib/module.py"
        if "/" in normalized_pattern.rstrip("/"):  # Check if it's a directory pattern
            dir_pattern = normalized_pattern.rstrip("/") + "/"
            if rel_path_str.startswith(dir_pattern):
                logger.debug(
                    f"Excluding '{rel_path_str}' due to directory exclude pattern '{dir_pattern}'"
                )
                return True

    # logger.debug(f"Path '{rel_path_str}' not excluded by any pattern.") # Too verbose
    return False


def discover_files(
    paths: List[Path],  # Note: This argument seems unused for the main globbing logic
    config: Config,
    default_includes_if_missing: Optional[List[str]] = None,
    explicit_exclude_paths: Optional[Set[Path]] = None,
) -> List[Path]:
    """
    Discovers files using pathlib glob/rglob based on include patterns from
    pyproject.toml, respecting the pattern's implied scope, then filters
    using exclude patterns.

    If `include_globs` is missing from the configuration:
    - If `default_includes_if_missing` is provided, uses those patterns and logs a warning.
    - Otherwise, logs an error and returns an empty list.

    Exclusions from `config.exclude_globs` are always applied. Explicitly
    provided paths are also excluded.

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
    _explicit_excludes = {
        p.resolve() for p in (explicit_exclude_paths or set())
    }  # Resolve explicit excludes

    # --- Load include/exclude globs (Same as before) ---
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
                "of pyproject.toml. No include patterns specified. Add 'include_globs' "
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

    exclude_globs_config = config.get("exclude_globs", [])
    if not isinstance(exclude_globs_config, list):
        logger.error(
            f"Configuration error: 'exclude_globs' in pyproject.toml must be a list. "
            f"Found type {type(exclude_globs_config)}. Ignoring exclusions."
        )
        exclude_globs_effective = []
    else:
        exclude_globs_effective = exclude_globs_config
    normalized_exclude_globs = [p.replace("\\", "/") for p in exclude_globs_effective]

    logger.debug(f"Starting file discovery from project root: {project_root}")
    logger.debug(f"Effective Include globs: {normalized_includes}")
    logger.debug(f"Exclude globs: {normalized_exclude_globs}")
    logger.debug(f"Explicit excludes: {_explicit_excludes}")

    start_time = time.time()
    total_glob_yield_count = 0

    for pattern in normalized_includes:
        pattern_start_time = time.time()
        logger.debug(f"Processing include pattern: '{pattern}'")
        glob_method = project_root.rglob if "**" in pattern else project_root.glob
        pattern_yield_count = 0
        pattern_added_count = 0

        # --- Determine expected base directory for anchored patterns ---
        expected_base_dir: Optional[Path] = None
        pattern_path = Path(pattern)
        # Check if the first part of the pattern is a simple directory name (not a wildcard)
        # and the pattern contains a separator (implying it's not just a root file pattern)
        if (
            not pattern_path.is_absolute()
            and pattern_path.parts
            and not any(c in pattern_path.parts[0] for c in "*?[]")
            and ("/" in pattern or "\\" in pattern)  # Check if it contains a path separator
        ):
            expected_base_dir = project_root / pattern_path.parts[0]
            logger.debug(f"Pattern '{pattern}' implies base directory: {expected_base_dir}")

        try:
            logger.debug(f"Running {glob_method.__name__}('{pattern}')...")
            for p in glob_method(pattern):
                pattern_yield_count += 1
                total_glob_yield_count += 1
                abs_p = p.resolve()  # Resolve once

                logger.debug(
                    f"  Glob yielded (from '{pattern}'): {abs_p} (orig: {p}, is_file: {p.is_file()})"
                )

                # --- <<< FIX: Post-Glob Validation >>> ---
                is_valid_for_pattern = True  # Assume valid unless proven otherwise
                if expected_base_dir:
                    # If pattern implies a base dir (e.g., "src/..."), check path is relative to it
                    try:
                        # Use resolved paths for reliable check
                        abs_p.relative_to(expected_base_dir.resolve())
                    except ValueError:
                        logger.debug(
                            f"    -> Skipping {abs_p}: Yielded by anchored pattern '{pattern}' but not relative to expected base {expected_base_dir}"
                        )
                        is_valid_for_pattern = False
                    except Exception as path_err:
                        logger.warning(
                            f"    -> Error checking relative path for {abs_p} against {expected_base_dir}: {path_err}. Allowing through."
                        )
                elif (
                    "/" not in pattern
                    and "\\" not in pattern
                    and not any(c in pattern for c in "*?[]")
                ):
                    # If pattern is a simple filename (e.g., "pyproject.toml"), check it's directly under root
                    if abs_p.parent != project_root:
                        logger.debug(
                            f"    -> Skipping {abs_p}: Yielded by root pattern '{pattern}' but not in project root directory."
                        )
                        is_valid_for_pattern = False

                if not is_valid_for_pattern:
                    continue  # Skip this path if it didn't belong to the pattern's scope
                # --- <<< END FIX >>> ---

                if p.is_symlink():
                    logger.debug(f"    -> Skipping discovered symlink: {p}")
                    continue

                # We only care about files from here on for candidacy
                if p.is_file():
                    # Add the *resolved* absolute path to the candidates
                    logger.debug(f"    -> Adding candidate: {abs_p} (from pattern '{pattern}')")
                    candidate_files.add(abs_p)
                    pattern_added_count += 1
                # else: # Log directories yielded if needed for debugging
                #      logger.debug(f"    -> Ignoring directory yielded by glob: {p}")

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

    logger.debug(f"Applying exclude rules to {len(candidate_files)} candidates...")
    final_files_set: Set[Path] = set()
    exclusion_start_time = time.time()

    # Sort candidates for deterministic processing order (optional but good)
    sorted_candidates = sorted(list(candidate_files), key=str)

    for file_path_abs in sorted_candidates:
        if not _is_excluded(
            file_path_abs, project_root, normalized_exclude_globs, _explicit_excludes
        ):
            logger.debug(f"Including file: {file_path_abs}")
            final_files_set.add(file_path_abs)
        # else: # No need to log every exclusion unless debugging excludes specifically
        # try:
        #     rel_path_exc = get_relative_path(file_path_abs, project_root)
        #     logger.debug(f"Excluding file based on rules: {rel_path_exc}")
        # except ValueError:
        #      logger.debug(f"Excluding file based on rules: {file_path_abs}")

    exclusion_time = time.time() - exclusion_start_time
    logger.debug(f"Exclusion phase finished in {exclusion_time:.4f} seconds.")

    # --- VCS Warning Logic (Optional, keep if desired) ---
    # ... (keep the existing VCS warning logic if you want it) ...
    vcs_warnings: Set[Path] = set()
    if final_files_set:  # Check against final included files
        for file_path in final_files_set:
            try:
                is_in_vcs_dir = any(
                    part in _VCS_DIRS for part in file_path.relative_to(project_root).parts
                )
                if is_in_vcs_dir:
                    # Check if it *would* have been excluded if a pattern existed
                    # This check is slightly complex - maybe simplify the warning?
                    # For now, let's just warn if *any* included file is in a dir matching VCS name parts
                    vcs_warnings.add(file_path)
            except ValueError:  # Outside project root
                pass
            except Exception as e_vcs:
                logger.debug(f"Error during VCS check for {file_path}: {e_vcs}")

    if vcs_warnings:
        logger.warning(
            f"Found {len(vcs_warnings)} included files within potential VCS directories "
            f"({', '.join(_VCS_DIRS)}). Consider adding patterns like '.git/**' to 'exclude_globs' "
            "in your [tool.vibelint] section if this was unintended."
        )
        # Log first few examples
        try:
            paths_to_log = [
                get_relative_path(p, project_root) for p in sorted(list(vcs_warnings), key=str)[:5]
            ]
            for rel_path_warn in paths_to_log:
                logger.warning(f"  - {rel_path_warn}")
            if len(vcs_warnings) > 5:
                logger.warning(f"  - ... and {len(vcs_warnings) - 5} more.")
        except ValueError:
            logger.warning("  (Could not display example relative paths - outside project root?)")
        except Exception as e_log:
            logger.warning(f"  (Error logging example paths: {e_log})")

    # --- Final Count Logging (Same as before) ---
    final_count = len(final_files_set)
    if final_count == 0 and len(candidate_files) > 0 and include_globs_effective:
        logger.warning("All candidate files were excluded. Check your exclude_globs patterns.")
    elif final_count == 0 and not include_globs_effective:
        pass  # Expected if includes are empty
    elif final_count == 0:
        if include_globs_effective and total_glob_yield_count == 0:
            logger.warning("No files found matching include_globs patterns.")

    logger.debug(f"Discovery complete. Returning {final_count} files.")
    return sorted(list(final_files_set))
