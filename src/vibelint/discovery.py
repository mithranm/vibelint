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

vibelint/src/vibelint/discovery.py
"""

import fnmatch
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

from .config import Config
from .utils import get_relative_path

__all__ = ["discover_files", "discover_files_from_paths"]
logger = logging.getLogger(__name__)

_VCS_DIRS = {".git", ".hg", ".svn"}

# Default exclude patterns to avoid __pycache__, .git, etc. when using custom paths
_DEFAULT_EXCLUDE_PATTERNS = [
    "__pycache__/**",
    "*.pyc",
    "*.pyo",
    ".git/**",
    ".hg/**",
    ".svn/**",
    ".pytest_cache/**",
    ".coverage",
    ".mypy_cache/**",
    ".tox/**",
    "venv/**",
    ".venv/**",
    "env/**",
    ".env/**",
    "node_modules/**",
    ".DS_Store",
    "*.egg-info/**",
]


def _is_excluded(
    path_abs: Path,
    project_root: Path,
    exclude_globs: list[str],
    explicit_exclude_paths: set[Path],
    is_checking_directory_for_prune: bool = False,
) -> bool:
    """
    Checks if a discovered path (file or directory) should be excluded.

    For files: checks explicit paths first, then exclude globs.
    For directories (for pruning): checks if the directory itself matches an exclude glob.

    Args:
    path_abs: The absolute path of the file or directory to check.
    project_root: The absolute path of the project root.
    exclude_globs: List of glob patterns for exclusion from config.
    explicit_exclude_paths: Set of absolute paths to exclude explicitly (applies to files).
    is_checking_directory_for_prune: True if checking a directory for os.walk pruning.

    Returns:
    True if the path should be excluded/pruned, False otherwise.

    vibelint/src/vibelint/discovery.py
    """

    if not is_checking_directory_for_prune and path_abs in explicit_exclude_paths:
        logger.debug(f"Excluding explicitly provided path: {path_abs}")
        return True

    try:
        # Use resolve() for consistent comparison base
        rel_path = path_abs.resolve().relative_to(project_root.resolve())
        # Normalize for fnmatch and consistent comparisons
        rel_path_str = str(rel_path).replace("\\", "/")
    except ValueError:
        # Path is outside project root, consider it excluded for safety
        logger.warning(f"Path {path_abs} is outside project root {project_root}. Excluding.")
        return True
    except (OSError, TypeError) as e:
        logger.error(f"Error getting relative path for exclusion check on {path_abs}: {e}")
        return True  # Exclude if relative path fails

    for pattern in exclude_globs:
        normalized_pattern = pattern.replace("\\", "/")

        if is_checking_directory_for_prune:
            # Logic for pruning directories:
            # 1. Exact match: pattern "foo", rel_path_str "foo" (dir name)
            if fnmatch.fnmatch(rel_path_str, normalized_pattern):
                logger.debug(
                    f"Pruning dir '{rel_path_str}' due to direct match with exclude pattern '{pattern}'"
                )
                return True
            # 2. Dir pattern like "foo/" or "foo/**":
            #    pattern "build/", rel_path_str "build" -> match
            #    pattern "build/**", rel_path_str "build" -> match
            if normalized_pattern.endswith("/"):
                if rel_path_str == normalized_pattern[:-1]:  # pattern "build/", rel_path "build"
                    logger.debug(
                        f"Pruning dir '{rel_path_str}' due to match with dir pattern '{pattern}'"
                    )
                    return True
            elif normalized_pattern.endswith("/**"):
                if (
                    rel_path_str == normalized_pattern[:-3]
                ):  # e.g. pattern 'dir/**', rel_path_str 'dir'
                    logger.debug(
                        f"Pruning dir '{rel_path_str}' due to match with dir/** pattern '{pattern}'"
                    )
                    return True
        else:
            # Logic for excluding files:
            # Rule 1: File path matches the glob pattern directly
            # This handles patterns like "*.pyc", "temp/*", "specific_file.txt"
            if fnmatch.fnmatch(rel_path_str, normalized_pattern):
                logger.debug(f"Excluding file '{rel_path_str}' due to exclude pattern '{pattern}'")
                return True

            # Rule 2: File is within a directory excluded by a pattern ending with '/' or '/**'
            # e.g., exclude_glob is "build/", file is "build/lib/module.py"
            # e.g., exclude_glob is "output/**", file is "output/data/log.txt"
            if normalized_pattern.endswith("/"):  # Pattern "build/"
                if rel_path_str.startswith(normalized_pattern):
                    logger.debug(
                        f"Excluding file '{rel_path_str}' because it's in excluded dir prefix '{normalized_pattern}'"
                    )
                    return True
            elif normalized_pattern.endswith("/**"):  # Pattern "build/**"
                # For "build/**", we want to match files starting with "build/"
                base_dir_pattern = normalized_pattern[:-2]  # Results in "build/"
                if rel_path_str.startswith(base_dir_pattern):
                    logger.debug(
                        f"Excluding file '{rel_path_str}' because it's in excluded dir prefix '{normalized_pattern}'"
                    )
                    return True
            # Note: A simple exclude pattern like "build" (without / or **) for files
            # will only match a file *named* "build" via the fnmatch rule above.
            # To exclude all contents of a directory "build", the pattern should be
            # "build/" or "build/**". The pruning logic for directories handles these
            # patterns effectively for `os.walk`.

    return False


def _recursive_glob_with_pruning(
    search_root_abs: Path,
    glob_suffix_pattern: str,  # e.g., "*.py" or "data/*.json"
    project_root: Path,
    config_exclude_globs: list[str],
    explicit_exclude_paths: set[Path],
) -> Iterator[Path]:
    """
    Recursively walks a directory, prunes excluded subdirectories, and yields files
    matching the glob_suffix_pattern that are not otherwise excluded.

    Args:
        search_root_abs: Absolute path to the directory to start the search from.
        glob_suffix_pattern: The glob pattern to match files against (relative to directories in the walk).
        project_root: Absolute path of the project root.
        config_exclude_globs: List of exclude glob patterns from config.
        explicit_exclude_paths: Set of absolute file paths to explicitly exclude.

    Yields:
        Absolute Path objects for matching files.
    """
    logger.debug(
        f"Recursive walk starting at '{search_root_abs}' for pattern '.../{glob_suffix_pattern}'"
    )
    for root_str, dir_names, file_names in os.walk(str(search_root_abs), topdown=True):
        current_dir_abs = Path(root_str)

        # Prune directories
        original_dir_count = len(dir_names)
        dir_names[:] = [
            d_name
            for d_name in dir_names
            if not _is_excluded(
                current_dir_abs / d_name,
                project_root,
                config_exclude_globs,
                explicit_exclude_paths,  # Not used for dir pruning but passed for func signature
                is_checking_directory_for_prune=True,
            )
        ]
        if len(dir_names) < original_dir_count:
            logger.debug(
                f"Pruned {original_dir_count - len(dir_names)} subdirectories under {current_dir_abs}"
            )

        # Match files in the current (potentially non-pruned) directory
        for f_name in file_names:
            file_abs = current_dir_abs / f_name

            # Path of file relative to where the glob_suffix_pattern matching should start (search_root_abs)
            try:
                rel_to_search_root = file_abs.relative_to(search_root_abs)
            except ValueError:
                # Should not happen if os.walk starts at search_root_abs and yields descendants
                logger.warning(
                    f"File {file_abs} unexpectedly not relative to search root {search_root_abs}. Skipping."
                )
                continue

            normalized_rel_to_search_root_str = str(rel_to_search_root).replace("\\", "/")

            if fnmatch.fnmatch(normalized_rel_to_search_root_str, glob_suffix_pattern):
                # File matches the include pattern's suffix.
                # Now, perform a final check against global exclude rules for this specific file.
                if not _is_excluded(
                    file_abs,
                    project_root,
                    config_exclude_globs,
                    explicit_exclude_paths,
                    is_checking_directory_for_prune=False,
                ):
                    yield file_abs.resolve()  # Yield resolved path


def discover_files(
    paths: list[Path],
    config: Config,
    default_includes_if_missing: list[str] | None = None,
    explicit_exclude_paths: set[Path] | None = None,
) -> list[Path]:
    """
    Discovers files based on include/exclude patterns from configuration.
    Uses a custom walker for recursive globs (**) to enable directory pruning.

    Args:
    paths: Initial paths (largely ignored, globs operate from project root).
    config: The vibelint configuration object (must have project_root set).
    default_includes_if_missing: Fallback include patterns if 'include_globs' is not in config.
    explicit_exclude_paths: A set of absolute file paths to explicitly exclude.

    Returns:
    A sorted list of unique absolute Path objects for the discovered files.

    Raises:
    ValueError: If config.project_root is None.
    """

    if config.project_root is None:
        raise ValueError("Cannot discover files without a project root defined in Config.")

    project_root = config.project_root.resolve()
    candidate_files: set[Path] = set()
    _explicit_excludes = {p.resolve() for p in (explicit_exclude_paths or set())}

    # Validate and process include_globs configuration
    include_globs_config = config.get("include_globs")
    include_globs_effective = []

    if include_globs_config is None:
        if default_includes_if_missing is not None:
            logger.warning(
                "Configuration key 'include_globs' missing in [tool.vibelint] section "
                f"of pyproject.toml. Using default patterns: {default_includes_if_missing}"
            )
            include_globs_effective = default_includes_if_missing
        else:
            logger.error(
                "Configuration key 'include_globs' missing. No include patterns specified."
            )
    elif not isinstance(include_globs_config, list):
        logger.error(
            f"Config error: 'include_globs' must be a list. Found {type(include_globs_config)}."
        )
    elif not include_globs_config:
        logger.warning("Config: 'include_globs' is empty. No files will be included.")
    else:
        include_globs_effective = include_globs_config

    # Early return if no valid include patterns
    if not include_globs_effective:
        return []

    normalized_includes = [p.replace("\\", "/") for p in include_globs_effective]

    exclude_globs_config = config.get("exclude_globs", [])
    if not isinstance(exclude_globs_config, list):
        logger.error(
            f"Config error: 'exclude_globs' must be a list. Found {type(exclude_globs_config)}. Ignoring."
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

    for pattern in normalized_includes:
        pattern_start_time = time.time()
        logger.debug(f"Processing include pattern: '{pattern}'")

        if "**" in pattern:
            parts = pattern.split("**", 1)
            base_dir_glob_part = parts[0].rstrip("/")  # "src" or ""
            # glob_suffix is the part after '**/', e.g., "*.py" or "some_dir/*.txt"
            glob_suffix = parts[1].lstrip("/")

            current_search_root_abs = project_root
            if base_dir_glob_part:
                # Handle potential multiple directory components in base_dir_glob_part
                # e.g. pattern "src/app/**/... -> base_dir_glob_part = "src/app"
                current_search_root_abs = (project_root / base_dir_glob_part).resolve()

            if not current_search_root_abs.is_dir():
                logger.debug(
                    f"Skipping include pattern '{pattern}': base '{current_search_root_abs}' not a directory."
                )
                continue

            logger.debug(
                f"Using recursive walker for pattern '{pattern}' starting at '{current_search_root_abs}', suffix '{glob_suffix}'"
            )
            for p_found in _recursive_glob_with_pruning(
                current_search_root_abs,
                glob_suffix,
                project_root,
                normalized_exclude_globs,
                _explicit_excludes,
            ):
                # _recursive_glob_with_pruning already yields resolved, filtered paths
                if p_found.is_file():  # Final check, though walker should only yield files
                    candidate_files.add(p_found)  # p_found is already resolved
        else:
            # Non-recursive glob (no "**")
            logger.debug(f"Using Path.glob for non-recursive pattern: '{pattern}'")
            try:
                for p in project_root.glob(pattern):
                    abs_p = p.resolve()
                    if p.is_symlink():
                        logger.debug(f"    -> Skipping discovered symlink: {p}")
                        continue
                    if p.is_file():
                        if not _is_excluded(
                            abs_p,
                            project_root,
                            normalized_exclude_globs,
                            _explicit_excludes,
                            False,
                        ):
                            candidate_files.add(abs_p)
            except PermissionError as e:
                logger.warning(
                    f"Permission denied for non-recursive glob '{pattern}': {e}. Skipping."
                )
            except (OSError, ValueError) as e:
                logger.error(f"Error during non-recursive glob '{pattern}': {e}", exc_info=True)

        pattern_time = time.time() - pattern_start_time
        logger.debug(f"Pattern '{pattern}' processing took {pattern_time:.4f} seconds.")

    discovery_time = time.time() - start_time
    logger.debug(
        f"Globbing and initial filtering finished in {discovery_time:.4f} seconds. Total candidates: {len(candidate_files)}"
    )

    final_files_set = candidate_files

    # VCS Warning Logic
    vcs_warnings: set[Path] = set()
    if final_files_set:
        for file_path in final_files_set:
            try:
                if any(part in _VCS_DIRS for part in file_path.relative_to(project_root).parts):
                    is_actually_excluded_by_vcs_pattern = False
                    for vcs_dir_name in _VCS_DIRS:
                        if _is_excluded(
                            file_path,
                            project_root,
                            [f"{vcs_dir_name}/", f"{vcs_dir_name}/**"],
                            set(),
                            False,
                        ):
                            is_actually_excluded_by_vcs_pattern = True
                            break
                    if not is_actually_excluded_by_vcs_pattern:
                        vcs_warnings.add(file_path)
            except ValueError:
                pass
            except (OSError, TypeError) as e_vcs:
                logger.debug(f"Error during VCS check for {file_path}: {e_vcs}")

    if vcs_warnings:
        logger.warning(
            f"Found {len(vcs_warnings)} included files within potential VCS directories "
            f"({', '.join(_VCS_DIRS)}). Consider adding patterns like '.git/**' to 'exclude_globs' "
            "in your [tool.vibelint] section if this was unintended."
        )
        try:
            paths_to_log = [
                get_relative_path(p, project_root) for p in sorted(list(vcs_warnings), key=str)[:5]
            ]
            for rel_path_warn in paths_to_log:
                logger.warning(f"  - {rel_path_warn}")
            if len(vcs_warnings) > 5:
                logger.warning(f"  - ... and {len(vcs_warnings) - 5} more.")
        except Exception as e_log:
            logger.warning(f"  (Error logging VCS warning example paths: {e_log})")

    final_count = len(final_files_set)
    if final_count == 0 and include_globs_effective:
        logger.warning("No files found matching include_globs patterns or all were excluded.")

    logger.debug(f"Discovery complete. Returning {final_count} files.")
    return sorted(list(final_files_set))


def discover_files_from_paths(
    custom_paths: list[Path],
    config: Config,
    explicit_exclude_paths: set[Path] | None = None,
) -> list[Path]:
    """
    Discover files from explicitly provided paths (include_globs override).
    
    This function handles user-provided paths as an override to the configured
    include_globs, while still respecting exclude_globs and sensible defaults
    to avoid processing __pycache__, .git, etc.
    
    Args:
        custom_paths: List of file or directory paths (include_globs override)
        config: The vibelint configuration object
        explicit_exclude_paths: Additional paths to explicitly exclude
        
    Returns:
        A sorted list of unique absolute Path objects for Python files
    """
    if config.project_root is None:
        raise ValueError("Cannot discover files without a project root defined in Config.")
    
    project_root = config.project_root.resolve()
    candidate_files: set[Path] = set()
    _explicit_excludes = {p.resolve() for p in (explicit_exclude_paths or set())}
    
    # Combine config exclude patterns with defaults
    config_exclude_globs = config.get("exclude_globs", [])
    if not isinstance(config_exclude_globs, list):
        config_exclude_globs = []
    
    # Always apply default exclude patterns to avoid __pycache__, .git, etc.
    all_exclude_patterns = _DEFAULT_EXCLUDE_PATTERNS + config_exclude_globs
    
    logger.info(f"Include globs override: processing {len(custom_paths)} custom path(s)")
    logger.debug(f"Using exclude patterns: {all_exclude_patterns}")
    
    for path in custom_paths:
        abs_path = path.resolve()
        
        if abs_path.is_file():
            # Single file - check if it's a Python file and not excluded
            if abs_path.suffix == '.py':
                if not _is_excluded(
                    abs_path,
                    project_root,
                    all_exclude_patterns,
                    _explicit_excludes,
                    is_checking_directory_for_prune=False,
                ):
                    candidate_files.add(abs_path)
                else:
                    logger.debug(f"Excluding file {abs_path} due to exclude patterns")
        
        elif abs_path.is_dir():
            # Directory - recursively find Python files while respecting exclusions
            logger.debug(f"Scanning directory: {abs_path}")
            
            # Use the existing recursive walker with Python file pattern
            for py_file in _recursive_glob_with_pruning(
                abs_path,
                "*.py",  # Only Python files
                project_root,
                all_exclude_patterns,
                _explicit_excludes,
            ):
                candidate_files.add(py_file)
        else:
            logger.warning(f"Path does not exist or is not a file/directory: {abs_path}")
    
    sorted_files = sorted(candidate_files)
    logger.info(f"Include globs override result: discovered {len(sorted_files)} Python files")
    return sorted_files
