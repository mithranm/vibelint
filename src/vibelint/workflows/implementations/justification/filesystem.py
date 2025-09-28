"""
File system crawler for justification workflow.

Handles discovery, filtering, and basic file analysis for the justification engine.
Separated from the main engine to improve modularity and testability.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class FileSystemCrawler:
    """Handles file system discovery and basic file operations for justification analysis."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def generate_project_tree(self, directory_path: Path) -> str:
        """Generate a basic project tree structure."""
        tree_lines = ["# Project Snapshot", "", "## Filesystem Tree", "", "```"]

        for root, dirs, files in os.walk(directory_path):
            level = root.replace(str(directory_path), '').count(os.sep)
            indent = '│   ' * level
            tree_lines.append(f"{indent}├── {os.path.basename(root)}/")
            subindent = '│   ' * (level + 1)
            for file in files:
                if not file.startswith('.'):
                    tree_lines.append(f"{subindent}├── {file}")

        tree_lines.append("```")
        return '\n'.join(tree_lines)

    def discover_python_files(self, directory_path: Path,
                             include_patterns: Optional[List[str]] = None,
                             exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """Discover Python files matching the given patterns."""
        python_files = []

        # Default patterns
        if include_patterns is None:
            include_patterns = ["**/*.py"]
        if exclude_patterns is None:
            exclude_patterns = [
                "**/__pycache__/**",
                "**/.*",
                "build/**",
                "dist/**",
                "*cache*/**",
                "*.vibelint*",
                ".vibelint-*/**",
                "*.pyc", "*.pyo", "*.pyd",
                "*.log"
            ]

        # Use pathlib for pattern matching
        for pattern in include_patterns:
            for file_path in directory_path.glob(pattern):
                if file_path.is_file() and file_path.suffix == '.py':
                    # Check exclusions
                    should_exclude = False
                    for exclude_pattern in exclude_patterns:
                        if file_path.match(exclude_pattern):
                            should_exclude = True
                            break

                    if not should_exclude:
                        python_files.append(file_path)

        return sorted(list(set(python_files)))  # Remove duplicates and sort

    def discover_all_files(self, directory_path: Path, size_limit_mb: int = 1) -> List[Path]:
        """Discover all files under size limit for backup detection and analysis."""
        all_files = []
        size_limit_bytes = size_limit_mb * 1024 * 1024

        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.stat().st_size < size_limit_bytes:
                        all_files.append(file_path)
                except (OSError, ValueError):
                    continue

        return all_files

    def read_file_safely(self, file_path: Path) -> Optional[str]:
        """Read file content safely with error handling."""
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                return file_path.read_text(encoding='latin-1')
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                return None
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            return None

    def find_file_in_directory(self, directory_path: Path, target_filename: str) -> Optional[Path]:
        """Find a specific file in the directory tree."""
        for file_path in directory_path.rglob(target_filename):
            if file_path.is_file():
                return file_path
        return None

    def calculate_directory_stats(self, directory_path: Path) -> Dict[str, int]:
        """Calculate basic statistics about the directory."""
        stats = {
            "total_files": 0,
            "python_files": 0,
            "total_size_bytes": 0,
            "subdirectories": 0
        }

        for item_path in directory_path.rglob("*"):
            if item_path.is_file():
                stats["total_files"] += 1
                if item_path.suffix == '.py':
                    stats["python_files"] += 1
                try:
                    stats["total_size_bytes"] += item_path.stat().st_size
                except OSError:
                    pass
            elif item_path.is_dir():
                stats["subdirectories"] += 1

        return stats

    def get_file_relative_path(self, file_path: Path, base_directory: Path) -> str:
        """Get the relative path from base directory."""
        try:
            return str(file_path.relative_to(base_directory))
        except ValueError:
            return str(file_path)

    def filter_files_by_size(self, file_paths: List[Path], max_lines: int = 2000) -> List[Path]:
        """Filter files by line count to avoid processing huge files."""
        filtered_files = []

        for file_path in file_paths:
            try:
                content = self.read_file_safely(file_path)
                if content and len(content.splitlines()) <= max_lines:
                    filtered_files.append(file_path)
                elif content:
                    logger.debug(f"Skipping large file {file_path}: {len(content.splitlines())} lines")
            except Exception as e:
                logger.debug(f"Error checking file size {file_path}: {e}")

        return filtered_files