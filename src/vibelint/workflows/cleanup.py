"""
Vibelint Project Cleanup Workflow

Implements systematic project cleanup based on Workflow 7 principles.
Human-in-the-loop orchestration for cleaning up messy repositories.
"""

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DuplicateFile:
    """Represents a duplicate file found in the project."""

    original: str
    duplicate: str
    size: int
    hash: str


@dataclass
class TempFile:
    """Represents a temporary file or directory."""

    path: str
    type: str  # "temp_file" or "temp_directory"
    size: int
    pattern: str


@dataclass
class UnusedFile:
    """Represents a potentially unused Python module."""

    path: str
    type: str
    size: int


@dataclass
class LargeFile:
    """Represents an unusually large file."""

    path: str
    size: int
    size_mb: float


@dataclass
class ConfigFile:
    """Represents a configuration file."""

    path: str
    type: str
    pattern: str
    size: int


@dataclass
class DebugScript:
    """Represents a debug/test script."""

    path: str
    type: str
    pattern: str
    size: int


@dataclass
class BackupFile:
    """Represents a backup file."""

    path: str
    type: str
    pattern: str
    size: int


@dataclass
class UntrackedFile:
    """Represents an untracked file that might be important."""

    path: str
    type: str
    size: int


@dataclass
class CleanupRecommendation:
    """Represents a cleanup recommendation."""

    type: str
    priority: str
    description: str
    impact: str
    files: List[Any]


@dataclass
class ProjectAnalysis:
    """Complete project analysis results."""

    duplicate_files: List[DuplicateFile]
    temp_files: List[TempFile]
    unused_files: List[UnusedFile]
    large_files: List[LargeFile]
    empty_directories: List[str]
    config_fragments: List[ConfigFile]
    debug_scripts: List[DebugScript]
    backup_files: List[BackupFile]
    untracked_important: List[UntrackedFile]
    mess_score: float
    recommendations: List[CleanupRecommendation]


@dataclass
class CleanupAction:
    """Represents a cleanup action that was executed or skipped."""

    description: str
    type: str = ""
    path: str = ""


@dataclass
class CleanupError:
    """Represents an error encountered during cleanup."""

    error: str
    path: str = ""
    action_type: str = ""


@dataclass
class CleanupResults:
    """Results from executing cleanup actions."""

    executed: List[CleanupAction] = field(default_factory=list)
    skipped: List[CleanupAction] = field(default_factory=list)
    errors: List[CleanupError] = field(default_factory=list)
    space_saved: int = 0


@dataclass
class WorkflowStatus:
    """Status of the cleanup workflow."""

    analysis: ProjectAnalysis
    workflow: "ProjectCleanupWorkflow"
    next_step: str


class ProjectCleanupWorkflow:
    """Systematic project cleanup with human decision points."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.cleanup_log = []
        self.temp_backup_dir = None

    def analyze_project_mess(self) -> ProjectAnalysis:
        """
        Analyze the project to identify cleanup opportunities.
        Human Decision Point: What types of mess to look for.
        """
        duplicate_files = self._find_duplicate_files()
        temp_files = self._find_temp_files()
        unused_files = self._find_unused_files()
        large_files = self._find_large_files()
        empty_directories = self._find_empty_directories()
        config_fragments = self._find_config_fragments()
        debug_scripts = self._find_debug_scripts()
        backup_files = self._find_backup_files()
        untracked_important = self._find_untracked_important_files()

        # Calculate mess score
        mess_score = self._calculate_mess_score(
            duplicate_files,
            temp_files,
            unused_files,
            empty_directories,
            debug_scripts,
            backup_files,
            large_files,
        )
        recommendations = self._generate_cleanup_recommendations(
            duplicate_files,
            temp_files,
            unused_files,
            empty_directories,
            debug_scripts,
            backup_files,
        )

        return ProjectAnalysis(
            duplicate_files=duplicate_files,
            temp_files=temp_files,
            unused_files=unused_files,
            large_files=large_files,
            empty_directories=empty_directories,
            config_fragments=config_fragments,
            debug_scripts=debug_scripts,
            backup_files=backup_files,
            untracked_important=untracked_important,
            mess_score=mess_score,
            recommendations=recommendations,
        )

    def _find_duplicate_files(self) -> List[DuplicateFile]:
        """Find duplicate files by content hash."""
        file_hashes = {}
        duplicates = []

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                try:
                    with open(file_path, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()

                    if file_hash in file_hashes:
                        duplicates.append(
                            DuplicateFile(
                                original=str(file_hashes[file_hash]),
                                duplicate=str(file_path),
                                size=file_path.stat().st_size,
                                hash=file_hash,
                            )
                        )
                    else:
                        file_hashes[file_hash] = file_path

                except (IOError, OSError):
                    continue

        return duplicates

    def _find_temp_files(self) -> List[TempFile]:
        """Find temporary and build artifacts."""
        temp_patterns = [
            "*.tmp",
            "*.temp",
            "*.bak",
            "*.swp",
            "*.swo",
            "*~",
            "#*#",
            ".#*",
            "*.orig",
            "*.rej",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".pytest_cache/",
            ".coverage",
            "*.egg-info/",
            "node_modules/",
            ".npm/",
            "yarn-error.log",
            ".DS_Store",
            "Thumbs.db",
            "desktop.ini",
            ".vscode/",
            ".idea/",
            "*.log",
        ]

        temp_files = []
        for pattern in temp_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.exists():
                    temp_files.append(
                        TempFile(
                            path=str(file_path),
                            type="temp_file" if file_path.is_file() else "temp_directory",
                            size=self._get_size(file_path),
                            pattern=pattern,
                        )
                    )

        return temp_files

    def _find_unused_files(self) -> List[UnusedFile]:
        """Find files that appear unused (not imported/referenced)."""
        # Simple heuristic - look for Python files that aren't imported
        python_files = list(self.project_root.rglob("*.py"))
        all_content = ""

        # Read all Python files to check for imports
        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    all_content += f.read() + "\n"
            except (IOError, UnicodeDecodeError):
                continue

        unused_files = []
        for py_file in python_files:
            # Skip if it's a main script or test file
            if py_file.name in ["__main__.py", "__init__.py"] or "test" in py_file.name.lower():
                continue

            module_name = py_file.stem
            # Check if module is imported anywhere
            if (
                f"import {module_name}" not in all_content
                and f"from {module_name}" not in all_content
            ):
                try:
                    size = py_file.stat().st_size
                except OSError:
                    size = 0
                unused_files.append(
                    UnusedFile(
                        path=str(py_file), type="potentially_unused_python_module", size=size
                    )
                )

        return unused_files

    def _find_large_files(self) -> List[LargeFile]:
        """Find unusually large files that might need attention."""
        large_files = []
        size_threshold = 1024 * 1024  # 1MB

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size > size_threshold:
                large_files.append(
                    LargeFile(
                        path=str(file_path),
                        size=file_path.stat().st_size,
                        size_mb=file_path.stat().st_size / (1024 * 1024),
                    )
                )

        return sorted(large_files, key=lambda x: x.size, reverse=True)

    def _find_empty_directories(self) -> List[str]:
        """Find empty directories that can be removed."""
        empty_dirs = []

        for dir_path in self.project_root.rglob("*"):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                empty_dirs.append(str(dir_path))

        return empty_dirs

    def _find_config_fragments(self) -> List[ConfigFile]:
        """Find scattered configuration files that might be consolidated."""
        config_patterns = [
            "*.toml",
            "*.yaml",
            "*.yml",
            "*.json",
            "*.ini",
            "*.cfg",
            ".env*",
            "Dockerfile*",
            "requirements*.txt",
            "setup.py",
            "pyproject.toml",
            "setup.cfg",
            "tox.ini",
            ".gitignore",
        ]

        config_files = []
        for pattern in config_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    config_files.append(
                        ConfigFile(
                            path=str(file_path),
                            type="config_file",
                            pattern=pattern,
                            size=file_path.stat().st_size,
                        )
                    )

        return config_files

    def _find_debug_scripts(self) -> List[DebugScript]:
        """Find debug/test scripts that might be temporary."""
        debug_patterns = [
            "debug_*.py",
            "test_*.py",
            "*_debug.py",
            "*_test.py",
            "scratch*.py",
            "temp*.py",
            "fix_*.py",
            "quick_*.py",
        ]

        debug_files = []
        for pattern in debug_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    debug_files.append(
                        DebugScript(
                            path=str(file_path),
                            type="debug_script",
                            pattern=pattern,
                            size=file_path.stat().st_size,
                        )
                    )

        return debug_files

    def _find_backup_files(self) -> List[BackupFile]:
        """Find backup files that can be cleaned up."""
        backup_patterns = [
            "*.backup",
            "*.bkp",
            "*_backup.*",
            "*.old",
            "*_old.*",
            "*.save",
            "*_save.*",
            "*.copy",
        ]

        backup_files = []
        for pattern in backup_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    backup_files.append(
                        BackupFile(
                            path=str(file_path),
                            type="backup_file",
                            pattern=pattern,
                            size=file_path.stat().st_size,
                        )
                    )

        return backup_files

    def _find_untracked_important_files(self) -> List[UntrackedFile]:
        """Find untracked files that might be important."""
        # Do not swallow subprocess errors – allow caller to handle if needed.
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        untracked_files = []
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    file_path = self.project_root / line
                    if file_path.is_file():
                        try:
                            size = file_path.stat().st_size
                        except OSError:
                            size = 0
                        untracked_files.append(
                            UntrackedFile(path=str(file_path), type="untracked_file", size=size)
                        )

        else:
            # If git failed, raise to avoid silently returning a fallback result.
            raise RuntimeError(
                f"git ls-files failed with return code {result.returncode}: {result.stderr}"
            )

        return untracked_files

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored in analysis."""
        ignore_patterns = [
            ".git/",
            "__pycache__/",
            ".pytest_cache/",
            "node_modules/",
            ".venv/",
            "venv/",
            ".env/",
            "env/",
        ]

        path_str = str(file_path)
        return any(pattern in path_str for pattern in ignore_patterns)

    def _get_size(self, path: Path) -> int:
        """Get size of file or directory."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return 0

    def _calculate_mess_score(
        self,
        duplicate_files: List[DuplicateFile],
        temp_files: List[TempFile],
        unused_files: List[UnusedFile],
        empty_directories: List[str],
        debug_scripts: List[DebugScript],
        backup_files: List[BackupFile],
        large_files: List[LargeFile],
    ) -> float:
        """Calculate overall mess score (0-100)."""
        score = 0

        # Weight different types of mess
        score += len(duplicate_files) * 5
        score += len(temp_files) * 2
        score += len(unused_files) * 3
        score += len(empty_directories) * 1
        score += len(debug_scripts) * 2
        score += len(backup_files) * 3

        # Large files add to mess
        total_large_size = sum(f.size for f in large_files)
        score += total_large_size / (1024 * 1024 * 10)  # 10MB = 1 point

        return min(score, 100)  # Cap at 100

    def _generate_cleanup_recommendations(
        self,
        duplicate_files: List[DuplicateFile],
        temp_files: List[TempFile],
        unused_files: List[UnusedFile],
        empty_directories: List[str],
        debug_scripts: List[DebugScript],
        backup_files: List[BackupFile],
    ) -> List[CleanupRecommendation]:
        """Generate prioritized cleanup recommendations."""
        recommendations = []

        if duplicate_files:
            recommendations.append(
                CleanupRecommendation(
                    type="remove_duplicates",
                    priority="high",
                    description=f"Remove {len(duplicate_files)} duplicate files",
                    impact="disk_space",
                    files=duplicate_files,
                )
            )

        if temp_files:
            recommendations.append(
                CleanupRecommendation(
                    type="remove_temp_files",
                    priority="high",
                    description=f"Remove {len(temp_files)} temporary files",
                    impact="cleanliness",
                    files=temp_files,
                )
            )

        if backup_files:
            recommendations.append(
                CleanupRecommendation(
                    type="remove_backup_files",
                    priority="medium",
                    description=f"Remove {len(backup_files)} backup files",
                    impact="cleanliness",
                    files=backup_files,
                )
            )

        if debug_scripts:
            recommendations.append(
                CleanupRecommendation(
                    type="review_debug_scripts",
                    priority="medium",
                    description=f"Review {len(debug_scripts)} debug scripts",
                    impact="organization",
                    files=debug_scripts,
                )
            )

        if empty_directories:
            recommendations.append(
                CleanupRecommendation(
                    type="remove_empty_dirs",
                    priority="low",
                    description=f"Remove {len(empty_directories)} empty directories",
                    impact="cleanliness",
                    files=empty_directories,
                )
            )

        return recommendations

    def commit_cleanup_results(self, results: CleanupResults, cleanup_name: str):
        """Commit cleanup results with detailed message."""
        if not results.executed:
            print("No cleanup actions were executed - nothing to commit")
            return

        # Stage all changes
        subprocess.run(["git", "add", "-A"], cwd=self.project_root, check=True)

        # Create comprehensive commit message
        commit_msg = f"Cleanup: {cleanup_name}\n\n"
        commit_msg += f"Space saved: {results.space_saved / (1024*1024):.1f} MB\n"
        commit_msg += f"Actions executed: {len(results.executed)}\n"
        commit_msg += f"Actions skipped: {len(results.skipped)}\n\n"

        if results.executed:
            commit_msg += "Executed cleanup actions:\n"
            for action in results.executed:
                commit_msg += f"- {action.description}\n"

        if results.errors:
            commit_msg += f"\nErrors encountered: {len(results.errors)}\n"
            for error in results.errors:
                commit_msg += f"- {error.error}\n"

        commit_msg += "\nGenerated with vibelint cleanup workflow"

        # Commit changes
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=self.project_root, check=True)

        print(f"Committed cleanup results: {cleanup_name}")


def run_cleanup_workflow(project_root: str, cleanup_name: str = "general") -> WorkflowStatus:
    """
    Main entry point for cleanup workflow.
    Human Decision Points throughout the process.
    """
    workflow = ProjectCleanupWorkflow(Path(project_root))

    print(f"Starting cleanup analysis for: {project_root}")

    # Step 1: Analyze project mess
    print("Analyzing project structure...")
    analysis = workflow.analyze_project_mess()

    print(f"Mess score: {analysis.mess_score:.1f}/100")
    print(f"Found {len(analysis.recommendations)} cleanup recommendations")

    # Step 2: Present recommendations to human
    print("\nCleanup Recommendations:")
    for i, rec in enumerate(analysis.recommendations, 1):
        print(f"{i}. [{rec.priority.upper()}] {rec.description}")
        print(f"   Impact: {rec.impact}")

    # HUMAN DECISION POINT: Which recommendations to execute
    print("\nHUMAN DECISION REQUIRED:")
    print("Which cleanup actions would you like to execute?")
    print("Available types:", [rec.type for rec in analysis.recommendations])

    # For now, return analysis for human review
    # In interactive mode, human would approve specific actions
    return WorkflowStatus(analysis=analysis, workflow=workflow, next_step="human_approval_required")
