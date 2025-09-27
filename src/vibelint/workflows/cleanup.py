"""
Vibelint Project Cleanup Workflow

Implements systematic project cleanup based on Workflow 7 principles.
Human-in-the-loop orchestration for cleaning up messy repositories.
"""

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProjectCleanupWorkflow:
 """Systematic project cleanup with human decision points."""

 def __init__(self, project_root: Path):
 self.project_root = Path(project_root)
 self.cleanup_log = []
 self.temp_backup_dir = None

 def analyze_project_mess(self) -> Dict[str, Any]:
 """
 Analyze the project to identify cleanup opportunities.
 Human Decision Point: What types of mess to look for.
 """
 analysis = {
 "duplicate_files": self._find_duplicate_files(),
 "temp_files": self._find_temp_files(),
 "unused_files": self._find_unused_files(),
 "large_files": self._find_large_files(),
 "empty_directories": self._find_empty_directories(),
 "config_fragments": self._find_config_fragments(),
 "debug_scripts": self._find_debug_scripts(),
 "backup_files": self._find_backup_files(),
 "untracked_important": self._find_untracked_important_files(),
 }

 # Calculate mess score
 analysis["mess_score"] = self._calculate_mess_score(analysis)
 analysis["recommendations"] = self._generate_cleanup_recommendations(analysis)

 return analysis

 def _find_duplicate_files(self) -> List[Dict[str, Any]]:
 """Find duplicate files by content hash."""
 import hashlib

 file_hashes = {}
 duplicates = []

 for file_path in self.project_root.rglob("*"):
 if file_path.is_file() and not self._should_ignore_file(file_path):
 try:
 with open(file_path, "rb") as f:
 file_hash = hashlib.md5(f.read()).hexdigest()

 if file_hash in file_hashes:
 duplicates.append({
 "original": str(file_hashes[file_hash]),
 "duplicate": str(file_path),
 "size": file_path.stat().st_size,
 "hash": file_hash
 })
 else:
 file_hashes[file_hash] = file_path

 except (IOError, OSError):
 continue

 return duplicates

 def _find_temp_files(self) -> List[Dict[str, Any]]:
 """Find temporary and build artifacts."""
 temp_patterns = [
 "*.tmp", "*.temp", "*.bak", "*.swp", "*.swo",
 "*~", "#*#", ".#*", "*.orig", "*.rej",
 "__pycache__/", "*.pyc", "*.pyo", "*.pyd",
 ".pytest_cache/", ".coverage", "*.egg-info/",
 "node_modules/", ".npm/", "yarn-error.log",
 ".DS_Store", "Thumbs.db", "desktop.ini",
 ".vscode/", ".idea/", "*.log"
 ]

 temp_files = []
 for pattern in temp_patterns:
 for file_path in self.project_root.rglob(pattern):
 if file_path.exists():
 temp_files.append({
 "path": str(file_path),
 "type": "temp_file" if file_path.is_file() else "temp_directory",
 "size": self._get_size(file_path),
 "pattern": pattern
 })

 return temp_files

 def _find_unused_files(self) -> List[Dict[str, Any]]:
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
 if f"import {module_name}" not in all_content and f"from {module_name}" not in all_content:
 unused_files.append({
 "path": str(py_file),
 "type": "potentially_unused_python_module",
 "size": py_file.stat().st_size
 })

 return unused_files

 def _find_large_files(self) -> List[Dict[str, Any]]:
 """Find unusually large files that might need attention."""
 large_files = []
 size_threshold = 1024 * 1024 # 1MB

 for file_path in self.project_root.rglob("*"):
 if file_path.is_file() and file_path.stat().st_size > size_threshold:
 large_files.append({
 "path": str(file_path),
 "size": file_path.stat().st_size,
 "size_mb": file_path.stat().st_size / (1024 * 1024)
 })

 return sorted(large_files, key=lambda x: x["size"], reverse=True)

 def _find_empty_directories(self) -> List[str]:
 """Find empty directories that can be removed."""
 empty_dirs = []

 for dir_path in self.project_root.rglob("*"):
 if dir_path.is_dir() and not any(dir_path.iterdir()):
 empty_dirs.append(str(dir_path))

 return empty_dirs

 def _find_config_fragments(self) -> List[Dict[str, Any]]:
 """Find scattered configuration files that might be consolidated."""
 config_patterns = [
 "*.toml", "*.yaml", "*.yml", "*.json", "*.ini", "*.cfg",
 ".env*", "Dockerfile*", "requirements*.txt", "setup.py",
 "pyproject.toml", "setup.cfg", "tox.ini", ".gitignore"
 ]

 config_files = []
 for pattern in config_patterns:
 for file_path in self.project_root.rglob(pattern):
 if file_path.is_file():
 config_files.append({
 "path": str(file_path),
 "type": "config_file",
 "pattern": pattern,
 "size": file_path.stat().st_size
 })

 return config_files

 def _find_debug_scripts(self) -> List[Dict[str, Any]]:
 """Find debug/test scripts that might be temporary."""
 debug_patterns = [
 "debug_*.py", "test_*.py", "*_debug.py", "*_test.py",
 "scratch*.py", "temp*.py", "fix_*.py", "quick_*.py"
 ]

 debug_files = []
 for pattern in debug_patterns:
 for file_path in self.project_root.rglob(pattern):
 if file_path.is_file():
 debug_files.append({
 "path": str(file_path),
 "type": "debug_script",
 "pattern": pattern,
 "size": file_path.stat().st_size
 })

 return debug_files

 def _find_backup_files(self) -> List[Dict[str, Any]]:
 """Find backup files that can be cleaned up."""
 backup_patterns = [
 "*.backup", "*.bkp", "*_backup.*", "*.old",
 "*_old.*", "*.save", "*_save.*", "*.copy"
 ]

 backup_files = []
 for pattern in backup_patterns:
 for file_path in self.project_root.rglob(pattern):
 if file_path.is_file():
 backup_files.append({
 "path": str(file_path),
 "type": "backup_file",
 "pattern": pattern,
 "size": file_path.stat().st_size
 })

 return backup_files

 def _find_untracked_important_files(self) -> List[Dict[str, Any]]:
 """Find untracked files that might be important."""
 try:
 result = subprocess.run(
 ["git", "ls-files", "--others", "--exclude-standard"],
 cwd=self.project_root,
 capture_output=True,
 text=True
 )

 untracked_files = []
 if result.returncode == 0:
 for line in result.stdout.strip().split("\n"):
 if line:
 file_path = self.project_root / line
 if file_path.is_file():
 untracked_files.append({
 "path": str(file_path),
 "type": "untracked_file",
 "size": file_path.stat().st_size if file_path.exists() else 0
 })

 return untracked_files

 except subprocess.SubprocessError:
 return []

 def _should_ignore_file(self, file_path: Path) -> bool:
 """Check if file should be ignored in analysis."""
 ignore_patterns = [
 ".git/", "__pycache__/", ".pytest_cache/", "node_modules/",
 ".venv/", "venv/", ".env/", "env/"
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

 def _calculate_mess_score(self, analysis: Dict[str, Any]) -> float:
 """Calculate overall mess score (0-100)."""
 score = 0

 # Weight different types of mess
 score += len(analysis["duplicate_files"]) * 5
 score += len(analysis["temp_files"]) * 2
 score += len(analysis["unused_files"]) * 3
 score += len(analysis["empty_directories"]) * 1
 score += len(analysis["debug_scripts"]) * 2
 score += len(analysis["backup_files"]) * 3

 # Large files add to mess
 total_large_size = sum(f["size"] for f in analysis["large_files"])
 score += total_large_size / (1024 * 1024 * 10) # 10MB = 1 point

 return min(score, 100) # Cap at 100

 def _generate_cleanup_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
 """Generate prioritized cleanup recommendations."""
 recommendations = []

 if analysis["duplicate_files"]:
 recommendations.append({
 "type": "remove_duplicates",
 "priority": "high",
 "description": f"Remove {len(analysis['duplicate_files'])} duplicate files",
 "impact": "disk_space",
 "files": analysis["duplicate_files"]
 })

 if analysis["temp_files"]:
 recommendations.append({
 "type": "remove_temp_files",
 "priority": "high",
 "description": f"Remove {len(analysis['temp_files'])} temporary files",
 "impact": "cleanliness",
 "files": analysis["temp_files"]
 })

 if analysis["backup_files"]:
 recommendations.append({
 "type": "remove_backup_files",
 "priority": "medium",
 "description": f"Remove {len(analysis['backup_files'])} backup files",
 "impact": "cleanliness",
 "files": analysis["backup_files"]
 })

 if analysis["debug_scripts"]:
 recommendations.append({
 "type": "review_debug_scripts",
 "priority": "medium",
 "description": f"Review {len(analysis['debug_scripts'])} debug scripts",
 "impact": "organization",
 "files": analysis["debug_scripts"]
 })

 if analysis["empty_directories"]:
 recommendations.append({
 "type": "remove_empty_dirs",
 "priority": "low",
 "description": f"Remove {len(analysis['empty_directories'])} empty directories",
 "impact": "cleanliness",
 "files": analysis["empty_directories"]
 })

 return recommendations

 def commit_cleanup_results(self, results: Dict[str, Any], cleanup_name: str):
 """Commit cleanup results with detailed message."""
 if not results["executed"]:
 print("No cleanup actions were executed - nothing to commit")
 return

 # Stage all changes
 subprocess.run(["git", "add", "-A"], cwd=self.project_root, check=True)

 # Create comprehensive commit message
 commit_msg = f"Cleanup: {cleanup_name}\n\n"
 commit_msg += f"Space saved: {results['space_saved'] / (1024*1024):.1f} MB\n"
 commit_msg += f"Actions executed: {len(results['executed'])}\n"
 commit_msg += f"Actions skipped: {len(results['skipped'])}\n\n"

 if results["executed"]:
 commit_msg += "Executed cleanup actions:\n"
 for action in results["executed"]:
 commit_msg += f"- {action['description']}\n"

 if results["errors"]:
 commit_msg += f"\nErrors encountered: {len(results['errors'])}\n"
 for error in results["errors"]:
 commit_msg += f"- {error['error']}\n"

 commit_msg += f"\nGenerated with vibelint cleanup workflow"

 # Commit changes
 subprocess.run(
 ["git", "commit", "-m", commit_msg],
 cwd=self.project_root,
 check=True
 )

 print(f"Committed cleanup results: {cleanup_name}")


def run_cleanup_workflow(project_root: str, cleanup_name: str = "general") -> Dict[str, Any]:
 """
 Main entry point for cleanup workflow.
 Human Decision Points throughout the process.
 """
 workflow = ProjectCleanupWorkflow(Path(project_root))

 print(f"Starting cleanup analysis for: {project_root}")

 # Step 1: Analyze project mess
 print("Analyzing project structure...")
 analysis = workflow.analyze_project_mess()

 print(f"Mess score: {analysis['mess_score']:.1f}/100")
 print(f"Found {len(analysis['recommendations'])} cleanup recommendations")

 # Step 2: Present recommendations to human
 print("\nCleanup Recommendations:")
 for i, rec in enumerate(analysis["recommendations"], 1):
 print(f"{i}. [{rec['priority'].upper()}] {rec['description']}")
 print(f" Impact: {rec['impact']}")

 # HUMAN DECISION POINT: Which recommendations to execute
 print("\nHUMAN DECISION REQUIRED:")
 print("Which cleanup actions would you like to execute?")
 print("Available types:", [rec["type"] for rec in analysis["recommendations"]])

 # For now, return analysis for human review
 # In interactive mode, human would approve specific actions
 return {
 "analysis": analysis,
 "workflow": workflow,
 "next_step": "human_approval_required"
 }