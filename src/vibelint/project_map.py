"""
Project mapping and file discovery system for scalable organization.

Automatically generates project structure maps, detects organizational issues,
and suggests improvements for large codebases.

vibelint/src/vibelint/project_map.py
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

logger = logging.getLogger(__name__)

__all__ = ["ProjectMapper", "FileNode", "ModuleGroup"]


@dataclass
class FileNode:
    """Represents a file in the project structure."""

    path: str
    name: str
    size: int
    file_type: str
    purpose: Optional[str] = None
    dependencies: List[str] = None
    exports: List[str] = None
    lines_of_code: Optional[int] = None
    last_modified: Optional[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.exports is None:
            self.exports = []


@dataclass
class ModuleGroup:
    """Represents a logical grouping of related files."""

    name: str
    files: List[FileNode]
    cohesion_score: float
    suggested_location: str
    grouping_reason: str


class ProjectMapper:
    """Generates comprehensive project structure maps and organization suggestions."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.file_nodes: Dict[str, FileNode] = {}
        self.module_groups: List[ModuleGroup] = []

    def generate_project_map(self) -> Dict[str, Any]:
        """Generate comprehensive project structure map."""

        # 1. Discover all files
        self._discover_files()

        # 2. Analyze dependencies
        self._analyze_dependencies()

        # 3. Detect module groups
        self._detect_module_groups()

        # 4. Calculate organization metrics
        metrics = self._calculate_organization_metrics()

        # 5. Generate recommendations
        recommendations = self._generate_recommendations()

        return {
            "project_root": str(self.project_root),
            "total_files": len(self.file_nodes),
            "file_tree": self._build_file_tree(),
            "module_groups": [asdict(group) for group in self.module_groups],
            "organization_metrics": metrics,
            "recommendations": recommendations,
            "file_index": {path: asdict(node) for path, node in self.file_nodes.items()},
        }

    def _discover_files(self):
        """Discover and catalog all project files."""
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                relative_path = str(file_path.relative_to(self.project_root))

                node = FileNode(
                    path=relative_path,
                    name=file_path.name,
                    size=file_path.stat().st_size,
                    file_type=file_path.suffix or "no_extension",
                    purpose=self._infer_file_purpose(file_path),
                    lines_of_code=self._count_lines_of_code(file_path),
                    last_modified=file_path.stat().st_mtime,
                )

                self.file_nodes[relative_path] = node

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored in analysis."""
        ignore_patterns = {
            # Directories
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".mypy_cache",
            ".tox",
            "build",
            "dist",
            ".venv",
            "venv",
            # File patterns
            ".pyc",
            ".pyo",
            ".pyd",
            ".so",
            ".dylib",
            ".dll",
            ".log",
            ".tmp",
            ".temp",
            ".cache",
        }

        return any(pattern in str(file_path) for pattern in ignore_patterns)

    def _infer_file_purpose(self, file_path: Path) -> str:
        """Infer file purpose from name, location, and content."""
        name = file_path.name.lower()

        # Configuration files
        if name in {"pyproject.toml", "setup.py", "requirements.txt", "tox.ini"}:
            return "configuration"

        # Documentation
        if file_path.suffix in {".md", ".rst", ".txt"} or "readme" in name:
            return "documentation"

        # Tests
        if "test" in name or "tests" in str(file_path):
            return "testing"

        # Python modules
        if file_path.suffix == ".py":
            if name == "__init__.py":
                return "package_init"
            elif name in {"cli.py", "main.py", "__main__.py"}:
                return "entry_point"
            elif any(keyword in name for keyword in ["validator", "check", "lint"]):
                return "validation_logic"
            elif any(keyword in name for keyword in ["manager", "handler", "service"]):
                return "business_logic"
            elif any(keyword in name for keyword in ["util", "helper", "tool"]):
                return "utility"
            else:
                return "module"

        # Scripts
        if file_path.suffix in {".sh", ".bat", ".ps1"}:
            return "script"

        return "unknown"

    def _count_lines_of_code(self, file_path: Path) -> Optional[int]:
        """Count lines of code in file."""
        try:
            if file_path.suffix in {".py", ".js", ".ts", ".java", ".cpp", ".c", ".h"}:
                content = file_path.read_text(encoding="utf-8")
                return len(
                    [
                        line
                        for line in content.splitlines()
                        if line.strip() and not line.strip().startswith("#")
                    ]
                )
        except (UnicodeDecodeError, PermissionError):
            pass
        return None

    def _build_file_tree(self) -> Dict[str, Any]:
        """Build hierarchical file tree structure."""
        tree = {}

        for file_path in self.file_nodes.keys():
            parts = file_path.split("/")
            current = tree

            for part in parts[:-1]:  # Directories
                if part not in current:
                    current[part] = {"type": "directory", "children": {}}
                current = current[part]["children"]

            # File
            filename = parts[-1]
            current[filename] = {
                "type": "file",
                "purpose": self.file_nodes[file_path].purpose,
                "size": self.file_nodes[file_path].size,
                "lines_of_code": self.file_nodes[file_path].lines_of_code,
            }

        return tree

    def _analyze_dependencies(self):
        """Analyze import dependencies between files."""
        # This would parse Python files to extract imports
        # Simplified for now
        pass

    def _detect_module_groups(self):
        """Detect logical groupings of related files."""
        # Group by purpose and naming patterns
        purpose_groups = defaultdict(list)

        for node in self.file_nodes.values():
            if node.purpose != "unknown":
                purpose_groups[node.purpose].append(node)

        # Create module groups for related files
        for purpose, files in purpose_groups.items():
            if len(files) > 1:
                cohesion_score = self._calculate_cohesion_score(files)
                suggested_location = f"src/vibelint/{purpose}/"

                group = ModuleGroup(
                    name=purpose,
                    files=files,
                    cohesion_score=cohesion_score,
                    suggested_location=suggested_location,
                    grouping_reason=f"Files with shared purpose: {purpose}",
                )
                self.module_groups.append(group)

    def _calculate_cohesion_score(self, files: List[FileNode]) -> float:
        """Calculate cohesion score for a group of files."""
        # Simplified scoring based on naming patterns and purposes
        if len(files) < 2:
            return 0.0

        # Score based on naming similarity
        names = [f.name for f in files]
        common_prefixes = self._find_common_prefixes(names)
        naming_score = len(common_prefixes) / len(files)

        return min(naming_score * 2, 1.0)  # Scale to 0-1

    def _find_common_prefixes(self, names: List[str]) -> Set[str]:
        """Find common prefixes in file names."""
        prefixes = set()
        for name in names:
            parts = name.replace("_", " ").replace("-", " ").split()
            for part in parts:
                if len(part) > 2:
                    prefixes.add(part)
        return prefixes

    def _calculate_organization_metrics(self) -> Dict[str, Any]:
        """Calculate metrics about project organization quality."""
        total_files = len(self.file_nodes)

        # Files by purpose
        purpose_distribution = defaultdict(int)
        for node in self.file_nodes.values():
            purpose_distribution[node.purpose] += 1

        # Directory depth analysis
        max_depth = max(len(path.split("/")) for path in self.file_nodes.keys())
        avg_depth = sum(len(path.split("/")) for path in self.file_nodes.keys()) / total_files

        # Grouping potential
        groupable_files = sum(1 for group in self.module_groups if group.cohesion_score > 0.5)

        return {
            "total_files": total_files,
            "max_directory_depth": max_depth,
            "average_directory_depth": avg_depth,
            "purpose_distribution": dict(purpose_distribution),
            "potential_module_groups": len(self.module_groups),
            "groupable_files": groupable_files,
            "organization_score": self._calculate_organization_score(),
        }

    def _calculate_organization_score(self) -> float:
        """Calculate overall organization quality score (0-1)."""
        # Factors that indicate good organization
        factors = []

        # 1. Purpose clarity (how many files have clear purposes)
        clear_purpose_ratio = sum(
            1 for node in self.file_nodes.values() if node.purpose != "unknown"
        ) / len(self.file_nodes)
        factors.append(clear_purpose_ratio)

        # 2. Module cohesion (how well files are grouped)
        if self.module_groups:
            avg_cohesion = sum(group.cohesion_score for group in self.module_groups) / len(
                self.module_groups
            )
            factors.append(avg_cohesion)
        else:
            factors.append(0.5)  # Neutral score for no groups

        # 3. Directory structure depth (not too deep, not too flat)
        total_files = len(self.file_nodes)
        max_depth = max(len(path.split("/")) for path in self.file_nodes.keys())

        # Ideal depth: 2-4 levels for most projects
        if total_files < 20:
            ideal_depth = 2
        elif total_files < 100:
            ideal_depth = 3
        else:
            ideal_depth = 4

        depth_score = max(0, 1 - abs(max_depth - ideal_depth) / ideal_depth)
        factors.append(depth_score)

        return sum(factors) / len(factors)

    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate actionable recommendations for improving organization."""
        recommendations = []

        # Recommend grouping scattered modules
        for group in self.module_groups:
            if group.cohesion_score > 0.7 and len(group.files) >= 2:
                file_names = [f.name for f in group.files]
                recommendations.append(
                    {
                        "type": "module_grouping",
                        "priority": "medium",
                        "description": f"Group {group.name} files into subpackage",
                        "action": f"mkdir {group.suggested_location} && mv {' '.join(file_names)} {group.suggested_location}",
                        "reason": group.grouping_reason,
                    }
                )

        # Recommend documentation for unclear files
        unclear_files = [node for node in self.file_nodes.values() if node.purpose == "unknown"]
        if unclear_files:
            recommendations.append(
                {
                    "type": "documentation",
                    "priority": "high",
                    "description": f"Document purpose of {len(unclear_files)} unclear files",
                    "action": "Add docstrings or comments explaining file purposes",
                    "reason": "Files without clear purpose indicate organizational debt",
                }
            )

        # Recommend directory restructuring if too flat/deep
        organization_score = self._calculate_organization_score()
        if organization_score < 0.6:
            recommendations.append(
                {
                    "type": "restructuring",
                    "priority": "high",
                    "description": "Consider major project restructuring",
                    "action": "Group related files into logical subpackages",
                    "reason": f"Organization score is low ({organization_score:.2f})",
                }
            )

        return recommendations

    def save_project_map(self, output_path: Path):
        """Save project map to JSON file."""
        project_map = self.generate_project_map()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(project_map, f, indent=2, default=str)

        logger.info(f"Project map saved to {output_path}")
        return project_map
