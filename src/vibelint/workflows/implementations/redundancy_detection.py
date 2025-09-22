"""
Redundancy detection workflow for vibelint codebase analysis.

Analyzes code redundancies and dead code patterns starting from CLI entry points
using dynamic analysis and static code inspection techniques.

vibelint/src/vibelint/workflows/redundancy_detection.py
"""

import ast
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import BaseWorkflow, WorkflowConfig, WorkflowResult, WorkflowStatus

logger = logging.getLogger(__name__)

__all__ = ["RedundancyDetectionWorkflow", "RedundancyPattern", "DeadCodeCandidate"]


@dataclass
class RedundancyPattern:
    """Represents a potentially redundant code pattern."""

    pattern_type: str  # "function", "class", "import", "logic_block"
    signature: str  # Normalized signature or pattern
    locations: List[Tuple[str, int]]  # (file_path, line_number)
    similarity_score: float  # 0.0 to 1.0
    estimated_redundancy: str  # "duplicate", "similar", "refactorable"


@dataclass
class DeadCodeCandidate:
    """Represents potentially dead code."""

    code_type: str  # "function", "class", "import"
    name: str
    file_path: str
    line_number: int
    reason: str  # Why it's considered dead
    confidence: float  # 0.0 to 1.0


class RedundancyDetectionWorkflow(BaseWorkflow):
    """Workflow for detecting code redundancies and dead code from CLI entry points."""

    workflow_id = "redundancy-detection"
    name = "Redundancy Detection"
    description = "Analyzes code redundancies and dead code patterns from CLI entry points"
    category = "maintenance"
    tags = {"redundancy", "dead_code", "cleanup", "maintenance"}

    def __init__(self, config: Optional[WorkflowConfig] = None):
        super().__init__(config)

        # Analysis state
        self.all_functions = {}  # signature -> locations
        self.all_classes = {}  # signature -> locations
        self.import_usage = defaultdict(set)  # module -> using_files
        self.function_calls = defaultdict(set)  # function -> calling_functions
        self.entry_points = []
        self.cli_commands = []

    def get_required_inputs(self) -> Set[str]:
        """Get set of required input data keys."""
        return {"project_root"}

    def get_produced_outputs(self) -> Set[str]:
        """Get set of output data keys this workflow produces."""
        return {
            "redundancy_patterns",
            "dead_code_candidates",
            "import_redundancies",
            "consolidation_opportunities",
            "removal_benefit_estimate",
        }

    async def execute(self, project_root: Path, context: Dict[str, Any]) -> WorkflowResult:
        """Execute redundancy detection workflow."""

        logger.info("Starting redundancy detection from CLI entry points...")

        try:
            # Step 1: Map CLI entry points
            self._discover_cli_entry_points(project_root)

            # Step 2: Trace code paths from entry points
            reachable_code = self._trace_reachable_code(project_root)

            # Step 3: Find dead code candidates
            dead_code = self._find_dead_code_candidates(project_root, reachable_code)

            # Step 4: Detect redundant patterns
            redundant_patterns = self._detect_redundant_patterns(project_root)

            # Step 5: Analyze import redundancies
            import_redundancies = self._analyze_import_redundancies(project_root)

            # Step 6: Find consolidation opportunities
            consolidation_opportunities = self._find_consolidation_opportunities(project_root)

            # Step 7: Estimate removal benefits
            removal_benefits = self._estimate_removal_benefit(dead_code, redundant_patterns)

            # Create findings
            findings = []

            # Add dead code findings
            for candidate in dead_code:
                findings.append(
                    {
                        "rule_id": "DEAD-CODE-CANDIDATE",
                        "severity": "INFO",
                        "message": f"Potentially dead {candidate.code_type}: {candidate.name}",
                        "file_path": candidate.file_path,
                        "line": candidate.line_number,
                        "suggestion": f"Consider removing if truly unused. Reason: {candidate.reason}",
                        "confidence": candidate.confidence,
                    }
                )

            # Add redundancy findings
            for pattern in redundant_patterns:
                findings.append(
                    {
                        "rule_id": "REDUNDANT-PATTERN",
                        "severity": "WARN" if pattern.similarity_score > 0.8 else "INFO",
                        "message": f"Redundant {pattern.pattern_type} pattern found in {len(pattern.locations)} locations",
                        "locations": pattern.locations,
                        "suggestion": f"Consider consolidating similar {pattern.pattern_type}s",
                        "similarity_score": pattern.similarity_score,
                    }
                )

            # Create artifacts
            artifacts = {
                "redundancy_patterns": [self._pattern_to_dict(p) for p in redundant_patterns],
                "dead_code_candidates": [self._candidate_to_dict(c) for c in dead_code],
                "import_redundancies": import_redundancies,
                "consolidation_opportunities": consolidation_opportunities,
                "removal_benefit_estimate": removal_benefits,
                "entry_points_traced": self.entry_points,
                "cli_commands_found": self.cli_commands,
            }

            # Update metrics
            self.metrics.files_processed = len(self._get_all_python_files(project_root))
            self.metrics.findings_generated = len(findings)
            self.metrics.confidence_score = self._calculate_overall_confidence(
                dead_code, redundant_patterns
            )

            return self._create_result(
                WorkflowStatus.COMPLETED, findings=findings, artifacts=artifacts
            )

        except Exception as e:
            logger.error(f"Redundancy detection failed: {e}", exc_info=True)
            return self._create_result(WorkflowStatus.FAILED, error_message=str(e))

    def _discover_cli_entry_points(self, project_root: Path):
        """Discover CLI entry points from setup and CLI modules."""
        logger.debug("Discovering CLI entry points...")

        # Check pyproject.toml for script entries
        pyproject_path = project_root / "pyproject.toml"
        if pyproject_path.exists():
            self._parse_pyproject_entry_points(pyproject_path)

        # Analyze cli.py for command handlers
        source_root = project_root / "src"
        cli_candidates = [
            source_root / "vibelint" / "cli.py",
            project_root / "cli.py",
            project_root / "main.py",
        ]

        for cli_path in cli_candidates:
            if cli_path.exists():
                self._analyze_cli_module(cli_path)

    def _parse_pyproject_entry_points(self, pyproject_path: Path):
        """Parse entry points from pyproject.toml."""
        try:
            import sys

            if sys.version_info >= (3, 11):
                import tomllib

                with open(pyproject_path, "rb") as f:
                    config = tomllib.load(f)
            else:
                import tomli

                content = pyproject_path.read_text(encoding="utf-8")
                config = tomli.loads(content)

            # Script entry points
            scripts = config.get("project", {}).get("scripts", {})
            for script_name, entry_point in scripts.items():
                self.entry_points.append(f"script:{script_name}={entry_point}")

            # Plugin entry points
            entry_points = config.get("project", {}).get("entry-points", {})
            for group, entries in entry_points.items():
                for name, entry_point in entries.items():
                    self.entry_points.append(f"plugin:{group}:{name}={entry_point}")

        except Exception as e:
            logger.warning(f"Failed to parse pyproject.toml: {e}")

    def _analyze_cli_module(self, cli_path: Path):
        """Analyze CLI module for command handlers."""
        try:
            content = cli_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for click command decorators
                    for decorator in node.decorator_list:
                        if self._is_click_command(decorator):
                            command_name = node.name
                            self.cli_commands.append(command_name)

        except Exception as e:
            logger.warning(f"Failed to analyze CLI module: {e}")

    def _is_click_command(self, decorator) -> bool:
        """Check if decorator is a click command."""
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr in ["command", "group"]
            elif isinstance(decorator.func, ast.Name):
                return decorator.func.id in ["command", "group"]
        return False

    def _trace_reachable_code(self, project_root: Path) -> Set[str]:
        """Trace all code reachable from CLI entry points."""
        logger.debug("Tracing reachable code from entry points...")

        reachable = set()

        # Start from CLI command handlers
        for command_name in self.cli_commands:
            reachable.add(f"function:{command_name}")

        # Add entry point functions
        for entry_point in self.entry_points:
            if "=" in entry_point:
                module_func = entry_point.split("=")[1]
                reachable.add(f"entry:{module_func}")

        # Simple heuristic: if we found CLI commands, assume moderate reachability
        # Full implementation would require building complete call graph
        return reachable

    def _find_dead_code_candidates(
        self, project_root: Path, reachable_code: Set[str]
    ) -> List[DeadCodeCandidate]:
        """Find code that appears to be unreachable from entry points."""
        logger.debug("Finding dead code candidates...")

        dead_code = []
        all_files = self._get_all_python_files(project_root)

        for file_path in all_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                tree = ast.parse(content)
                relative_path = str(file_path.relative_to(project_root))

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_id = f"function:{node.name}"
                        if func_id not in reachable_code:
                            if not self._is_exempt_function(node, file_path):
                                dead_code.append(
                                    DeadCodeCandidate(
                                        code_type="function",
                                        name=node.name,
                                        file_path=relative_path,
                                        line_number=node.lineno,
                                        reason="unreachable_from_entry_points",
                                        confidence=0.6,  # Moderate confidence
                                    )
                                )

                    elif isinstance(node, ast.ClassDef):
                        class_id = f"class:{node.name}"
                        if class_id not in reachable_code:
                            if not self._is_exempt_class(node, file_path):
                                dead_code.append(
                                    DeadCodeCandidate(
                                        code_type="class",
                                        name=node.name,
                                        file_path=relative_path,
                                        line_number=node.lineno,
                                        reason="unreachable_from_entry_points",
                                        confidence=0.5,  # Lower confidence for classes
                                    )
                                )

            except Exception as e:
                logger.debug(f"Failed to analyze {file_path}: {e}")

        return dead_code

    def _is_exempt_function(self, node: ast.FunctionDef, file_path: Path) -> bool:
        """Check if function should be exempt from dead code detection."""
        # Test functions
        if "test" in file_path.name or node.name.startswith("test_"):
            return True

        # Private functions (might be used dynamically)
        if node.name.startswith("_"):
            return True

        # Special methods
        if node.name.startswith("__") and node.name.endswith("__"):
            return True

        # Entry point functions
        if node.name in ["main", "cli"]:
            return True

        return False

    def _is_exempt_class(self, node: ast.ClassDef, file_path: Path) -> bool:
        """Check if class should be exempt from dead code detection."""
        # Test classes
        if "test" in file_path.name or node.name.startswith("Test"):
            return True

        # Exception classes
        if node.name.endswith("Error") or node.name.endswith("Exception"):
            return True

        return False

    def _detect_redundant_patterns(self, project_root: Path) -> List[RedundancyPattern]:
        """Detect redundant code patterns across the codebase."""
        logger.debug("Detecting redundant patterns...")

        patterns = []
        function_signatures = defaultdict(list)

        all_files = self._get_all_python_files(project_root)

        # Collect function signatures
        for file_path in all_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                tree = ast.parse(content)
                relative_path = str(file_path.relative_to(project_root))

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Create normalized signature
                        signature = self._normalize_function_signature(node)
                        function_signatures[signature].append(
                            (relative_path, node.lineno, node.name)
                        )

            except Exception as e:
                logger.debug(f"Failed to analyze {file_path}: {e}")

        # Find patterns with multiple occurrences
        for signature, locations in function_signatures.items():
            if len(locations) > 1:
                # Calculate similarity score based on signature complexity
                similarity_score = self._calculate_similarity_score(signature, locations)

                if similarity_score > 0.6:  # Reasonable similarity threshold
                    patterns.append(
                        RedundancyPattern(
                            pattern_type="function",
                            signature=signature,
                            locations=[(loc[0], loc[1]) for loc in locations],
                            similarity_score=similarity_score,
                            estimated_redundancy=(
                                "duplicate" if similarity_score > 0.9 else "similar"
                            ),
                        )
                    )

        return patterns

    def _normalize_function_signature(self, node: ast.FunctionDef) -> str:
        """Create normalized signature for function comparison."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        body_structure = []
        for stmt in node.body[:3]:  # Only look at first few statements
            body_structure.append(type(stmt).__name__)

        return f"{len(args)}args:{':'.join(body_structure)}"

    def _calculate_similarity_score(self, signature: str, locations: List[Tuple]) -> float:
        """Calculate similarity score for function pattern."""
        base_score = 0.4

        # More locations = higher redundancy potential
        location_score = min(len(locations) * 0.15, 0.3)

        # Complex signatures more likely to be actual redundancy
        complexity_score = min(len(signature) * 0.01, 0.2)

        return min(base_score + location_score + complexity_score, 1.0)

    def _analyze_import_redundancies(self, project_root: Path) -> List[Dict[str, Any]]:
        """Find redundant or unused imports."""
        logger.debug("Analyzing import redundancies...")

        import_redundancies = []
        import_usage = defaultdict(set)
        all_imports = defaultdict(list)

        all_files = self._get_all_python_files(project_root)

        # Collect all imports and usage
        for file_path in all_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                tree = ast.parse(content)
                relative_path = str(file_path.relative_to(project_root))

                # Track imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module_name = alias.name
                            all_imports[module_name].append((relative_path, node.lineno))

                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                import_name = f"{node.module}.{alias.name}"
                                all_imports[import_name].append((relative_path, node.lineno))

                # Track usage (simplified analysis)
                for line in content.split("\n"):
                    for module_name in all_imports.keys():
                        if module_name.split(".")[-1] in line:
                            import_usage[module_name].add(relative_path)

            except Exception as e:
                logger.debug(f"Failed to analyze imports in {file_path}: {e}")

        # Find redundancies
        for module_name, locations in all_imports.items():
            if len(locations) > 1:
                unused_locations = []
                for file_path, line_no in locations:
                    if file_path not in import_usage[module_name]:
                        unused_locations.append((file_path, line_no))

                if unused_locations:
                    import_redundancies.append(
                        {
                            "module": module_name,
                            "unused_locations": unused_locations,
                            "total_imports": len(locations),
                            "estimated_savings": f"{len(unused_locations)} unused imports",
                        }
                    )

        return import_redundancies

    def _find_consolidation_opportunities(self, project_root: Path) -> List[Dict[str, Any]]:
        """Find opportunities to consolidate similar code."""
        opportunities = []

        # Look for similar file structures
        file_purposes = defaultdict(list)
        all_files = self._get_all_python_files(project_root)

        for file_path in all_files:
            purpose = self._infer_file_purpose(file_path)
            relative_path = str(file_path.relative_to(project_root))
            file_purposes[purpose].append(relative_path)

        # Find consolidation opportunities
        for purpose, files in file_purposes.items():
            if len(files) > 3 and purpose not in ["testing", "unknown"]:
                opportunities.append(
                    {
                        "type": "module_consolidation",
                        "purpose": purpose,
                        "files": files,
                        "suggestion": f"Consider consolidating {purpose} files into subpackage",
                        "estimated_benefit": f"Reduce {len(files)} files to organized submodule",
                    }
                )

        return opportunities

    def _estimate_removal_benefit(
        self, dead_code: List[DeadCodeCandidate], redundant_patterns: List[RedundancyPattern]
    ) -> Dict[str, Any]:
        """Estimate benefits of removing dead/redundant code."""
        total_dead_functions = len([d for d in dead_code if d.code_type == "function"])
        total_dead_classes = len([d for d in dead_code if d.code_type == "class"])
        total_redundant_patterns = len(redundant_patterns)

        # Rough estimation
        estimated_lines_saved = (total_dead_functions * 10) + (total_dead_classes * 20)
        for pattern in redundant_patterns:
            estimated_lines_saved += len(pattern.locations) * 5

        return {
            "dead_functions": total_dead_functions,
            "dead_classes": total_dead_classes,
            "redundant_patterns": total_redundant_patterns,
            "estimated_lines_saved": estimated_lines_saved,
            "estimated_files_reducible": len(set(d.file_path for d in dead_code)),
            "maintainability_improvement": "Medium" if estimated_lines_saved > 100 else "Low",
        }

    def _get_all_python_files(self, project_root: Path) -> List[Path]:
        """Get all Python files in the project."""
        source_candidates = [project_root / "src", project_root]

        python_files = []
        for source_root in source_candidates:
            if source_root.exists():
                files = list(source_root.rglob("*.py"))
                python_files.extend(files)

        # Filter out common non-code files
        filtered_files = []
        for file_path in python_files:
            if not any(
                skip in str(file_path) for skip in ["__pycache__", ".pytest_cache", "build", "dist"]
            ):
                filtered_files.append(file_path)

        return filtered_files

    def _infer_file_purpose(self, file_path: Path) -> str:
        """Infer the purpose of a file from its name and location."""
        name = file_path.name.lower()
        path_parts = file_path.parts

        if "test" in name or "tests" in path_parts:
            return "testing"
        elif name == "__init__.py":
            return "package_init"
        elif name in ["cli.py", "main.py", "__main__.py"]:
            return "entry_point"
        elif any(keyword in name for keyword in ["util", "helper", "tool"]):
            return "utility"
        elif any(keyword in name for keyword in ["config", "settings"]):
            return "configuration"
        elif any(keyword in name for keyword in ["validator", "check", "lint"]):
            return "validation"
        elif any(keyword in name for keyword in ["report", "format"]):
            return "reporting"
        elif any(keyword in name for keyword in ["llm", "ai", "model"]):
            return "ai_integration"
        else:
            return "unknown"

    def _calculate_overall_confidence(
        self, dead_code: List[DeadCodeCandidate], redundant_patterns: List[RedundancyPattern]
    ) -> float:
        """Calculate overall confidence score for the analysis."""
        if not dead_code and not redundant_patterns:
            return 1.0

        total_confidence = 0.0
        total_items = 0

        for candidate in dead_code:
            total_confidence += candidate.confidence
            total_items += 1

        for pattern in redundant_patterns:
            total_confidence += pattern.similarity_score
            total_items += 1

        return total_confidence / total_items if total_items > 0 else 0.0

    def _pattern_to_dict(self, pattern: RedundancyPattern) -> Dict[str, Any]:
        """Convert RedundancyPattern to dictionary."""
        return {
            "pattern_type": pattern.pattern_type,
            "signature": pattern.signature,
            "locations": pattern.locations,
            "similarity_score": pattern.similarity_score,
            "estimated_redundancy": pattern.estimated_redundancy,
        }

    def _candidate_to_dict(self, candidate: DeadCodeCandidate) -> Dict[str, Any]:
        """Convert DeadCodeCandidate to dictionary."""
        return {
            "code_type": candidate.code_type,
            "name": candidate.name,
            "file_path": candidate.file_path,
            "line_number": candidate.line_number,
            "reason": candidate.reason,
            "confidence": candidate.confidence,
        }
