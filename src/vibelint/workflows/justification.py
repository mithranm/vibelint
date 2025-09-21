#!/usr/bin/env python3
"""
File and Method Existence Justification Workflow

A comprehensive workflow that analyzes every file and method to justify their existence,
then uses embeddings to find similar justifications and identify redundancies.

This is implemented as a proper vibelint workflow following the BaseWorkflow interface.
"""

import asyncio
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from vibelint.workflows.base import BaseWorkflow, WorkflowResult, WorkflowStatus, WorkflowConfig, WorkflowMetrics
from vibelint.workflows.justification_analysis import JustificationAnalysisWorkflow


class FileJustificationWorkflow(BaseWorkflow):
    """Workflow for comprehensive file and method existence justification analysis."""

    # Workflow identification
    workflow_id = "file_justification"
    name = "File and Method Existence Justification"
    description = "Analyzes every file and method to justify existence and find redundancies"
    version = "1.0.0"

    # Workflow categorization
    category = "analysis"
    tags = {"justification", "redundancy", "embeddings", "architecture"}

    def __init__(self, config: Optional[WorkflowConfig] = None):
        """Initialize the justification workflow."""
        super().__init__(config)

        # Workflow-specific configuration
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.85) if config else 0.85
        self.min_complexity = getattr(config, 'min_complexity', 2) if config else 2
        self.enable_embeddings = getattr(config, 'enable_embeddings', True) if config else True

        # Initialize the analysis engine
        analysis_config = {
            "similarity_threshold": self.similarity_threshold,
            "min_complexity": self.min_complexity,
            "model_name": "all-MiniLM-L6-v2" if self.enable_embeddings else None
        }
        self.analysis_workflow = JustificationAnalysisWorkflow(analysis_config)

    async def execute(self, project_root: Path, context: Dict[str, Any]) -> WorkflowResult:
        """Execute the justification analysis workflow."""

        self.metrics.start_time = time.time()
        self._status = WorkflowStatus.RUNNING

        try:
            # Get all files to analyze
            all_files = await self._get_all_files(project_root)
            self.metrics.files_processed = len(all_files)

            # Separate Python and non-Python files
            python_files = [f for f in all_files if f.suffix == '.py']
            non_python_files = [f for f in all_files if f.suffix != '.py']

            findings = []
            artifacts = {}

            # Analyze Python files with content
            for file_path in python_files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    result = self.analysis_workflow.analyze_file(file_path, content)

                    # Convert to findings format
                    file_findings = await self._convert_to_findings(file_path, result)
                    findings.extend(file_findings)

                except Exception as e:
                    self.metrics.errors_encountered += 1
                    findings.append({
                        "type": "error",
                        "file": str(file_path),
                        "message": f"Failed to analyze Python file: {e}",
                        "severity": "error"
                    })

            # Analyze non-Python files
            for file_path in non_python_files:
                try:
                    result = self.analysis_workflow.analyze_file(file_path)

                    # Convert to findings format
                    file_findings = await self._convert_to_findings(file_path, result)
                    findings.extend(file_findings)

                except Exception as e:
                    self.metrics.errors_encountered += 1
                    findings.append({
                        "type": "error",
                        "file": str(file_path),
                        "message": f"Failed to analyze file: {e}",
                        "severity": "error"
                    })

            # Generate comprehensive analysis
            comprehensive_report = self.analysis_workflow.generate_report()

            # Find redundancies
            redundancies = self.analysis_workflow.find_redundancies()

            # Convert redundancies to findings
            redundancy_findings = await self._convert_redundancies_to_findings(redundancies)
            findings.extend(redundancy_findings)

            # Prepare artifacts
            artifacts = {
                "comprehensive_report": comprehensive_report,
                "redundancy_clusters": [self._serialize_cluster(r) for r in redundancies],
                "file_justifications": {
                    k: self._serialize_justification(v)
                    for k, v in self.analysis_workflow.file_justifications.items()
                },
                "method_justifications": [
                    self._serialize_method_justification(m)
                    for m in self.analysis_workflow.method_justifications
                ],
                "analysis_summary": {
                    "total_files_analyzed": len(all_files),
                    "python_files_analyzed": len(python_files),
                    "non_python_files_analyzed": len(non_python_files),
                    "total_methods_analyzed": len(self.analysis_workflow.method_justifications),
                    "redundancy_clusters_found": len(redundancies),
                    "embedding_analysis_enabled": self.analysis_workflow.model is not None
                }
            }

            # Generate recommendations
            recommendations = await self._generate_recommendations(comprehensive_report, redundancies)

            # Finalize metrics
            self.metrics.end_time = time.time()
            self.metrics.finalize()
            self.metrics.findings_generated = len(findings)
            self.metrics.confidence_score = self._calculate_confidence_score(comprehensive_report)

            # Update custom metrics
            self.metrics.custom_metrics = {
                "redundancy_clusters": len(redundancies),
                "high_similarity_clusters": len([r for r in redundancies if r.similarity_score > 0.95]),
                "files_with_unclear_purpose": comprehensive_report["summary"]["files_with_unclear_purpose"],
                "methods_needing_documentation": comprehensive_report["summary"]["methods_without_documentation"],
                "total_redundant_files": comprehensive_report["redundancy_analysis"]["total_redundant_files"],
                "total_redundant_methods": comprehensive_report["redundancy_analysis"]["total_redundant_methods"]
            }

            self._status = WorkflowStatus.COMPLETED

            return WorkflowResult(
                workflow_id=self.workflow_id,
                status=self._status,
                metrics=self.metrics,
                findings=findings,
                artifacts=artifacts,
                recommendations=recommendations
            )

        except Exception as e:
            self.metrics.end_time = time.time()
            self.metrics.finalize()
            self.metrics.errors_encountered += 1
            self._status = WorkflowStatus.FAILED

            return WorkflowResult(
                workflow_id=self.workflow_id,
                status=self._status,
                metrics=self.metrics,
                error_message=str(e)
            )

    def get_required_inputs(self) -> Set[str]:
        """Get set of required input data keys."""
        return set()  # This workflow analyzes files directly, no specific inputs required

    def get_produced_outputs(self) -> Set[str]:
        """Get set of output data keys this workflow produces."""
        return {
            "file_justifications",
            "method_justifications",
            "redundancy_clusters",
            "justification_report",
            "consolidation_opportunities"
        }

    def get_dependencies(self) -> List[str]:
        """Get list of workflow dependencies."""
        return []  # This workflow has no dependencies

    async def _get_all_files(self, project_root: Path) -> List[Path]:
        """Get all files to analyze using git ls-files if available."""
        try:
            # Try git ls-files first
            result = await asyncio.create_subprocess_exec(
                'git', 'ls-files',
                cwd=project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                # Convert relative paths to absolute paths
                tracked_files = []
                for line in stdout.decode().strip().split('\n'):
                    if line:  # Skip empty lines
                        file_path = project_root / line
                        if file_path.exists():  # Only include existing files
                            tracked_files.append(file_path)
                return tracked_files

        except Exception:
            pass  # Fall back to filesystem scanning

        # Fallback: scan filesystem
        return self._get_all_relevant_files(project_root)

    def _get_all_relevant_files(self, project_root: Path) -> List[Path]:
        """Get all relevant files (fallback when VCS is not available)."""
        all_files = []

        # Patterns to exclude
        exclude_patterns = {
            '__pycache__', '.pytest_cache', 'node_modules', '.tox', 'venv', 'env',
            '.git', '.hg', '.svn', 'build', 'dist', '.eggs', '*.egg-info',
            '.mypy_cache', '.coverage', 'htmlcov', '.DS_Store'
        }

        for file_path in project_root.rglob('*'):
            if file_path.is_file():
                # Skip if any part of the path matches exclude patterns
                if not any(pattern in str(file_path) for pattern in exclude_patterns):
                    all_files.append(file_path)

        return all_files

    async def _convert_to_findings(self, file_path: Path, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert analysis result to vibelint findings format."""
        findings = []

        if "error" in result:
            findings.append({
                "type": "error",
                "file": str(file_path),
                "message": result["error"],
                "severity": "error"
            })
            return findings

        file_justification = result.get("file_justification", {})
        analysis_quality = result.get("analysis_quality", {})

        # Generate findings based on analysis quality
        quality_score = analysis_quality.get("quality_score", 0)
        issues = analysis_quality.get("issues", [])

        if quality_score < 50:
            findings.append({
                "type": "justification",
                "file": str(file_path),
                "message": f"Poor justification quality (score: {quality_score})",
                "severity": "warning",
                "details": {
                    "primary_purpose": file_justification.get("primary_purpose"),
                    "issues": issues,
                    "quality_score": quality_score
                }
            })

        # Check for specific issues
        if file_justification.get("primary_purpose", "").startswith("unclear"):
            findings.append({
                "type": "unclear_purpose",
                "file": str(file_path),
                "message": "File purpose is unclear - consider adding documentation",
                "severity": "info",
                "details": {
                    "primary_purpose": file_justification.get("primary_purpose"),
                    "suggestion": "Add clear module docstring or README explaining purpose"
                }
            })

        # Check method-level issues
        for method in result.get("method_justifications", []):
            if method.get("complexity_score", 0) > 5 and not method.get("docstring"):
                findings.append({
                    "type": "undocumented_complexity",
                    "file": str(file_path),
                    "line": method.get("line_number", 1),
                    "message": f"Complex method '{method.get('method_name')}' lacks documentation",
                    "severity": "warning",
                    "details": {
                        "method_name": method.get("method_name"),
                        "complexity_score": method.get("complexity_score"),
                        "suggestion": "Add docstring explaining method purpose and complexity"
                    }
                })

        return findings

    async def _convert_redundancies_to_findings(self, redundancies) -> List[Dict[str, Any]]:
        """Convert redundancy clusters to findings."""
        findings = []

        for cluster in redundancies:
            severity = "error" if cluster.similarity_score > 0.95 else "warning"

            findings.append({
                "type": "redundancy_cluster",
                "message": f"{cluster.cluster_type.title()} redundancy cluster detected",
                "severity": severity,
                "details": {
                    "cluster_type": cluster.cluster_type,
                    "similarity_score": cluster.similarity_score,
                    "common_purpose": cluster.common_purpose,
                    "items": cluster.items,
                    "recommendation": cluster.recommendation,
                    "item_count": len(cluster.items)
                }
            })

        return findings

    async def _generate_recommendations(self, comprehensive_report: Dict[str, Any],
                                      redundancies) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # High-priority redundancies
        high_priority = [r for r in redundancies if r.similarity_score > 0.9]
        if high_priority:
            recommendations.append(
                f"🔥 HIGH PRIORITY: Consolidate {len(high_priority)} clusters with >90% similarity "
                f"affecting {sum(len(r.items) for r in high_priority)} items"
            )

        # Files with unclear purpose
        unclear_files = comprehensive_report["summary"]["files_with_unclear_purpose"]
        if unclear_files > 0:
            recommendations.append(
                f"📝 Add clear documentation to {unclear_files} files with unclear purpose"
            )

        # Methods needing documentation
        undocumented_methods = comprehensive_report["summary"]["methods_without_documentation"]
        if undocumented_methods > 0:
            recommendations.append(
                f"📚 Document {undocumented_methods} complex methods lacking documentation"
            )

        # File consolidation opportunities
        file_redundancies = [r for r in redundancies if r.cluster_type == "file"]
        if file_redundancies:
            recommendations.append(
                f"📁 Review {len(file_redundancies)} file clusters for consolidation opportunities"
            )

        # Method extraction opportunities
        method_redundancies = [r for r in redundancies if r.cluster_type == "method"]
        if method_redundancies:
            recommendations.append(
                f"🔧 Extract common functionality from {len(method_redundancies)} method clusters"
            )

        return recommendations

    def _calculate_confidence_score(self, comprehensive_report: Dict[str, Any]) -> float:
        """Calculate overall confidence in the analysis."""
        summary = comprehensive_report.get("summary", {})

        total_files = summary.get("total_files_analyzed", 1)
        unclear_files = summary.get("files_with_unclear_purpose", 0)
        embedding_enabled = summary.get("embedding_analysis_enabled", False)

        # Base confidence
        confidence = 0.8 if embedding_enabled else 0.6

        # Reduce confidence for unclear purposes
        clarity_ratio = 1.0 - (unclear_files / total_files)
        confidence *= clarity_ratio

        return min(confidence, 1.0)

    def _serialize_cluster(self, cluster) -> Dict[str, Any]:
        """Serialize a redundancy cluster for JSON export."""
        return {
            "cluster_type": cluster.cluster_type,
            "similarity_score": cluster.similarity_score,
            "items": cluster.items,
            "common_purpose": cluster.common_purpose,
            "recommendation": cluster.recommendation
        }

    def _serialize_justification(self, justification) -> Dict[str, Any]:
        """Serialize a file justification for JSON export."""
        from dataclasses import asdict
        return asdict(justification)

    def _serialize_method_justification(self, method) -> Dict[str, Any]:
        """Serialize a method justification for JSON export."""
        from dataclasses import asdict
        return asdict(method)

    def _validate_configuration(self):
        """Validate workflow configuration."""
        # Only validate if attributes are set (during actual usage, not registry creation)
        if hasattr(self, 'similarity_threshold'):
            if self.similarity_threshold < 0.5 or self.similarity_threshold > 1.0:
                raise ValueError("Similarity threshold must be between 0.5 and 1.0")

        if hasattr(self, 'min_complexity'):
            if self.min_complexity < 1:
                raise ValueError("Minimum complexity must be at least 1")


# Register the workflow
__all__ = ["FileJustificationWorkflow"]