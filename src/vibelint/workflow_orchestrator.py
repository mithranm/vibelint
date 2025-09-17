"""
Multi-level workflow orchestrator for comprehensive code quality assessment.

Coordinates tree-level, content-level, and deep analysis using specialized
LLM agents to catch organizational violations at different granularities.

vibelint/src/vibelint/workflow_orchestrator.py
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

from .context.prompts import AgentPrompts, AnalysisLevel
from .context.analyzer import ContextAnalyzer, TreeViolation
from .llm import LLMManager, LLMRequest, LLMRole
from .project_map import ProjectMapper
from .plugin_system import Finding, Severity

logger = logging.getLogger(__name__)

__all__ = ["AnalysisOrchestrator", "OrchestrationResult", "AnalysisReport"]


@dataclass
class OrchestrationResult:
    """Result of multi-level analysis orchestration."""
    success: bool
    analysis_duration: float
    reports_generated: List[str]
    total_findings: int
    critical_issues: int
    quick_wins: List[str]
    strategic_recommendations: List[str]
    error_message: Optional[str] = None


@dataclass
class AnalysisReport:
    """Comprehensive analysis report for development feedback."""
    timestamp: str
    project_root: str
    analysis_levels_completed: List[str]

    # Level-specific results
    tree_violations: List[Dict[str, Any]]
    content_findings: List[Dict[str, Any]]
    architectural_findings: List[Dict[str, Any]]

    # Synthesized recommendations
    executive_summary: Dict[str, Any]
    priority_actions: List[Dict[str, Any]]
    quick_wins: List[str]
    strategic_initiatives: List[str]

    # Metrics and tracking
    current_health_scores: Dict[str, float]
    improvement_targets: Dict[str, float]
    next_review_triggers: List[str]


class AnalysisOrchestrator:
    """Orchestrates multi-level code quality analysis using specialized agents."""

    def __init__(self, llm_manager: LLMManager, project_root: Path):
        self.llm = llm_manager
        self.project_root = project_root
        self.context_analyzer = ContextAnalyzer(project_root)
        self.project_mapper = ProjectMapper(project_root)

    async def run_comprehensive_analysis(
        self,
        target_files: Optional[List[Path]] = None,
        analysis_levels: Optional[List[str]] = None
    ) -> OrchestrationResult:
        """Run comprehensive multi-level analysis."""
        start_time = time.time()

        if analysis_levels is None:
            analysis_levels = [AnalysisLevel.TREE, AnalysisLevel.CONTENT, AnalysisLevel.DEEP]

        logger.info(f"Starting comprehensive analysis with levels: {analysis_levels}")

        try:
            # Step 1: Tree-level analysis (organizational structure)
            tree_results = {}
            if AnalysisLevel.TREE in analysis_levels:
                tree_results = await self._run_tree_analysis()

            # Step 2: Content-level analysis (file structure)
            content_results = {}
            if AnalysisLevel.CONTENT in analysis_levels:
                content_results = await self._run_content_analysis(target_files)

            # Step 3: Deep analysis (architectural assessment)
            deep_results = {}
            if AnalysisLevel.DEEP in analysis_levels:
                deep_results = await self._run_deep_analysis(target_files, tree_results, content_results)

            # Step 4: Synthesis and orchestration
            synthesis_result = await self._synthesize_results(
                tree_results, content_results, deep_results, analysis_levels
            )

            # Step 5: Generate reports
            report_paths = await self._generate_reports(synthesis_result)

            duration = time.time() - start_time

            return OrchestrationResult(
                success=True,
                analysis_duration=duration,
                reports_generated=report_paths,
                total_findings=synthesis_result.get("total_findings", 0),
                critical_issues=synthesis_result.get("critical_issues", 0),
                quick_wins=synthesis_result.get("quick_wins", []),
                strategic_recommendations=synthesis_result.get("strategic_initiatives", [])
            )

        except Exception as e:
            logger.error(f"Analysis orchestration failed: {e}", exc_info=True)
            duration = time.time() - start_time

            return OrchestrationResult(
                success=False,
                analysis_duration=duration,
                reports_generated=[],
                total_findings=0,
                critical_issues=0,
                quick_wins=[],
                strategic_recommendations=[],
                error_message=str(e)
            )

    async def _run_tree_analysis(self) -> Dict[str, Any]:
        """Run tree-level organizational analysis."""
        logger.info("Running tree-level analysis...")

        # Generate project map
        project_map = self.project_mapper.generate_project_map()

        # Use context analyzer for quick organizational checks
        quick_violations = self.context_analyzer.analyze_tree_level()

        # Prepare context for LLM analysis
        context_data = {
            "project_map": json.dumps(project_map, indent=2, default=str),
            "quick_violations": [asdict(v) for v in quick_violations]
        }

        prompt = AgentPrompts.get_prompt_for_analysis_level(AnalysisLevel.TREE)
        context = AgentPrompts.get_context_for_analysis(AnalysisLevel.TREE, context_data)

        # Use fast LLM for tree analysis
        llm_request = LLMRequest(
            content=f"{prompt}\n\n{context}",
            task_type="tree_analysis",
            max_tokens=2048,
            temperature=0.1
        )

        try:
            response = await self.llm.process_request(llm_request)
            llm_analysis = self._parse_llm_response(response["content"])

            return {
                "project_map": project_map,
                "quick_violations": [asdict(v) for v in quick_violations],
                "llm_analysis": llm_analysis,
                "organization_score": project_map.get("organization_metrics", {}).get("organization_score", 0.5)
            }

        except Exception as e:
            logger.warning(f"LLM tree analysis failed, using quick analysis only: {e}")
            return {
                "project_map": project_map,
                "quick_violations": [asdict(v) for v in quick_violations],
                "llm_analysis": {"violations": [], "organization_score": 0.5},
                "organization_score": project_map.get("organization_metrics", {}).get("organization_score", 0.5)
            }

    async def _run_content_analysis(self, target_files: Optional[List[Path]]) -> Dict[str, Any]:
        """Run content-level structural analysis."""
        logger.info("Running content-level analysis...")

        if target_files is None:
            # Discover Python files to analyze
            target_files = list(self.project_root.rglob("*.py"))
            target_files = [f for f in target_files if not self._should_skip_file(f)]

        file_analyses = []

        # Analyze up to 10 most important files to avoid overwhelming the LLM
        important_files = self._select_important_files(target_files)

        for file_path in important_files[:10]:
            try:
                file_analysis = await self._analyze_single_file(file_path)
                file_analyses.append(file_analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")

        return {
            "files_analyzed": len(file_analyses),
            "total_files": len(target_files),
            "file_analyses": file_analyses,
            "structural_health": self._calculate_structural_health(file_analyses)
        }

    async def _analyze_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file for structural issues."""
        try:
            content = file_path.read_text(encoding="utf-8")
            relative_path = file_path.relative_to(self.project_root)

            context_data = {
                "file_path": str(relative_path),
                "file_content": content,
                "file_size": len(content),
                "file_purpose": self._infer_file_purpose(file_path),
                "dependencies": self._extract_imports(content),
                "exports": self._extract_exports(content)
            }

            prompt = AgentPrompts.get_prompt_for_analysis_level(AnalysisLevel.CONTENT)
            context = AgentPrompts.get_context_for_analysis(AnalysisLevel.CONTENT, context_data)

            llm_request = LLMRequest(
                content=f"{prompt}\n\n{context}",
                task_type="content_analysis",
                max_tokens=1500,
                temperature=0.1
            )

            response = await self.llm.process_request(llm_request)
            return {
                "file_path": str(relative_path),
                "analysis": self._parse_llm_response(response["content"]),
                "metadata": {
                    "size": len(content),
                    "purpose": context_data["file_purpose"],
                    "lines": content.count('\n') + 1
                }
            }

        except Exception as e:
            logger.warning(f"Single file analysis failed for {file_path}: {e}")
            return {
                "file_path": str(file_path.relative_to(self.project_root)),
                "analysis": {"findings": [], "file_health": {}},
                "error": str(e)
            }

    async def _run_deep_analysis(
        self,
        target_files: Optional[List[Path]],
        tree_results: Dict[str, Any],
        content_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run deep architectural analysis."""
        logger.info("Running deep architectural analysis...")

        # Check if orchestrator LLM is available
        if not self.llm.is_llm_available(LLMRole.ORCHESTRATOR):
            logger.warning("Orchestrator LLM not available, skipping deep analysis")
            return {"skipped": True, "reason": "Orchestrator LLM not configured"}

        # Select key files for deep analysis
        if target_files is None:
            target_files = list(self.project_root.rglob("*.py"))
            target_files = [f for f in target_files if not self._should_skip_file(f)]

        key_files = self._select_key_files_for_deep_analysis(target_files)

        # Prepare comprehensive context
        files_content = {}
        for file_path in key_files[:5]:  # Limit to 5 files for context size
            try:
                content = file_path.read_text(encoding="utf-8")
                relative_path = file_path.relative_to(self.project_root)
                files_content[str(relative_path)] = content
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        context_data = {
            "project_context": f"Project: {self.project_root.name}",
            "files_content": self._format_files_for_analysis(files_content),
            "tree_results": json.dumps(tree_results, indent=2, default=str),
            "content_results": json.dumps(content_results, indent=2, default=str)
        }

        prompt = AgentPrompts.get_prompt_for_analysis_level(AnalysisLevel.DEEP)
        context = AgentPrompts.get_context_for_analysis(AnalysisLevel.DEEP, context_data)

        # Use orchestrator LLM for deep analysis
        llm_request = LLMRequest(
            content=f"{prompt}\n\n{context}",
            task_type="architectural_analysis",
            max_tokens=4096,
            temperature=0.2
        )

        try:
            response = await self.llm.process_request(llm_request)
            deep_analysis = self._parse_llm_response(response["content"])

            return {
                "files_analyzed": list(files_content.keys()),
                "architectural_analysis": deep_analysis,
                "analysis_scope": "comprehensive"
            }

        except Exception as e:
            logger.error(f"Deep analysis failed: {e}")
            return {
                "files_analyzed": [],
                "architectural_analysis": {"architectural_findings": [], "code_smells": []},
                "error": str(e)
            }

    async def _synthesize_results(
        self,
        tree_results: Dict[str, Any],
        content_results: Dict[str, Any],
        deep_results: Dict[str, Any],
        analysis_levels: List[str]
    ) -> Dict[str, Any]:
        """Synthesize multi-level results into actionable recommendations."""
        logger.info("Synthesizing analysis results...")

        # Count total findings
        total_findings = 0
        critical_issues = 0

        if tree_results.get("quick_violations"):
            total_findings += len(tree_results["quick_violations"])
        if tree_results.get("llm_analysis", {}).get("violations"):
            total_findings += len(tree_results["llm_analysis"]["violations"])

        for file_analysis in content_results.get("file_analyses", []):
            findings = file_analysis.get("analysis", {}).get("findings", [])
            total_findings += len(findings)
            critical_issues += len([f for f in findings if f.get("severity") == "BLOCK"])

        if deep_results.get("architectural_analysis"):
            arch_findings = deep_results["architectural_analysis"].get("architectural_findings", [])
            code_smells = deep_results["architectural_analysis"].get("code_smells", [])
            total_findings += len(arch_findings) + len(code_smells)
            critical_issues += len([f for f in arch_findings if f.get("severity") == "BLOCK"])

        # Prepare synthesis context
        synthesis_data = {
            "tree_analysis": tree_results,
            "content_analysis": content_results,
            "deep_analysis": deep_results,
            "total_findings": total_findings,
            "critical_issues": critical_issues
        }

        context = f"""ANALYSIS SYNTHESIS REQUEST

Analysis Results Summary:
- Tree Analysis: {len(tree_results.get('quick_violations', []))} organizational violations
- Content Analysis: {content_results.get('files_analyzed', 0)} files analyzed
- Deep Analysis: {len(deep_results.get('files_analyzed', []))} files examined

Detailed Results:
{json.dumps(synthesis_data, indent=2, default=str)}

Synthesize these results into actionable development feedback with prioritized recommendations."""

        prompt = AgentPrompts.get_orchestrator_prompt()

        llm_request = LLMRequest(
            content=f"{prompt}\n\n{context}",
            task_type="synthesis",
            max_tokens=3072,
            temperature=0.1
        )

        try:
            response = await self.llm.process_request(llm_request)
            synthesis = self._parse_llm_response(response["content"])

            # Add computed metrics
            synthesis["total_findings"] = total_findings
            synthesis["critical_issues"] = critical_issues
            synthesis["analysis_levels_completed"] = analysis_levels

            return synthesis

        except Exception as e:
            logger.warning(f"Synthesis failed, generating basic summary: {e}")

            # Fallback synthesis
            return {
                "executive_summary": {
                    "overall_health": 0.7,
                    "critical_issues": critical_issues,
                    "improvement_opportunities": total_findings,
                    "estimated_effort": "Unknown"
                },
                "priority_actions": [],
                "quick_wins": ["Review organizational violations", "Address structural issues"],
                "strategic_initiatives": ["Consider architectural improvements"],
                "total_findings": total_findings,
                "critical_issues": critical_issues,
                "analysis_levels_completed": analysis_levels
            }

    async def _generate_reports(self, synthesis_result: Dict[str, Any]) -> List[str]:
        """Generate analysis reports for development feedback."""
        reports_dir = self.project_root / ".vibelint-reports"
        reports_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_paths = []

        # Generate comprehensive JSON report
        json_report_path = reports_dir / f"comprehensive_analysis_{timestamp}.json"
        try:
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump(synthesis_result, f, indent=2, default=str)
            report_paths.append(str(json_report_path))
            logger.info(f"Generated JSON report: {json_report_path}")
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")

        # Generate human-readable summary
        summary_path = reports_dir / f"analysis_summary_{timestamp}.md"
        try:
            summary_content = self._generate_markdown_summary(synthesis_result)
            summary_path.write_text(summary_content, encoding='utf-8')
            report_paths.append(str(summary_path))
            logger.info(f"Generated summary report: {summary_path}")
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")

        return report_paths

    def _generate_markdown_summary(self, synthesis: Dict[str, Any]) -> str:
        """Generate human-readable markdown summary."""
        executive = synthesis.get("executive_summary", {})
        priority_actions = synthesis.get("priority_actions", [])
        quick_wins = synthesis.get("quick_wins", [])
        strategic = synthesis.get("strategic_initiatives", [])

        content = f"""# Vibelint Analysis Report

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
Project: {self.project_root.name}

## Executive Summary

- **Overall Health**: {executive.get('overall_health', 'Unknown')}
- **Critical Issues**: {executive.get('critical_issues', 0)}
- **Improvement Opportunities**: {executive.get('improvement_opportunities', 0)}
- **Estimated Effort**: {executive.get('estimated_effort', 'Unknown')}

## Priority Actions

"""

        for i, action in enumerate(priority_actions[:5], 1):
            content += f"""### {i}. {action.get('title', 'Unknown Action')} ({action.get('priority', 'P?')})

{action.get('description', 'No description available')}

**Effort**: {action.get('effort_hours', '?')} hours
**Risk if ignored**: {action.get('risk_if_ignored', 'Unknown')}

"""

        if quick_wins:
            content += "\n## Quick Wins\n\n"
            for win in quick_wins:
                content += f"- {win}\n"

        if strategic:
            content += "\n## Strategic Initiatives\n\n"
            for initiative in strategic:
                content += f"- {initiative}\n"

        return content

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response with fallback."""
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in LLM response")
                return {"error": "No JSON in response", "raw_response": response}

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return {"error": "Invalid JSON", "raw_response": response}

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped in analysis."""
        skip_patterns = [
            "__pycache__", ".git", ".pytest_cache", "build", "dist",
            ".venv", "venv", ".mypy_cache", ".tox"
        ]
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _select_important_files(self, files: List[Path]) -> List[Path]:
        """Select most important files for analysis."""
        # Prioritize entry points, main modules, and larger files
        scored_files = []

        for file_path in files:
            score = 0
            name = file_path.name.lower()

            # Entry points and main modules
            if name in ["__init__.py", "main.py", "__main__.py", "cli.py"]:
                score += 10
            elif "main" in name or "cli" in name:
                score += 5

            # Size-based scoring
            try:
                size = file_path.stat().st_size
                if size > 5000:  # Larger files likely more important
                    score += 3
                elif size > 1000:
                    score += 1
            except OSError:
                pass

            scored_files.append((score, file_path))

        # Sort by score (descending)
        scored_files.sort(key=lambda x: x[0], reverse=True)
        return [f[1] for f in scored_files]

    def _select_key_files_for_deep_analysis(self, files: List[Path]) -> List[Path]:
        """Select key files for architectural analysis."""
        # Similar to important files but focus on architectural significance
        return self._select_important_files(files)

    def _infer_file_purpose(self, file_path: Path) -> str:
        """Infer the purpose of a file."""
        name = file_path.name.lower()

        if name == "__init__.py":
            return "package_init"
        elif name in ["main.py", "__main__.py", "cli.py"]:
            return "entry_point"
        elif "test" in name:
            return "testing"
        elif "config" in name:
            return "configuration"
        elif any(keyword in name for keyword in ["util", "helper", "tool"]):
            return "utility"
        else:
            return "module"

    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from Python code."""
        imports = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        return imports

    def _extract_exports(self, content: str) -> List[str]:
        """Extract __all__ exports from Python code."""
        exports = []
        in_all = False
        all_content = ""

        for line in content.split('\n'):
            if '__all__' in line:
                in_all = True
                all_content = line
            elif in_all:
                all_content += line
                if ']' in line:
                    break

        if all_content:
            try:
                # Simple extraction - could be improved with AST
                import re
                matches = re.findall(r'"([^"]+)"', all_content)
                matches.extend(re.findall(r"'([^']+)'", all_content))
                exports = matches
            except Exception:
                pass

        return exports

    def _calculate_structural_health(self, file_analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall structural health scores."""
        if not file_analyses:
            return {"overall": 0.5}

        total_files = len(file_analyses)
        healthy_files = sum(1 for analysis in file_analyses
                          if analysis.get("analysis", {}).get("file_health", {}).get("overall", False))

        return {
            "overall": healthy_files / total_files if total_files > 0 else 0.5,
            "files_analyzed": total_files,
            "healthy_files": healthy_files
        }

    def _format_files_for_analysis(self, files_content: Dict[str, str]) -> str:
        """Format multiple files for LLM analysis."""
        formatted = ""
        for file_path, content in files_content.items():
            formatted += f"\n\n=== {file_path} ===\n```python\n{content}\n```"
        return formatted