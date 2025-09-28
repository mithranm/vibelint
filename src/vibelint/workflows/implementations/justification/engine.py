"""
Refactored JustificationEngine - modular architecture.

This is the new modular implementation that replaces the monolithic 2,243 LOC engine.
Uses composition with separate components for filesystem, static analysis, LLM orchestration, and reporting.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .filesystem import FileSystemCrawler
from .llm_orchestrator import JustificationLLMOrchestrator
from .reporter import JustificationReporter
from .static_analyzer import StaticAnalyzer

logger = logging.getLogger(__name__)


class JustificationEngine:
    """
    Modular justification engine with clear separation of concerns.

    Uses composition-based architecture:
    - FileSystemCrawler: Handles file discovery and basic operations
    - StaticAnalyzer: Performs AST parsing and static analysis
    - JustificationLLMOrchestrator: Manages all LLM interactions
    - JustificationReporter: Handles report generation and quality gates
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize components
        self.filesystem = FileSystemCrawler(config)
        self.static_analyzer = StaticAnalyzer(config)
        self.reporter = JustificationReporter(config)

        # Initialize LLM orchestrator if available
        self.llm_orchestrator = None
        self._initialize_llm_orchestrator()

        # Track quality gate result for CLI access
        self.last_quality_gate_result = None

    def _initialize_llm_orchestrator(self):
        """Initialize LLM orchestrator if LLM manager is available."""
        try:
            from vibelint.llm.manager import LLMManager
            from vibelint.llm.llm_config import get_llm_config

            # Try to get LLM config from passed config first, then fall back to auto-discovery
            llm_config = self.config.get("llm") if "llm" in self.config else None
            if not llm_config:
                # Auto-discover config from pyproject.toml
                discovered_config = get_llm_config()
                if discovered_config:
                    # Wrap in the expected structure for LLMManager
                    config_for_manager = {"llm": discovered_config}
                    llm_manager = LLMManager(config_for_manager)
                else:
                    logger.warning("No LLM configuration found")
                    return
            else:
                llm_manager = LLMManager(self.config)

            self.llm_orchestrator = JustificationLLMOrchestrator(llm_manager, self.config)
            logger.debug("LLM orchestrator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM orchestrator: {e}")

    def run_justification_workflow(self, directory_path: Path) -> str:
        """
        Run the complete justification workflow with modular components.

        Args:
            directory_path: Path to analyze

        Returns:
            Comprehensive justification report
        """
        start_time = time.time()
        logger.info("Starting modular justification workflow...")

        try:
            # Step 1: File system discovery
            logger.info("Phase 1: File system discovery")
            project_tree = self.filesystem.generate_project_tree(directory_path)
            python_files = self.filesystem.discover_python_files(directory_path)
            logger.info(f"Discovered {len(python_files)} Python files")

            # Step 2: Static analysis
            logger.info("Phase 2: Static analysis")
            file_analyses = {}
            for file_path in python_files:
                content = self.filesystem.read_file_safely(file_path)
                if content:
                    analysis = self.static_analyzer.analyze_file_structure(file_path, content)
                    file_analyses[file_path] = analysis

            # Step 3: Dependency analysis
            logger.info("Phase 3: Dependency analysis")
            dependency_graph = self.static_analyzer.build_dependency_graph(file_analyses)
            circular_deps = self.static_analyzer.detect_circular_imports(dependency_graph)

            # Step 4: Detect misplaced files
            logger.info("Phase 4: Misplaced file detection")
            misplaced_files = self.static_analyzer.detect_misplaced_files(file_analyses, directory_path)

            # Step 5: LLM enhancement (if available)
            enhanced_tree = project_tree
            backup_files = []

            if self.llm_orchestrator:
                logger.info("Phase 5: LLM enhancement")

                # Enhance tree with LLM file analysis
                enhanced_sections = []
                enhanced_sections.append("# Enhanced Project Analysis\n")

                for file_path, analysis in list(file_analyses.items())[:50]:  # Limit for performance
                    content = self.filesystem.read_file_safely(file_path)
                    if content:
                        relative_path = self.filesystem.get_file_relative_path(file_path, directory_path)
                        purpose = self.llm_orchestrator.analyze_file_purpose(file_path, analysis, content)
                        enhanced_sections.append(f"**{relative_path}**: {purpose}")

                enhanced_tree = project_tree + "\n\n" + "\n".join(enhanced_sections)

                # Detect backup files
                backup_files = self.llm_orchestrator.detect_backup_files(directory_path)

            # Step 6: Compile analysis results
            dependency_analysis = {
                "files_analyzed": list(file_analyses.keys()),
                "entry_points": [],  # Could be enhanced later
                "circular_imports": circular_deps,
                "misplaced_files": misplaced_files,
                "backup_files": backup_files
            }

            # Step 7: Generate reports
            logger.info("Phase 6: Report generation")
            analysis_id = time.strftime("%Y%m%d_%H%M%S")

            # Generate main justification report
            llm_logs = self.llm_orchestrator.get_llm_logs() if self.llm_orchestrator else []
            justification_report = self.reporter.generate_justification_report(
                enhanced_tree, dependency_analysis, directory_path, llm_logs
            )

            # Save main report
            main_report_file = self.reporter.save_justification_report(
                justification_report, directory_path, analysis_id
            )

            # Step 8: Final LLM analysis and quality gate
            final_analysis = "No LLM orchestrator available for final analysis"
            if self.llm_orchestrator:
                logger.info("Phase 7: Final architectural analysis")

                static_issues = {
                    "backup_files": backup_files,
                    "misplaced_files": misplaced_files,
                    "circular_dependencies": circular_deps
                }

                final_analysis = self.llm_orchestrator.perform_final_analysis(
                    justification_report, static_issues
                )

            # Save final analysis
            final_report_file = self.reporter.save_final_analysis(final_analysis, analysis_id)

            # Create quality assessment
            assessment_file = self.reporter.create_quality_assessment_report(final_analysis, analysis_id)

            # Save logs
            self.reporter.save_logs(llm_logs, analysis_id)

            # Step 9: Quality gate enforcement
            quality_result = self.reporter.enforce_quality_gate(final_analysis)
            self.last_quality_gate_result = quality_result["gate_passed"]

            # Generate summary
            analysis_time = time.time() - start_time
            report_files = [main_report_file, final_report_file, assessment_file]
            summary = self.reporter.generate_summary_report(
                len(python_files), analysis_time, quality_result, report_files
            )

            logger.info(f"Justification workflow completed in {analysis_time:.1f} seconds")
            logger.info(f"Quality gate: {'PASSED' if quality_result['gate_passed'] else 'FAILED'}")

            # Print results for CLI
            if not quality_result["gate_passed"]:
                report_paths = ", ".join(str(f) for f in report_files)
                print(f"Quality gate FAILED: No LGTM found in final analysis")
                print(f"Review reports: {report_paths}")

            return justification_report

        except Exception as e:
            logger.error(f"Justification workflow failed: {e}")
            raise

    def run_justification_safe(self, directory_path: Path) -> dict:
        """
        Run justification workflow in safe mode - never exits, returns all results.

        Returns:
            dict with success status, results, and error info
        """
        try:
            result = self.run_justification_workflow(directory_path)
            quality_gate = self.last_quality_gate_result

            return {
                "success": True,
                "report": result,
                "quality_gate_passed": quality_gate,
                "exit_code": 0 if quality_gate else 2,
                "error": None
            }

        except Exception as e:
            logger.error(f"Safe justification workflow failed: {e}")
            return {
                "success": False,
                "report": None,
                "quality_gate_passed": False,
                "exit_code": 1,
                "error": str(e)
            }

    def get_quality_gate_result(self) -> Optional[bool]:
        """Get the last quality gate result."""
        return self.last_quality_gate_result

    def enforce_quality_gate(self, safe_mode: bool = False) -> Optional[dict]:
        """
        Enforce quality gate based on last analysis.

        Args:
            safe_mode: If True, doesn't exit process

        Returns:
            Quality gate result dict
        """
        if self.last_quality_gate_result is None:
            logger.warning("No quality gate result available - run analysis first")
            return None

        result = {
            "gate_passed": self.last_quality_gate_result,
            "exit_code": 0 if self.last_quality_gate_result else 2
        }

        if not safe_mode and not self.last_quality_gate_result:
            import sys
            logger.error("Quality gate FAILED - exiting with code 2")
            sys.exit(2)

        return result

    # Backward compatibility methods for existing workflow wrapper
    def justify_file(self, file_path: Path, rule_id: str = None):
        """Legacy compatibility method for CLI."""
        # Simple file analysis for backward compatibility
        content = self.filesystem.read_file_safely(file_path)
        if not content:
            return "Could not read file"

        analysis = self.static_analyzer.analyze_file_structure(file_path, content)

        if self.llm_orchestrator:
            return self.llm_orchestrator.analyze_file_purpose(file_path, analysis, content)
        else:
            return self.llm_orchestrator._generate_pattern_based_justification(file_path, analysis)

    def save_session_logs(self, output_dir: Path):
        """Legacy compatibility method."""
        if self.llm_orchestrator:
            logs = self.llm_orchestrator.get_llm_logs()
            log_file = output_dir / "session_logs.json"

            import json
            log_file.write_text(json.dumps(logs, indent=2))
            logger.info(f"Session logs saved to {log_file}")
            return str(log_file)

        return None