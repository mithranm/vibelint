"""
Workflow wrapper for the JustificationEngine.

This provides a proper BaseWorkflow interface while preserving the existing
JustificationEngine implementation for backward compatibility.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

from ...core.base import BaseWorkflow, WorkflowConfig, WorkflowResult, WorkflowStatus
from .engine import JustificationEngine

logger = logging.getLogger(__name__)


class JustificationWorkflow(BaseWorkflow):
    """Workflow wrapper for justification analysis."""

    # Workflow identification
    workflow_id = "justification"
    name = "Code Justification Analysis"
    description = "Analyze and justify code decisions using static analysis and LLM reasoning"
    version = "2.0"
    category = "analysis"
    tags = {"code-quality", "llm-analysis", "documentation"}

    def __init__(self, config: Optional[WorkflowConfig] = None):
        super().__init__(config)
        self._engine = None

    def _get_engine(self) -> JustificationEngine:
        """Lazy-load the justification engine."""
        if self._engine is None:
            # Convert WorkflowConfig to engine config format
            engine_config = {}
            if self.config and hasattr(self.config, 'parameters'):
                engine_config.update(self.config.parameters)

            self._engine = JustificationEngine(engine_config)
        return self._engine

    async def execute(self, project_root: Path, context: Dict[str, Any]) -> WorkflowResult:
        """Execute the justification workflow."""
        context = context or {}

        try:
            self._status = WorkflowStatus.RUNNING
            engine = self._get_engine()

            # Use project_root as target directory
            target_dir = str(project_root)

            # Run the analysis
            result = engine.run_full_analysis(target_dir)

            # Convert engine result to WorkflowResult
            workflow_result = WorkflowResult(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.COMPLETED,
                data=result,
                artifacts=result.get('reports', {}),
                summary=f"Justification analysis completed. Exit code: {result.get('exit_code', 'unknown')}"
            )

            self._status = WorkflowStatus.COMPLETED
            return workflow_result

        except Exception as e:
            logger.error(f"Justification workflow failed: {e}")
            self._status = WorkflowStatus.FAILED

            return WorkflowResult(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.FAILED,
                error=str(e),
                summary=f"Justification analysis failed: {e}"
            )

    def validate_dependencies(self) -> bool:
        """Validate that required dependencies are available."""
        try:
            # Check if we can create the engine
            engine = self._get_engine()
            return True
        except Exception as e:
            logger.warning(f"Justification workflow dependencies not satisfied: {e}")
            return False

    def get_required_inputs(self) -> Set[str]:
        """Get required workflow inputs."""
        return set()  # All inputs optional for justification

    def get_produced_outputs(self) -> Set[str]:
        """Get outputs this workflow produces."""
        return {
            'justification_reports',   # Generated justification reports
            'analysis_summary',        # Summary of analysis results
            'backup_files_detected',   # List of backup files found
            'code_quality_metrics',    # Quality metrics
        }

    # Backward compatibility methods - delegate to engine
    def justify_file(self, file_path: Path, rule_id: str = None):
        """Backward compatibility method."""
        return self._get_engine().justify_file(file_path, rule_id)

    def save_session_logs(self, output_dir: Path):
        """Backward compatibility method."""
        return self._get_engine().save_session_logs(output_dir)

    def run_full_analysis(self, target_directory: str = "."):
        """Backward compatibility method."""
        return self._get_engine().run_full_analysis(target_directory)