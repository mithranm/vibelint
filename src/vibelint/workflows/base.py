"""
Base workflow system for extensible analysis tasks.

Provides framework for creating modular, composable workflows with
built-in evaluation, metrics collection, and plugin integration.

vibelint/src/vibelint/workflows/base.py
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union

logger = logging.getLogger(__name__)

__all__ = [
    "WorkflowStatus", "WorkflowPriority", "WorkflowConfig", "WorkflowMetrics",
    "WorkflowResult", "BaseWorkflow"
]


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowPriority(Enum):
    """Workflow execution priority."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""

    # Basic settings
    enabled: bool = True
    priority: WorkflowPriority = WorkflowPriority.MEDIUM
    timeout_seconds: Optional[int] = None

    # Dependencies and requirements
    required_tools: Set[str] = field(default_factory=set)
    required_data: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)  # Other workflow IDs

    # Execution settings
    parallel_execution: bool = False
    max_retries: int = 0
    cache_results: bool = True

    # Input/output settings
    input_filters: Dict[str, Any] = field(default_factory=dict)
    output_format: str = "json"

    # Custom parameters
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowMetrics:
    """Metrics collected during workflow execution."""

    # Timing metrics
    start_time: float
    end_time: Optional[float] = None
    execution_time_seconds: Optional[float] = None

    # Resource metrics
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    # Analysis metrics
    files_processed: int = 0
    findings_generated: int = 0
    errors_encountered: int = 0

    # Quality metrics
    confidence_score: float = 0.0
    accuracy_score: Optional[float] = None
    coverage_percentage: Optional[float] = None

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def finalize(self):
        """Finalize metrics calculation."""
        if self.end_time and self.start_time:
            self.execution_time_seconds = self.end_time - self.start_time


@dataclass
class WorkflowResult:
    """Result of workflow execution."""

    # Execution info
    workflow_id: str
    status: WorkflowStatus
    metrics: WorkflowMetrics

    # Results
    findings: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Metadata
    timestamp: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


class BaseWorkflow(ABC):
    """Abstract base class for vibelint workflows."""

    # Workflow identification
    workflow_id: str = ""
    name: str = ""
    description: str = ""
    version: str = "1.0"

    # Workflow categorization
    category: str = "analysis"  # analysis, validation, reporting, maintenance
    tags: Set[str] = set()

    def __init__(self, config: Optional[WorkflowConfig] = None):
        """Initialize workflow with configuration."""
        self.config = config or WorkflowConfig()
        self.metrics = WorkflowMetrics(start_time=time.time())
        self._status = WorkflowStatus.PENDING

        # Validate workflow setup
        self._validate_configuration()

    @abstractmethod
    async def execute(
        self,
        project_root: Path,
        context: Dict[str, Any]
    ) -> WorkflowResult:
        """Execute the workflow with given context.

        Args:
            project_root: Root directory of the project
            context: Execution context with shared data

        Returns:
            WorkflowResult with findings and artifacts
        """
        pass

    @abstractmethod
    def get_required_inputs(self) -> Set[str]:
        """Get set of required input data keys."""
        pass

    @abstractmethod
    def get_produced_outputs(self) -> Set[str]:
        """Get set of output data keys this workflow produces."""
        pass

    def can_execute(self, context: Dict[str, Any]) -> bool:
        """Check if workflow can execute with given context."""
        # Check if enabled
        if not self.config.enabled:
            return False

        # Check required inputs
        required_inputs = self.get_required_inputs()
        available_inputs = set(context.keys())

        if not required_inputs.issubset(available_inputs):
            missing = required_inputs - available_inputs
            logger.debug(f"Workflow {self.workflow_id} missing inputs: {missing}")
            return False

        # Check required tools
        if self.config.required_tools:
            # This would check for tool availability
            pass

        return True

    def estimate_execution_time(self, context: Dict[str, Any]) -> float:
        """Estimate execution time in seconds based on context."""
        # Default implementation - workflows can override
        base_time = 10.0  # 10 seconds base

        # Scale by number of files if available
        if "file_count" in context:
            file_count = context["file_count"]
            base_time += file_count * 0.1  # 0.1 seconds per file

        return base_time

    def get_dependencies(self) -> List[str]:
        """Get list of workflow IDs this workflow depends on."""
        return self.config.dependencies

    def get_priority(self) -> WorkflowPriority:
        """Get workflow execution priority."""
        return self.config.priority

    def supports_parallel_execution(self) -> bool:
        """Check if workflow supports parallel execution."""
        return self.config.parallel_execution

    def _validate_configuration(self):
        """Validate workflow configuration."""
        if not self.workflow_id:
            raise ValueError(f"Workflow {self.__class__.__name__} must define workflow_id")

        if not self.name:
            raise ValueError(f"Workflow {self.workflow_id} must define name")

    def _update_status(self, status: WorkflowStatus):
        """Update workflow execution status."""
        self._status = status

        if status == WorkflowStatus.COMPLETED:
            self.metrics.end_time = time.time()
            self.metrics.finalize()

    def _create_result(
        self,
        status: WorkflowStatus,
        findings: Optional[List[Dict[str, Any]]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> WorkflowResult:
        """Create workflow result."""
        self._update_status(status)

        return WorkflowResult(
            workflow_id=self.workflow_id,
            status=status,
            metrics=self.metrics,
            findings=findings or [],
            artifacts=artifacts or {},
            error_message=error_message
        )

    async def _execute_with_error_handling(
        self,
        project_root: Path,
        context: Dict[str, Any]
    ) -> WorkflowResult:
        """Execute workflow with comprehensive error handling."""
        try:
            self._update_status(WorkflowStatus.RUNNING)

            # Check timeout
            if self.config.timeout_seconds:
                # Implementation would use asyncio.wait_for
                pass

            # Execute the workflow
            result = await self.execute(project_root, context)

            # Validate result
            if result.status == WorkflowStatus.PENDING:
                result.status = WorkflowStatus.COMPLETED

            return result

        except Exception as e:
            logger.error(f"Workflow {self.workflow_id} failed: {e}", exc_info=True)
            self.metrics.errors_encountered += 1

            return self._create_result(
                WorkflowStatus.FAILED,
                error_message=str(e)
            )

    def get_evaluation_criteria(self) -> Dict[str, Any]:
        """Get criteria for evaluating workflow effectiveness."""
        return {
            "performance": {
                "max_execution_time": 60.0,  # seconds
                "max_memory_usage": 500.0,   # MB
            },
            "quality": {
                "min_confidence_score": 0.7,
                "min_coverage_percentage": 80.0,
            },
            "reliability": {
                "max_error_rate": 0.05,  # 5%
                "max_timeout_rate": 0.01,  # 1%
            }
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.workflow_id}, status={self._status.value})"