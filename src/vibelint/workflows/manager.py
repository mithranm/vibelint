"""
Workflow manager for orchestrating and executing analysis workflows.

Handles workflow scheduling, dependency resolution, parallel execution,
and results aggregation with comprehensive error handling.

vibelint/src/vibelint/workflow_manager.py
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .core.base import BaseWorkflow, WorkflowResult, WorkflowStatus
from .evaluation import WorkflowEvaluator
from .registry import workflow_registry

logger = logging.getLogger(__name__)

__all__ = ["WorkflowManager", "WorkflowExecutionPlan", "WorkflowSession"]


class WorkflowExecutionPlan:
    """Execution plan for workflow orchestration."""

    def __init__(self):
        self.workflows: List[BaseWorkflow] = []
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.execution_order: List[List[str]] = []  # Batches of workflows
        self.estimated_duration: float = 0.0

    def add_workflow(self, workflow: BaseWorkflow):
        """Add workflow to execution plan."""
        self.workflows.append(workflow)

        # Build dependency graph
        workflow_id = workflow.workflow_id
        dependencies = workflow.get_dependencies()

        for dep_id in dependencies:
            self.dependency_graph[workflow_id].add(dep_id)

    def resolve_execution_order(self):
        """Resolve workflow execution order using topological sort."""
        # Topological sort with batching for parallel execution
        in_degree = defaultdict(int)
        workflow_map = {w.workflow_id: w for w in self.workflows}

        # Calculate in-degrees
        for workflow_id in workflow_map:
            for dep_id in self.dependency_graph[workflow_id]:
                in_degree[workflow_id] += 1

        # Initialize queue with workflows that have no dependencies
        queue = deque()
        for workflow in self.workflows:
            if in_degree[workflow.workflow_id] == 0:
                queue.append(workflow.workflow_id)

        execution_batches = []

        while queue:
            # Process all workflows in current batch (can run in parallel)
            current_batch = []
            batch_size = len(queue)

            for _ in range(batch_size):
                workflow_id = queue.popleft()
                current_batch.append(workflow_id)

                # Update in-degrees for dependent workflows
                for dependent_id, deps in self.dependency_graph.items():
                    if workflow_id in deps:
                        in_degree[dependent_id] -= 1
                        if in_degree[dependent_id] == 0:
                            queue.append(dependent_id)

            if current_batch:
                execution_batches.append(current_batch)

        self.execution_order = execution_batches

        # Estimate total duration
        self.estimated_duration = sum(
            max(workflow_map[wid].estimate_execution_time({}) for wid in batch)
            for batch in execution_batches
        )

    def get_workflow_by_id(self, workflow_id: str) -> Optional[BaseWorkflow]:
        """Get workflow by ID."""
        for workflow in self.workflows:
            if workflow.workflow_id == workflow_id:
                return workflow
        return None


class WorkflowSession:
    """Session for tracking workflow execution state."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None

        # Execution state
        self.context: Dict[str, Any] = {}
        self.results: Dict[str, WorkflowResult] = {}
        self.errors: List[str] = []

        # Metrics
        self.total_workflows = 0
        self.completed_workflows = 0
        self.failed_workflows = 0

    def add_result(self, result: WorkflowResult):
        """Add workflow result to session."""
        self.results[result.workflow_id] = result

        if result.status == WorkflowStatus.COMPLETED:
            self.completed_workflows += 1
        elif result.status == WorkflowStatus.FAILED:
            self.failed_workflows += 1
            if result.error_message:
                self.errors.append(f"{result.workflow_id}: {result.error_message}")

    def update_context(self, key: str, value: Any):
        """Update shared context data."""
        self.context[key] = value

    def get_success_rate(self) -> float:
        """Get workflow success rate."""
        if self.total_workflows == 0:
            return 0.0
        return self.completed_workflows / self.total_workflows

    def finalize(self):
        """Finalize session."""
        self.end_time = time.time()

    def get_duration(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time


class WorkflowManager:
    """Manages workflow execution and orchestration."""

    def __init__(self, evaluator: Optional[WorkflowEvaluator] = None):
        self.evaluator = evaluator or WorkflowEvaluator()
        self.active_sessions: Dict[str, WorkflowSession] = {}

    async def execute_workflows(
        self,
        workflow_ids: List[str],
        project_root: Path,
        initial_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> WorkflowSession:
        """Execute specified workflows with dependency resolution."""

        if session_id is None:
            session_id = f"session_{int(time.time())}"

        # Create session
        session = WorkflowSession(session_id)
        session.context.update(initial_context or {})
        self.active_sessions[session_id] = session

        try:
            # Load workflows
            workflows = self._load_workflows(workflow_ids)
            session.total_workflows = len(workflows)

            # Create execution plan
            plan = self._create_execution_plan(workflows, session.context)

            # Execute workflows
            await self._execute_plan(plan, project_root, session)

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            session.errors.append(f"Execution failed: {e}")

        finally:
            session.finalize()

        return session

    def _load_workflows(self, workflow_ids: List[str]) -> List[BaseWorkflow]:
        """Load workflow instances from registry."""
        workflows = []

        for workflow_id in workflow_ids:
            workflow_class = workflow_registry.get_workflow(workflow_id)
            if workflow_class:
                workflow = workflow_class()
                workflows.append(workflow)
            else:
                logger.warning(f"Workflow not found: {workflow_id}")

        return workflows

    def _create_execution_plan(
        self, workflows: List[BaseWorkflow], context: Dict[str, Any]
    ) -> WorkflowExecutionPlan:
        """Create optimized execution plan."""

        plan = WorkflowExecutionPlan()

        # Filter workflows that can execute
        executable_workflows = []
        for workflow in workflows:
            if workflow.can_execute(context):
                executable_workflows.append(workflow)
            else:
                logger.info(f"Skipping workflow {workflow.workflow_id}: requirements not met")

        # Add workflows to plan
        for workflow in executable_workflows:
            plan.add_workflow(workflow)

        # Resolve execution order
        plan.resolve_execution_order()

        logger.info(
            f"Created execution plan: {len(executable_workflows)} workflows in "
            f"{len(plan.execution_order)} batches, estimated duration: {plan.estimated_duration:.1f}s"
        )

        return plan

    async def _execute_plan(
        self, plan: WorkflowExecutionPlan, project_root: Path, session: WorkflowSession
    ):
        """Execute workflow plan with parallel batching."""

        for batch_idx, workflow_ids in enumerate(plan.execution_order):
            logger.info(
                f"Executing batch {batch_idx + 1}/{len(plan.execution_order)}: {workflow_ids}"
            )

            # Group workflows by parallel execution capability
            parallel_workflows = []
            sequential_workflows = []

            for workflow_id in workflow_ids:
                workflow = plan.get_workflow_by_id(workflow_id)
                if workflow and workflow.supports_parallel_execution():
                    parallel_workflows.append(workflow)
                else:
                    sequential_workflows.append(workflow)

            # Execute parallel workflows concurrently
            if parallel_workflows:
                tasks = []
                for workflow in parallel_workflows:
                    task = asyncio.create_task(
                        self._execute_single_workflow(workflow, project_root, session)
                    )
                    tasks.append(task)

                parallel_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in parallel_results:
                    if isinstance(result, Exception):
                        logger.error(f"Parallel workflow failed: {result}")
                    elif isinstance(result, WorkflowResult):
                        session.add_result(result)
                        self._update_context_from_result(session, result)

            # Execute sequential workflows
            for workflow in sequential_workflows:
                result = await self._execute_single_workflow(workflow, project_root, session)
                session.add_result(result)
                self._update_context_from_result(session, result)

    async def _execute_single_workflow(
        self, workflow: BaseWorkflow, project_root: Path, session: WorkflowSession
    ) -> WorkflowResult:
        """Execute single workflow with monitoring."""

        logger.debug(f"Starting workflow: {workflow.workflow_id}")

        try:
            # Execute workflow
            result = await workflow._execute_with_error_handling(project_root, session.context)

            # Evaluate workflow performance
            if self.evaluator:
                evaluation = self.evaluator.evaluate_workflow_execution(workflow, result)
                result.artifacts["evaluation"] = evaluation

            logger.info(
                f"Workflow {workflow.workflow_id} completed: "
                f"status={result.status.value}, "
                f"findings={len(result.findings)}, "
                f"duration={result.metrics.execution_time_seconds:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Workflow {workflow.workflow_id} execution failed: {e}", exc_info=True)

            # Create failed result
            return WorkflowResult(
                workflow_id=workflow.workflow_id,
                status=WorkflowStatus.FAILED,
                metrics=workflow.metrics,
                error_message=str(e),
            )

    def _update_context_from_result(self, session: WorkflowSession, result: WorkflowResult):
        """Update session context with workflow results."""

        # Add result artifacts to context
        for key, value in result.artifacts.items():
            context_key = f"{result.workflow_id}_{key}"
            session.update_context(context_key, value)

        # Add findings summary
        if result.findings:
            findings_key = f"{result.workflow_id}_findings"
            session.update_context(findings_key, result.findings)

        # Add metrics
        metrics_key = f"{result.workflow_id}_metrics"
        session.update_context(metrics_key, result.metrics)

    def get_available_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available workflows."""
        available = {}

        for workflow_id, workflow_class in workflow_registry.get_all_workflows().items():
            # Create temporary instance to get metadata
            try:
                temp_workflow = workflow_class()
                available[workflow_id] = {
                    "name": temp_workflow.name,
                    "description": temp_workflow.description,
                    "category": temp_workflow.category,
                    "version": temp_workflow.version,
                    "tags": list(temp_workflow.tags),
                    "required_inputs": list(temp_workflow.get_required_inputs()),
                    "produced_outputs": list(temp_workflow.get_produced_outputs()),
                    "dependencies": temp_workflow.get_dependencies(),
                    "supports_parallel": temp_workflow.supports_parallel_execution(),
                }
            except Exception as e:
                logger.warning(f"Failed to get metadata for workflow {workflow_id}: {e}")

        return available

    def validate_workflow_plan(self, workflow_ids: List[str]) -> Dict[str, Any]:
        """Validate workflow execution plan."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "execution_order": [],
            "estimated_duration": 0.0,
        }

        try:
            # Load workflows
            workflows = self._load_workflows(workflow_ids)

            if len(workflows) != len(workflow_ids):
                missing = set(workflow_ids) - {w.workflow_id for w in workflows}
                validation_result["errors"].append(f"Missing workflows: {missing}")
                validation_result["valid"] = False

            # Create execution plan
            plan = self._create_execution_plan(workflows, {})
            validation_result["execution_order"] = plan.execution_order
            validation_result["estimated_duration"] = plan.estimated_duration

            # Check for circular dependencies
            if self._has_circular_dependencies(plan):
                validation_result["errors"].append("Circular dependencies detected")
                validation_result["valid"] = False

        except Exception as e:
            validation_result["errors"].append(f"Validation failed: {e}")
            validation_result["valid"] = False

        return validation_result

    def _has_circular_dependencies(self, plan: WorkflowExecutionPlan) -> bool:
        """Check for circular dependencies in workflow plan."""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(workflow_id: str) -> bool:
            visited.add(workflow_id)
            rec_stack.add(workflow_id)

            for dep_id in plan.dependency_graph[workflow_id]:
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True

            rec_stack.remove(workflow_id)
            return False

        for workflow in plan.workflows:
            if workflow.workflow_id not in visited:
                if has_cycle(workflow.workflow_id):
                    return True

        return False

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of workflow session."""
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session_id,
            "duration": session.get_duration(),
            "total_workflows": session.total_workflows,
            "completed_workflows": session.completed_workflows,
            "failed_workflows": session.failed_workflows,
            "success_rate": session.get_success_rate(),
            "errors": session.errors,
            "context_keys": list(session.context.keys()),
        }
