"""
Distributed Vibelint Coordinator for Microservices Architecture

Coordinates validation tasks across multiple services:
- vibelint: Core validation engine
- kaia-guardrails: Orchestration and security
- Claude Code: Human-in-the-loop workflows

This enables validation at scale across your entire microservices ecosystem.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import tomli


@dataclass
class ServiceInfo:
    """Information about a distributed service."""

    name: str
    path: Path
    role: str  # "orchestrator", "validator", "coordinator"
    priority: int
    health_url: Optional[str] = None
    api_base: Optional[str] = None


@dataclass
class ValidationTask:
    """A validation task that can be distributed across services."""

    task_id: str
    task_type: str  # "single_file", "project_wide", "architecture", "security"
    target_files: List[str]
    requester_service: str
    priority: int
    assigned_services: List[str]
    status: str = "pending"  # "pending", "running", "completed", "failed"
    results: Dict[str, Any] = None
    created_at: float = None
    completed_at: Optional[float] = None


class DistributedVibelintCoordinator:
    """
    Coordinates vibelint validation across microservices architecture.

    Handles:
    - Service discovery and health monitoring
    - Task routing based on service capabilities
    - Result aggregation and conflict resolution
    - Shared state management via vector store
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_distributed_config()
        self.services: Dict[str, ServiceInfo] = {}
        self.active_tasks: Dict[str, ValidationTask] = {}
        self.task_counter = 0

        # Initialize services
        self._discover_services()

        # Setup shared resources
        self.vector_store_url = (
            self.config.get("shared_resources", {}).get("vector_store", {}).get("url")
        )
        self.coordination_collection = self.config.get("coordination", {}).get(
            "shared_cache_collection"
        )

    def _load_distributed_config(self) -> Dict[str, Any]:
        """Load distributed vibelint configuration."""
        try:
            with open(self.config_path, "rb") as f:
                return tomli.load(f)
        except Exception as e:
            print(f"Failed to load distributed config: {e}")
            return {}

    def _discover_services(self):
        """Discover available services based on configuration."""
        services_config = self.config.get("services", {})
        base_path = self.config_path.parent

        for service_name, service_config in services_config.items():
            service_path = base_path / service_config["path"]
            if service_path.exists():
                self.services[service_name] = ServiceInfo(
                    name=service_name,
                    path=service_path,
                    role=service_config["role"],
                    priority=service_config["priority"],
                )
                print(f"Discovered service: {service_name} at {service_path}")

    async def validate_project_distributed(self, target_files: List[str] = None) -> Dict[str, Any]:
        """
        Run distributed validation across all services.

        This is the main entry point for coordinated validation.
        """
        print("Starting distributed validation...")

        # Create validation tasks based on routing rules
        tasks = self._create_validation_tasks(target_files)

        # Execute tasks across services
        results = await self._execute_tasks_distributed(tasks)

        # Aggregate and resolve conflicts
        final_results = await self._aggregate_results(results)

        # Store results in shared state
        await self._store_validation_results(final_results)

        return final_results

    def _create_validation_tasks(self, target_files: List[str] = None) -> List[ValidationTask]:
        """Create validation tasks based on routing configuration."""
        tasks = []
        routing = self.config.get("validation_routing", {})

        if not target_files:
            # Discover all relevant files in the project
            target_files = self._discover_project_files()

        # Create tasks based on validation types
        for validation_type, assigned_services in routing.items():
            self.task_counter += 1
            task = ValidationTask(
                task_id=f"task_{self.task_counter:04d}",
                task_type=validation_type,
                target_files=target_files,
                requester_service="coordinator",
                priority=1,
                assigned_services=assigned_services,
                created_at=time.time(),
            )
            tasks.append(task)
            self.active_tasks[task.task_id] = task

        return tasks

    def _discover_project_files(self) -> List[str]:
        """Discover all Python files in the project."""
        project_root = self.config_path.parent
        python_files = []

        for service_name, service_info in self.services.items():
            service_files = list(service_info.path.rglob("*.py"))
            python_files.extend([str(f.relative_to(project_root)) for f in service_files])

        return python_files

    async def _execute_tasks_distributed(self, tasks: List[ValidationTask]) -> Dict[str, Any]:
        """Execute validation tasks across distributed services."""
        results = {}

        # Group tasks by assigned services to minimize coordination overhead
        service_tasks = self._group_tasks_by_service(tasks)

        # Execute tasks in parallel across services
        async_tasks = []
        for service_name, service_tasks_list in service_tasks.items():
            async_tasks.append(self._execute_service_tasks(service_name, service_tasks_list))

        # Wait for all services to complete
        service_results = await asyncio.gather(*async_tasks, return_exceptions=True)

        # Combine results from all services
        for i, (service_name, _) in enumerate(service_tasks.items()):
            service_result = service_results[i]
            if isinstance(service_result, Exception):
                print(f"Service {service_name} failed: {service_result}")
                results[service_name] = {"error": str(service_result)}
            else:
                results[service_name] = service_result

        return results

    def _group_tasks_by_service(
        self, tasks: List[ValidationTask]
    ) -> Dict[str, List[ValidationTask]]:
        """Group tasks by the services that need to execute them."""
        service_tasks = {}

        for task in tasks:
            for service_name in task.assigned_services:
                if service_name not in service_tasks:
                    service_tasks[service_name] = []
                service_tasks[service_name].append(task)

        return service_tasks

    async def _execute_service_tasks(
        self, service_name: str, tasks: List[ValidationTask]
    ) -> Dict[str, Any]:
        """Execute tasks for a specific service."""
        service_info = self.services.get(service_name)
        if not service_info:
            return {"error": f"Service {service_name} not found"}

        try:
            if service_name == "vibelint":
                return await self._execute_vibelint_tasks(service_info, tasks)
            elif service_name == "kaia-guardrails":
                return await self._execute_kaia_tasks(service_info, tasks)
            else:
                return {"error": f"Unknown service type: {service_name}"}

        except Exception as e:
            return {"error": f"Failed to execute tasks for {service_name}: {e}"}

    async def _execute_vibelint_tasks(
        self, service_info: ServiceInfo, tasks: List[ValidationTask]
    ) -> Dict[str, Any]:
        """Execute validation tasks using vibelint service."""
        # Import vibelint dynamically to avoid circular dependencies
        import sys

        sys.path.insert(0, str(service_info.path / "src"))

        try:
            from vibelint.config import load_config
            from vibelint.core import VibelintCore

            # Load service-specific configuration
            vibelint_config = load_config(service_info.path)
            core = VibelintCore(vibelint_config)

            results = {}
            for task in tasks:
                print(f"Executing {task.task_type} validation via vibelint...")

                # Execute validation based on task type
                if task.task_type in ["single_file", "project_wide", "architecture"]:
                    task_result = await core.validate_files(task.target_files)
                    results[task.task_id] = {
                        "task_type": task.task_type,
                        "files_validated": len(task.target_files),
                        "violations": task_result.get("violations", []),
                        "metrics": task_result.get("metrics", {}),
                        "service": "vibelint",
                    }
                    task.status = "completed"
                    task.completed_at = time.time()

            return results

        except ImportError as e:
            return {"error": f"Failed to import vibelint: {e}"}

    async def _execute_kaia_tasks(
        self, service_info: ServiceInfo, tasks: List[ValidationTask]
    ) -> Dict[str, Any]:
        """Execute tasks using kaia-guardrails service."""
        # Import kaia-guardrails dynamically
        import sys

        sys.path.insert(0, str(service_info.path / "src"))

        try:
            from kaia_guardrails.memory_system import memory_system
            from kaia_guardrails.orchestrator import KaiaOrchestrator

            orchestrator = KaiaOrchestrator()
            results = {}

            for task in tasks:
                print(f"Executing {task.task_type} analysis via kaia-guardrails...")

                if task.task_type == "architecture":
                    # Architecture analysis with memory integration
                    task_result = await orchestrator.analyze_architecture(task.target_files)
                    results[task.task_id] = {
                        "task_type": task.task_type,
                        "architecture_analysis": task_result,
                        "memory_conflicts": await self._check_memory_conflicts(task),
                        "service": "kaia-guardrails",
                    }

                elif task.task_type == "security":
                    # Security validation
                    task_result = await orchestrator.validate_security(task.target_files)
                    results[task.task_id] = {
                        "task_type": task.task_type,
                        "security_findings": task_result,
                        "service": "kaia-guardrails",
                    }

                task.status = "completed"
                task.completed_at = time.time()

            return results

        except ImportError as e:
            return {"error": f"Failed to import kaia-guardrails: {e}"}

    async def _check_memory_conflicts(self, task: ValidationTask) -> Dict[str, Any]:
        """Check for memory conflicts using kaia-guardrails EBR system."""
        try:
            from kaia_guardrails.memory_conflict_resolver import \
                MemoryConflictResolver

            resolver = MemoryConflictResolver()
            pattern_hash = f"validation_{task.task_type}_{hash(tuple(task.target_files))}"

            conflict = await resolver.detect_memory_conflicts(pattern_hash)
            if conflict:
                return {
                    "has_conflicts": True,
                    "conflict_details": asdict(conflict),
                    "requires_ebr": True,
                }
            else:
                return {"has_conflicts": False}

        except Exception as e:
            return {"error": f"Memory conflict check failed: {e}"}

    async def _aggregate_results(self, service_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate and resolve conflicts between service results."""
        aggregated = {
            "timestamp": datetime.now().isoformat(),
            "services_executed": list(service_results.keys()),
            "total_tasks": len(self.active_tasks),
            "service_results": service_results,
            "conflicts_detected": [],
            "resolution_strategy": "evidence_based_reasoning",
        }

        # Detect conflicts between service results
        conflicts = self._detect_result_conflicts(service_results)
        if conflicts:
            aggregated["conflicts_detected"] = conflicts
            # Use EBR to resolve conflicts if kaia-guardrails is available
            if "kaia-guardrails" in self.services:
                aggregated["conflict_resolution"] = await self._resolve_conflicts_via_ebr(conflicts)

        return aggregated

    def _detect_result_conflicts(self, service_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts between results from different services."""
        conflicts = []

        # Compare validation results for the same files across services
        file_results = {}

        for service_name, results in service_results.items():
            if isinstance(results, dict) and "error" not in results:
                for task_id, task_result in results.items():
                    task_type = task_result.get("task_type")
                    if task_type in file_results:
                        # Potential conflict - same task type from different services
                        conflicts.append(
                            {
                                "task_type": task_type,
                                "conflicting_services": [
                                    file_results[task_type]["service"],
                                    service_name,
                                ],
                                "task_ids": [file_results[task_type]["task_id"], task_id],
                            }
                        )
                    else:
                        file_results[task_type] = {
                            "service": service_name,
                            "task_id": task_id,
                            "result": task_result,
                        }

        return conflicts

    async def _resolve_conflicts_via_ebr(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts using Evidence-Based Reasoning."""
        try:
            from kaia_guardrails.memory_conflict_resolver import \
                MemoryConflictResolver

            resolver = MemoryConflictResolver()
            resolutions = {}

            for conflict in conflicts:
                # Create EBR context for this conflict
                pattern_hash = f"service_conflict_{hash(json.dumps(conflict, sort_keys=True))}"
                ebr_context = await resolver.prepare_ebr_context_for_llm(conflict)

                resolutions[pattern_hash] = {
                    "conflict": conflict,
                    "ebr_context": ebr_context,
                    "resolution": "trust_most_recent_evidence",  # Default strategy
                }

            return resolutions

        except Exception as e:
            return {"error": f"EBR conflict resolution failed: {e}"}

    async def _store_validation_results(self, results: Dict[str, Any]):
        """Store validation results in shared vector store for future reference."""
        if not self.vector_store_url or not self.coordination_collection:
            print("No vector store configured for result storage")
            return

        try:
            # Store results with timestamp for future EBR analysis
            storage_entry = {
                "type": "distributed_validation_result",
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "services_involved": results.get("services_executed", []),
                "project_hash": hash(str(self.config_path.parent)),
            }

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.vector_store_url}/collections/{self.coordination_collection}/points",
                    json={
                        "points": [
                            {
                                "id": f"validation_{int(time.time())}",
                                "payload": storage_entry,
                                "vector": [0.0] * 768,  # Placeholder vector
                            }
                        ]
                    },
                )

                if response.status_code == 200:
                    print("Validation results stored in shared vector store")
                else:
                    print(f"Failed to store results: {response.status_code}")

        except Exception as e:
            print(f"Failed to store validation results: {e}")

    async def health_check_services(self) -> Dict[str, bool]:
        """Check health of all distributed services."""
        health_status = {}

        for service_name, service_info in self.services.items():
            try:
                # Check if service directory exists and has required files
                config_file = service_info.path / "pyproject.toml"
                src_dir = service_info.path / "src"

                is_healthy = config_file.exists() and src_dir.exists()
                health_status[service_name] = is_healthy

                if not is_healthy:
                    print(f"Service {service_name} is unhealthy - missing required files")

            except Exception as e:
                health_status[service_name] = False
                print(f"Health check failed for {service_name}: {e}")

        return health_status


# Example usage for distributed coordination
async def run_distributed_validation():
    """Example of running distributed validation across microservices."""
    config_path = Path(__file__).parent.parent.parent.parent.parent / "vibelint-distributed.toml"

    coordinator = DistributedVibelintCoordinator(config_path)

    # Check service health
    health = await coordinator.health_check_services()
    print(f"Service health: {health}")

    # Run distributed validation
    results = await coordinator.validate_project_distributed()
    print(f"Distributed validation complete: {json.dumps(results, indent=2)}")

    return results


if __name__ == "__main__":
    asyncio.run(run_distributed_validation())
