"""
Zero-Config Auto-Discovery for Multi-Project Vibelint

Automatically discovers:
- Project structure (single vs multi-project)
- Available services and their capabilities
- Shared resources (vector stores, LLM endpoints)
- Configuration inheritance without complex config files

Philosophy: Convention over configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import tomli


@dataclass
class ProjectInfo:
    """Discovered project information."""

    path: Path
    name: str
    type: str  # "library", "service", "orchestrator", "monorepo"
    python_package: Optional[str] = None
    has_pyproject: bool = False
    has_vibelint_config: bool = False
    dependencies: List[str] = None
    entry_points: Dict[str, str] = None


@dataclass
class ServiceCapabilities:
    """Auto-discovered service capabilities."""

    llm_endpoints: List[str] = None
    embedding_endpoints: List[str] = None
    vector_stores: List[str] = None
    validation_types: List[str] = None
    can_orchestrate: bool = False
    has_memory_system: bool = False


class AutoDiscovery:
    """
    Zero-configuration auto-discovery for vibelint multi-project scaling.

    Discovers project structure and capabilities without requiring configuration.
    Works for single projects, monorepos, and distributed microservices.
    """

    def __init__(self, starting_path: Path = None):
        self.starting_path = starting_path or Path.cwd()
        self.discovered_projects: Dict[str, ProjectInfo] = {}
        self.service_capabilities: Dict[str, ServiceCapabilities] = {}
        self.project_topology = {}

    def discover_project_structure(self) -> Dict[str, Any]:
        """
        Discover entire project structure starting from current directory.
        Returns complete topology for vibelint to operate on.
        """
        # Find project root(s)
        root_candidates = self._find_project_roots()

        # Determine if this is single project or multi-project setup
        if len(root_candidates) == 1:
            return self._discover_single_project(root_candidates[0])
        else:
            return self._discover_multi_project(root_candidates)

    def _find_project_roots(self) -> List[Path]:
        """Find all potential project roots."""
        roots = []
        current = self.starting_path

        # Walk up to find project markers
        for parent in [current] + list(current.parents):
            if self._is_project_root(parent):
                roots.append(parent)

            # Stop at git root or filesystem boundary
            if (parent / ".git").exists() or parent.parent == parent:
                break

        # If we found a git root, also search for sub-projects
        if roots:
            git_root = roots[-1] if (roots[-1] / ".git").exists() else roots[0]
            sub_projects = self._find_sub_projects(git_root)
            roots.extend(sub_projects)

        return list(set(roots))  # Deduplicate

    def _is_project_root(self, path: Path) -> bool:
        """Check if a path is a project root."""
        markers = ["pyproject.toml", "setup.py", "package.json", "Cargo.toml", ".git"]
        return any((path / marker).exists() for marker in markers)

    def _find_sub_projects(self, git_root: Path) -> List[Path]:
        """Find sub-projects within a git repository."""
        sub_projects = []

        # Common patterns for sub-projects
        search_patterns = [
            "tools/*/pyproject.toml",
            "services/*/pyproject.toml",
            "packages/*/pyproject.toml",
            "apps/*/pyproject.toml",
            "*/pyproject.toml",  # Direct subdirectories
        ]

        for pattern in search_patterns:
            for config_file in git_root.glob(pattern):
                project_dir = config_file.parent
                if project_dir != git_root:  # Don't include the root again
                    sub_projects.append(project_dir)

        return sub_projects

    def _discover_single_project(self, root: Path) -> Dict[str, Any]:
        """Discover single project structure."""
        project_info = self._analyze_project(root)
        self.discovered_projects[project_info.name] = project_info

        capabilities = self._discover_service_capabilities(project_info)
        self.service_capabilities[project_info.name] = capabilities

        return {
            "topology_type": "single_project",
            "root_project": project_info.name,
            "projects": {project_info.name: project_info},
            "capabilities": {project_info.name: capabilities},
            "routing": self._generate_single_project_routing(project_info, capabilities),
        }

    def _discover_multi_project(self, roots: List[Path]) -> Dict[str, Any]:
        """Discover multi-project structure."""
        projects = {}
        capabilities = {}

        for root in roots:
            project_info = self._analyze_project(root)
            projects[project_info.name] = project_info
            self.discovered_projects[project_info.name] = project_info

            project_capabilities = self._discover_service_capabilities(project_info)
            capabilities[project_info.name] = project_capabilities
            self.service_capabilities[project_info.name] = project_capabilities

        # Determine orchestrator and routing
        orchestrator = self._identify_orchestrator(projects, capabilities)
        routing = self._generate_multi_project_routing(projects, capabilities, orchestrator)

        return {
            "topology_type": "multi_project",
            "orchestrator": orchestrator,
            "projects": projects,
            "capabilities": capabilities,
            "routing": routing,
            "shared_resources": self._discover_shared_resources(projects),
        }

    def _analyze_project(self, path: Path) -> ProjectInfo:
        """Analyze a single project directory."""
        name = path.name

        # Check for Python package
        python_package = None
        if (path / "src").exists():
            # src layout
            src_dirs = [
                d for d in (path / "src").iterdir() if d.is_dir() and not d.name.startswith(".")
            ]
            if src_dirs:
                python_package = src_dirs[0].name
        elif any(
            (path / pkg).exists() and (path / pkg).is_dir()
            for pkg in [name, name.replace("-", "_")]
        ):
            # Direct package layout
            python_package = name.replace("-", "_")

        # Load pyproject.toml if exists
        pyproject_path = path / "pyproject.toml"
        dependencies = []
        entry_points = {}

        if pyproject_path.exists():
            try:
                with open(pyproject_path, "rb") as f:
                    pyproject = tomli.load(f)

                dependencies = pyproject.get("project", {}).get("dependencies", [])
                entry_points = pyproject.get("project", {}).get("entry-points", {})

                # Override name if specified in pyproject
                project_name = pyproject.get("project", {}).get("name")
                if project_name:
                    name = project_name

            except Exception:
                pass  # Continue with defaults

        # Determine project type
        project_type = self._infer_project_type(path, dependencies, entry_points)

        return ProjectInfo(
            path=path,
            name=name,
            type=project_type,
            python_package=python_package,
            has_pyproject=pyproject_path.exists(),
            has_vibelint_config=self._has_vibelint_config(path),
            dependencies=dependencies,
            entry_points=entry_points,
        )

    def _infer_project_type(
        self, path: Path, dependencies: List[str], entry_points: Dict[str, str]
    ) -> str:
        """Infer project type from structure and dependencies."""
        # Check for orchestrator patterns
        orchestrator_indicators = ["orchestrat", "guardrail", "coordinator", "manager", "control"]
        if any(indicator in path.name.lower() for indicator in orchestrator_indicators):
            return "orchestrator"

        # Check for service patterns
        service_indicators = ["service", "api", "worker", "daemon"]
        if any(indicator in path.name.lower() for indicator in service_indicators):
            return "service"

        # Check dependencies
        dep_strings = " ".join(dependencies).lower()
        if any(dep in dep_strings for dep in ["fastapi", "flask", "django"]):
            return "service"
        elif any(dep in dep_strings for dep in ["orchestrat", "celery", "prefect"]):
            return "orchestrator"

        # Check for tools directory pattern
        if "tools" in str(path):
            return "library"

        # Default based on structure
        if (path / "tests").exists() and (path / "src").exists():
            return "library"
        else:
            return "service"

    def _has_vibelint_config(self, path: Path) -> bool:
        """Check if project has vibelint configuration."""
        config_locations = [
            path / "pyproject.toml",
            path / "vibelint.toml",
            path / ".vibelint.toml",
        ]

        for config_path in config_locations:
            if config_path.exists():
                try:
                    with open(config_path, "rb") as f:
                        config = tomli.load(f)
                        if "tool" in config and "vibelint" in config["tool"]:
                            return True
                except Exception:
                    continue

        return False

    def _discover_service_capabilities(self, project: ProjectInfo) -> ServiceCapabilities:
        """Discover what capabilities a project/service has."""
        capabilities = ServiceCapabilities()

        # Check for LLM endpoint configuration
        if project.has_vibelint_config:
            config = self._load_project_config(project)
            llm_config = config.get("tool", {}).get("vibelint", {}).get("llm", {})

            endpoints = []
            if llm_config.get("fast_api_url"):
                endpoints.append("fast_llm")
            if llm_config.get("orchestrator_api_url"):
                endpoints.append("orchestrator_llm")

            capabilities.llm_endpoints = endpoints

            # Check embedding endpoints
            embedding_config = config.get("tool", {}).get("vibelint", {}).get("embeddings", {})
            embedding_endpoints = []
            if embedding_config.get("code_api_url"):
                embedding_endpoints.append("code_embeddings")
            if embedding_config.get("natural_api_url"):
                embedding_endpoints.append("natural_embeddings")

            capabilities.embedding_endpoints = embedding_endpoints

            # Check vector store
            vector_config = config.get("tool", {}).get("vibelint", {}).get("vector_store", {})
            if vector_config.get("backend"):
                capabilities.vector_stores = [vector_config["backend"]]

        # Check for orchestration capabilities
        if project.type == "orchestrator":
            capabilities.can_orchestrate = True

        # Check for memory system (look for specific imports/files)
        if project.python_package:
            memory_indicators = [
                "memory_system",
                "memory_manager",
                "conflict_resolver",
                "guardrails",
            ]

            for indicator in memory_indicators:
                potential_files = [
                    project.path / "src" / project.python_package / f"{indicator}.py",
                    project.path / project.python_package / f"{indicator}.py",
                ]

                if any(f.exists() for f in potential_files):
                    capabilities.has_memory_system = True
                    break

        # Determine validation types this project can handle
        validation_types = ["single_file"]  # All projects can do basic validation

        if project.type in ["orchestrator", "service"]:
            validation_types.extend(["project_wide", "architecture"])

        if capabilities.has_memory_system:
            validation_types.extend(["memory_conflicts", "security"])

        capabilities.validation_types = validation_types

        return capabilities

    def _load_project_config(self, project: ProjectInfo) -> Dict[str, Any]:
        """Load project configuration."""
        config_path = project.path / "pyproject.toml"
        if config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    return tomli.load(f)
            except Exception:
                pass
        return {}

    def _identify_orchestrator(
        self, projects: Dict[str, ProjectInfo], capabilities: Dict[str, ServiceCapabilities]
    ) -> Optional[str]:
        """Identify which project should act as orchestrator."""
        # Explicit orchestrator type
        for name, project in projects.items():
            if project.type == "orchestrator":
                return name

        # Project with orchestration capabilities
        for name, caps in capabilities.items():
            if caps.can_orchestrate:
                return name

        # Project with memory system (good orchestrator candidate)
        for name, caps in capabilities.items():
            if caps.has_memory_system:
                return name

        # Fallback: project with most capabilities
        if capabilities:
            best_project = max(capabilities.items(), key=lambda x: len(x[1].validation_types or []))
            return best_project[0]

        return None

    def _generate_single_project_routing(
        self, project: ProjectInfo, capabilities: ServiceCapabilities
    ) -> Dict[str, Any]:
        """Generate routing for single project setup."""
        return {
            "validation_routing": {
                vtype: [project.name]
                for vtype in (capabilities.validation_types or ["single_file"])
            },
            "llm_routing": {"primary": project.name if capabilities.llm_endpoints else None},
            "embedding_routing": {
                "primary": project.name if capabilities.embedding_endpoints else None
            },
        }

    def _generate_multi_project_routing(
        self,
        projects: Dict[str, ProjectInfo],
        capabilities: Dict[str, ServiceCapabilities],
        orchestrator: Optional[str],
    ) -> Dict[str, Any]:
        """Generate routing for multi-project setup."""
        routing = {"validation_routing": {}, "llm_routing": {}, "embedding_routing": {}}

        # Validation routing - distribute based on capabilities
        all_validation_types = set()
        for caps in capabilities.values():
            if caps.validation_types:
                all_validation_types.update(caps.validation_types)

        for vtype in all_validation_types:
            capable_services = [
                name
                for name, caps in capabilities.items()
                if caps.validation_types and vtype in caps.validation_types
            ]
            routing["validation_routing"][vtype] = capable_services

        # LLM routing - prefer orchestrator, fallback to any with LLM
        llm_services = [name for name, caps in capabilities.items() if caps.llm_endpoints]

        if orchestrator and orchestrator in llm_services:
            routing["llm_routing"]["primary"] = orchestrator
            routing["llm_routing"]["fallback"] = [s for s in llm_services if s != orchestrator]
        elif llm_services:
            routing["llm_routing"]["primary"] = llm_services[0]
            routing["llm_routing"]["fallback"] = llm_services[1:]

        # Embedding routing - distribute by type if possible
        embedding_services = [
            name for name, caps in capabilities.items() if caps.embedding_endpoints
        ]

        if embedding_services:
            routing["embedding_routing"]["primary"] = embedding_services[0]
            routing["embedding_routing"]["all"] = embedding_services

        return routing

    def _discover_shared_resources(self, projects: Dict[str, ProjectInfo]) -> Dict[str, Any]:
        """Discover shared resources across projects."""
        shared = {"vector_stores": [], "common_dependencies": [], "shared_configs": []}

        # Find common vector store configurations
        vector_stores = set()
        for project in projects.values():
            config = self._load_project_config(project)
            vector_config = config.get("tool", {}).get("vibelint", {}).get("vector_store", {})
            if vector_config.get("qdrant_collection"):
                vector_stores.add(vector_config["qdrant_collection"])

        shared["vector_stores"] = list(vector_stores)

        # Find common dependencies
        all_deps = []
        for project in projects.values():
            if project.dependencies:
                all_deps.extend(project.dependencies)

        # Count dependency frequency
        dep_counts = {}
        for dep in all_deps:
            dep_counts[dep] = dep_counts.get(dep, 0) + 1

        # Dependencies used by multiple projects
        common_deps = [dep for dep, count in dep_counts.items() if count > 1]
        shared["common_dependencies"] = common_deps

        return shared

    def get_vibelint_config(self) -> Dict[str, Any]:
        """
        Generate vibelint configuration based on auto-discovery.
        This replaces complex manual configuration.
        """
        structure = self.discover_project_structure()

        # Generate simple, unified config
        config = {
            "discovered_topology": structure["topology_type"],
            "auto_routing": structure["routing"],
            "services": {
                name: {
                    "path": str(info.path),
                    "type": info.type,
                    "capabilities": structure["capabilities"].get(name, {}),
                }
                for name, info in structure["projects"].items()
            },
        }

        # Add shared resources if multi-project
        if structure["topology_type"] == "multi_project":
            config["shared_resources"] = structure.get("shared_resources", {})
            config["orchestrator"] = structure.get("orchestrator")

        return config


# Convenience function for immediate discovery
def discover_and_configure(starting_path: Path = None) -> Dict[str, Any]:
    """
    One-function auto-discovery and configuration for vibelint.
    This is the main entry point for zero-config scaling.
    """
    discovery = AutoDiscovery(starting_path)
    return discovery.get_vibelint_config()


# Integration with existing vibelint
def integrate_with_vibelint_core():
    """
    Modify vibelint core to use auto-discovery by default.
    This makes scaling seamless without config changes.
    """
    # This would be integrated into vibelint's main config loading
    discovered_config = discover_and_configure()

    return {
        "auto_discovered": True,
        "manual_config_override": False,  # Set to True to use manual config instead
        "discovered_config": discovered_config,
    }
