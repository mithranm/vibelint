"""
LLM Context Engineering for Vibelint

The core innovation: Transform raw codebase into rich, structured context that enables
foundation models (GPT-OSS 120B, Claude) to make intelligent code improvements beyond
simple linting.

This is vibelint's secret sauce - sophisticated context engineering that gives LLMs
the deep codebase understanding they need for complex architectural decisions.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from .dependency_graph_manager import DependencyGraphManager, GraphQuery
from .multi_representation_analyzer import (ImprovementOpportunity,
                                            MultiRepresentationAnalyzer)


@dataclass
class LLMContext:
    """Rich context package for foundation model consumption."""

    # Core codebase understanding
    project_overview: Dict[str, Any]
    current_focus: Dict[str, Any]  # What the LLM should focus on

    # Multi-representation context
    filesystem_context: Dict[str, Any]
    semantic_context: Dict[str, Any]
    dependency_context: Dict[str, Any]
    execution_context: Dict[str, Any]

    # Problem-specific context
    improvement_opportunity: Optional[ImprovementOpportunity]
    similar_patterns: List[Dict[str, Any]]
    impact_analysis: Dict[str, Any]

    # Decision support
    suggested_approach: str
    risk_assessment: Dict[str, Any]
    implementation_guidance: List[str]

    # Meta-context
    confidence_indicators: Dict[str, float]
    context_completeness: float
    llm_instructions: str


@dataclass
class LLMRequest:
    """Request to foundation model with engineered context."""

    task_type: str  # "improve_code", "fix_architecture", "optimize_performance"
    context: LLMContext
    target_files: List[str]
    constraints: List[str]
    success_criteria: List[str]

    # LLM routing
    preferred_model: str  # "chip", "claudia", "claude_cli"
    reasoning_depth: str  # "shallow", "medium", "deep"
    creativity_level: float  # 0.0 = conservative, 1.0 = creative


@dataclass
class LLMResponse:
    """Structured response from foundation model."""

    reasoning_trace: str
    proposed_changes: List[Dict[str, Any]]
    confidence_score: float
    alternative_approaches: List[str]
    risk_mitigation: List[str]

    # Implementation details
    code_changes: List[Dict[str, str]]  # file_path -> new_content
    test_recommendations: List[str]
    documentation_updates: List[str]

    # Meta-response
    model_used: str
    processing_time_ms: int
    token_usage: Dict[str, int]


class VibelintContextEngineer:
    """
    The heart of vibelint: Engineers rich context from multi-representation analysis
    to enable foundation models to make sophisticated code improvements.
    """

    def __init__(
        self,
        project_root: Path,
        multi_rep_analyzer: MultiRepresentationAnalyzer,
        dependency_manager: DependencyGraphManager,
    ):
        self.project_root = project_root
        self.analyzer = multi_rep_analyzer
        self.dependency_manager = dependency_manager

        # Context engineering templates
        self.context_templates = self._load_context_templates()

        # LLM model configurations
        self.model_configs = {
            "chip": {
                "url": "https://chipllm-auth-worker.mithran-mohanraj.workers.dev",
                "strengths": ["deep_reasoning", "complex_architecture", "performance_optimization"],
                "max_context": 28800,
                "ideal_for": [
                    "architectural_changes",
                    "performance_optimization",
                    "complex_refactoring",
                ],
            },
            "claudia": {
                "url": "https://claudiallm-auth-worker.mithran-mohanraj.workers.dev",
                "strengths": ["fast_iteration", "simple_fixes", "code_generation"],
                "max_context": 900,
                "ideal_for": ["simple_fixes", "code_generation", "quick_improvements"],
            },
            "claude_cli": {
                "strengths": ["general_purpose", "explanation", "complex_reasoning"],
                "max_context": 200000,
                "ideal_for": [
                    "complex_analysis",
                    "architectural_decisions",
                    "comprehensive_refactoring",
                ],
            },
        }

    async def engineer_context_for_improvement(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> LLMContext:
        """
        Engineer rich context for a specific improvement opportunity.
        This is where the magic happens - transforming raw analysis into perfect LLM context.
        """
        print(f"🧠 Engineering context for: {opportunity.opportunity_id}")

        # Build project overview
        project_overview = await self._build_project_overview(analysis_results)

        # Focus context on the specific improvement
        current_focus = await self._build_focus_context(opportunity, analysis_results)

        # Extract relevant multi-representation context
        filesystem_context = await self._extract_filesystem_context(opportunity, analysis_results)
        semantic_context = await self._extract_semantic_context(opportunity, analysis_results)
        dependency_context = await self._extract_dependency_context(opportunity, analysis_results)
        execution_context = await self._extract_execution_context(opportunity, analysis_results)

        # Find similar patterns and examples
        similar_patterns = await self._find_similar_patterns(opportunity)

        # Analyze impact and risks
        impact_analysis = await self._analyze_impact(opportunity, analysis_results)

        # Generate approach and guidance
        suggested_approach = await self._suggest_approach(opportunity, analysis_results)
        risk_assessment = await self._assess_risks(opportunity, analysis_results)
        implementation_guidance = await self._generate_guidance(opportunity, analysis_results)

        # Calculate confidence and completeness
        confidence_indicators = await self._calculate_confidence(opportunity, analysis_results)
        context_completeness = await self._assess_completeness(opportunity, analysis_results)

        # Generate LLM instructions
        llm_instructions = await self._generate_llm_instructions(opportunity, analysis_results)

        return LLMContext(
            project_overview=project_overview,
            current_focus=current_focus,
            filesystem_context=filesystem_context,
            semantic_context=semantic_context,
            dependency_context=dependency_context,
            execution_context=execution_context,
            improvement_opportunity=opportunity,
            similar_patterns=similar_patterns,
            impact_analysis=impact_analysis,
            suggested_approach=suggested_approach,
            risk_assessment=risk_assessment,
            implementation_guidance=implementation_guidance,
            confidence_indicators=confidence_indicators,
            context_completeness=context_completeness,
            llm_instructions=llm_instructions,
        )

    async def _build_project_overview(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Build high-level project context."""
        return {
            "project_name": self.project_root.name,
            "project_type": "Python linting and code analysis tool",
            "key_characteristics": {
                "total_files": len(list(self.project_root.rglob("*.py"))),
                "architecture_style": "modular_with_plugins",
                "main_responsibilities": [
                    "Code quality analysis",
                    "Validation and linting",
                    "LLM-powered analysis",
                    "Self-improvement capabilities",
                ],
                "quality_score": analysis_results.get("summary", {}).get(
                    "overall_quality_score", 0.5
                ),
            },
            "current_challenges": [
                op.category
                for op in analysis_results.get("improvement_opportunities", [])
                if op.severity == "high"
            ],
            "architectural_patterns": analysis_results.get("vector_analysis", {}).get(
                "architectural_patterns", {}
            ),
            "dependency_complexity": {
                "total_nodes": analysis_results.get("graph_analysis", {})
                .get("dependency_metrics", {})
                .get("total_nodes", 0),
                "circular_dependencies": len(
                    analysis_results.get("graph_analysis", {})
                    .get("dependency_metrics", {})
                    .get("circular_dependencies", [])
                ),
                "max_depth": analysis_results.get("graph_analysis", {})
                .get("dependency_metrics", {})
                .get("longest_path", []),
            },
        }

    async def _build_focus_context(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build focused context for the specific improvement."""
        return {
            "improvement_target": {
                "category": opportunity.category,
                "severity": opportunity.severity,
                "file_path": opportunity.file_path,
                "line_number": opportunity.line_number,
                "affected_modules": opportunity.affected_modules,
            },
            "current_problem": {
                "description": opportunity.current_state,
                "impact_on_codebase": opportunity.estimated_impact,
                "execution_criticality": opportunity.execution_criticality,
            },
            "desired_outcome": {
                "target_state": opportunity.target_state,
                "success_metrics": opportunity.estimated_impact,
                "implementation_steps": opportunity.implementation_steps,
            },
            "scope_of_change": {
                "direct_files": opportunity.affected_modules,
                "dependent_files": opportunity.dependency_impact,
                "semantic_cluster": opportunity.semantic_clusters,
            },
        }

    async def _extract_filesystem_context(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract relevant filesystem organization context."""
        filesystem_analysis = analysis_results.get("filesystem_analysis", {})

        relevant_files = set(opportunity.affected_modules + opportunity.dependency_impact)

        # Extract module hierarchy for relevant files
        relevant_hierarchy = {}
        for file_path in relevant_files:
            # Find this file in the hierarchy
            hierarchy = filesystem_analysis.get("module_hierarchy", {})
            # Simplified - would traverse hierarchy to find relevant context
            relevant_hierarchy[file_path] = {
                "package": str(Path(file_path).parent),
                "module_purpose": self._infer_module_purpose(file_path),
                "related_files": self._find_related_files(file_path, hierarchy),
            }

        return {
            "relevant_modules": relevant_hierarchy,
            "package_organization": filesystem_analysis.get("package_organization", {}),
            "naming_patterns": filesystem_analysis.get("naming_conventions", {}),
            "architectural_layout": self._describe_architectural_layout(
                opportunity, filesystem_analysis
            ),
        }

    async def _extract_semantic_context(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract semantic understanding context."""
        vector_analysis = analysis_results.get("vector_analysis", {})

        return {
            "semantic_clusters": {
                cluster: elements
                for cluster, elements in vector_analysis.get("semantic_clusters", {}).items()
                if any(elem in opportunity.similar_code_patterns for elem in elements)
            },
            "code_duplication": [
                dup
                for dup in vector_analysis.get("code_duplication", [])
                if any(mod in [dup["file1"], dup["file2"]] for mod in opportunity.affected_modules)
            ],
            "naming_consistency": vector_analysis.get("naming_consistency", {}),
            "functional_cohesion": vector_analysis.get("functional_cohesion", {}),
            "semantic_purpose": await self._infer_semantic_purpose(opportunity, vector_analysis),
        }

    async def _extract_dependency_context(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract dependency graph context."""
        graph_analysis = analysis_results.get("graph_analysis", {})

        # Get dependency subgraph around the improvement target
        subgraph_context = await self._extract_dependency_subgraph(opportunity)

        return {
            "local_dependencies": subgraph_context,
            "impact_radius": await self._calculate_impact_radius(opportunity),
            "architectural_violations": [
                violation
                for violation in graph_analysis.get("architectural_violations", [])
                if any(
                    mod in violation.get("affected_modules", [])
                    for mod in opportunity.affected_modules
                )
            ],
            "critical_paths": await self._find_critical_paths_through_target(opportunity),
            "coupling_metrics": await self._get_coupling_metrics(opportunity),
        }

    async def _extract_execution_context(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract runtime execution context."""
        # This would use trace session data to understand how code actually executes
        return {
            "execution_frequency": opportunity.execution_criticality,
            "performance_profile": {
                "avg_execution_time": "unknown",  # Would come from trace data
                "memory_usage": "unknown",
                "io_operations": "unknown",
            },
            "call_patterns": [],  # Would come from trace data
            "error_patterns": [],  # Would come from trace data
            "usage_scenarios": await self._infer_usage_scenarios(opportunity),
        }

    async def _find_similar_patterns(
        self, opportunity: ImprovementOpportunity
    ) -> List[Dict[str, Any]]:
        """Find similar patterns in the codebase using embeddings."""
        if not opportunity.similar_code_patterns:
            return []

        # Query for similar patterns using semantic search
        query = GraphQuery(
            semantic_query=f"{opportunity.category} {opportunity.current_state}", max_results=5
        )

        similar_nodes = await self.dependency_manager.query_similar_dependencies(query)

        return [
            {
                "pattern_type": "semantic_similarity",
                "node_info": node["node_info"],
                "similarity_score": node["similarity_score"],
                "why_relevant": f"Similar {opportunity.category} pattern",
            }
            for node in similar_nodes
        ]

    async def _analyze_impact(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the impact of making this improvement."""
        return {
            "direct_benefits": opportunity.estimated_impact,
            "indirect_benefits": await self._calculate_indirect_benefits(
                opportunity, analysis_results
            ),
            "affected_stakeholders": await self._identify_stakeholders(opportunity),
            "cascade_effects": await self._predict_cascade_effects(opportunity, analysis_results),
            "metrics_improvement": await self._predict_metrics_improvement(
                opportunity, analysis_results
            ),
        }

    async def _suggest_approach(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> str:
        """Suggest the best approach for this improvement."""
        approach_templates = {
            "solid": "Apply SOLID principles systematically: {steps}",
            "complexity": "Reduce complexity through decomposition: {steps}",
            "pythonic": "Improve Pythonic practices: {steps}",
            "architecture": "Refactor architecture: {steps}",
        }

        template = approach_templates.get(opportunity.category, "General improvement: {steps}")
        steps = " -> ".join(opportunity.implementation_steps[:3])  # First 3 steps

        return template.format(steps=steps)

    async def _assess_risks(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks of implementing this improvement."""
        return {
            "implementation_risk": opportunity.risk_assessment,
            "breaking_changes_risk": await self._assess_breaking_changes(opportunity),
            "performance_impact_risk": await self._assess_performance_impact(opportunity),
            "maintenance_burden_risk": await self._assess_maintenance_burden(opportunity),
            "mitigation_strategies": await self._suggest_risk_mitigation(opportunity),
        }

    async def _generate_guidance(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> List[str]:
        """Generate implementation guidance."""
        return [
            f"Start with {opportunity.implementation_steps[0] if opportunity.implementation_steps else 'analysis'}",
            f"Focus on {opportunity.affected_modules[0] if opportunity.affected_modules else 'target file'} first",
            "Test changes incrementally",
            "Monitor impact on related modules",
            "Update documentation as needed",
        ]

    async def _calculate_confidence(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate confidence indicators."""
        return {
            "problem_identification": 0.9,  # High confidence in multi-rep analysis
            "solution_appropriateness": 0.8,  # Good confidence in suggested approach
            "impact_prediction": 0.7,  # Moderate confidence in impact analysis
            "risk_assessment": 0.6,  # Lower confidence without full testing
            "implementation_feasibility": 0.8,  # Good confidence based on patterns
        }

    async def _assess_completeness(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> float:
        """Assess completeness of context."""
        completeness_factors = [
            1.0 if opportunity.affected_modules else 0.0,  # Have target files
            1.0 if opportunity.implementation_steps else 0.0,  # Have implementation plan
            1.0 if opportunity.estimated_impact else 0.0,  # Have impact estimate
            1.0 if analysis_results.get("filesystem_analysis") else 0.0,  # Have filesystem context
            1.0 if analysis_results.get("vector_analysis") else 0.0,  # Have semantic context
            1.0 if analysis_results.get("graph_analysis") else 0.0,  # Have graph context
        ]

        return sum(completeness_factors) / len(completeness_factors)

    async def _generate_llm_instructions(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> str:
        """Generate specific instructions for the LLM."""
        instruction_template = self.context_templates.get(
            opportunity.category, self.context_templates["default"]
        )

        return instruction_template.format(
            category=opportunity.category,
            severity=opportunity.severity,
            current_state=opportunity.current_state,
            target_state=opportunity.target_state,
            file_path=opportunity.file_path,
            affected_modules=", ".join(opportunity.affected_modules[:3]),
        )

    def _load_context_templates(self) -> Dict[str, str]:
        """Load context templates for different improvement categories."""
        return {
            "solid": """
You are improving code quality by applying SOLID principles to {category} issues.

CURRENT PROBLEM: {current_state}
TARGET OUTCOME: {target_state}
SEVERITY: {severity}

FOCUS: Apply systematic refactoring to {file_path} and related modules: {affected_modules}

APPROACH:
1. Identify specific SOLID principle violations
2. Design minimal changes that address the root cause
3. Ensure changes improve testability and maintainability
4. Preserve existing functionality

OUTPUT: Provide specific code changes with detailed reasoning for each SOLID principle applied.
            """,
            "complexity": """
You are reducing code complexity to improve maintainability and readability.

CURRENT PROBLEM: {current_state}
TARGET OUTCOME: {target_state}
SEVERITY: {severity}

FOCUS: Simplify complex code in {file_path} affecting: {affected_modules}

APPROACH:
1. Identify sources of complexity (nested conditions, long functions, etc.)
2. Extract smaller, focused functions
3. Use early returns and guard clauses
4. Consider design patterns if appropriate

OUTPUT: Provide refactored code with complexity analysis showing improvement.
            """,
            "pythonic": """
You are improving Python code to follow best practices and idioms.

CURRENT PROBLEM: {current_state}
TARGET OUTCOME: {target_state}
SEVERITY: {severity}

FOCUS: Make code more Pythonic in {file_path} and related: {affected_modules}

APPROACH:
1. Apply Python idioms and best practices
2. Improve type hints and documentation
3. Use appropriate data structures and libraries
4. Follow PEP conventions

OUTPUT: Provide improved code following Python best practices with explanations.
            """,
            "architecture": """
You are improving software architecture to reduce coupling and improve modularity.

CURRENT PROBLEM: {current_state}
TARGET OUTCOME: {target_state}
SEVERITY: {severity}

FOCUS: Refactor architecture in {file_path} with impact on: {affected_modules}

APPROACH:
1. Identify architectural anti-patterns
2. Design cleaner interfaces and abstractions
3. Reduce dependencies and coupling
4. Improve module boundaries

OUTPUT: Provide architectural improvements with migration strategy.
            """,
            "default": """
You are improving code quality in the {category} category.

PROBLEM: {current_state}
GOAL: {target_state}
SEVERITY: {severity}

Focus on {file_path} and related modules: {affected_modules}

Provide specific improvements with clear reasoning and implementation steps.
            """,
        }

    # Helper methods for context extraction
    def _infer_module_purpose(self, file_path: str) -> str:
        """Infer the purpose of a module from its path and name."""
        path_parts = Path(file_path).parts
        filename = Path(file_path).stem

        purpose_indicators = {
            "config": "configuration management",
            "core": "core functionality",
            "utils": "utility functions",
            "validators": "validation logic",
            "cli": "command line interface",
            "api": "API interface",
            "models": "data models",
            "tests": "testing",
        }

        for indicator, purpose in purpose_indicators.items():
            if indicator in filename.lower() or any(
                indicator in part.lower() for part in path_parts
            ):
                return purpose

        return "general purpose module"

    def _find_related_files(self, file_path: str, hierarchy: Dict[str, Any]) -> List[str]:
        """Find files related to the target file."""
        # Simplified - would analyze imports, similar names, same directory
        file_dir = str(Path(file_path).parent)
        related = []

        # Files in same directory are likely related
        for path, data in hierarchy.items():
            if isinstance(data, dict) and data.get("type") == "file":
                if str(Path(data.get("path", "")).parent) == file_dir:
                    related.append(data.get("path", ""))

        return related[:5]  # Limit to 5 most related

    def _describe_architectural_layout(
        self, opportunity: ImprovementOpportunity, filesystem_analysis: Dict[str, Any]
    ) -> str:
        """Describe the architectural layout relevant to the opportunity."""
        if opportunity.category == "architecture":
            return "Complex architectural refactoring required with careful attention to module boundaries"
        elif opportunity.category == "solid":
            return "Focus on single responsibility and dependency inversion principles"
        else:
            return "Standard modular layout with clear separation of concerns"

    async def _infer_semantic_purpose(
        self, opportunity: ImprovementOpportunity, vector_analysis: Dict[str, Any]
    ) -> str:
        """Infer the semantic purpose of the improvement target."""
        # Use semantic clusters and naming patterns to understand purpose
        semantic_clusters = vector_analysis.get("semantic_clusters", {})

        for cluster_name, elements in semantic_clusters.items():
            if any(elem in opportunity.similar_code_patterns for elem in elements):
                return f"Part of {cluster_name} functionality cluster"

        return "Standalone functionality"

    async def _extract_dependency_subgraph(
        self, opportunity: ImprovementOpportunity
    ) -> Dict[str, Any]:
        """Extract dependency subgraph around the improvement target."""
        graph = self.dependency_manager.dependency_graph

        target_nodes = []
        for module in opportunity.affected_modules:
            # Find nodes in graph that match this module
            matching_nodes = [node for node in graph.nodes() if module in node]
            target_nodes.extend(matching_nodes)

        if not target_nodes:
            return {}

        # Get ego graph (node + immediate neighbors)
        subgraph_nodes = set()
        for node in target_nodes:
            if node in graph:
                subgraph_nodes.add(node)
                subgraph_nodes.update(graph.successors(node))
                subgraph_nodes.update(graph.predecessors(node))

        subgraph = graph.subgraph(subgraph_nodes)

        return {
            "nodes": list(subgraph.nodes()),
            "edges": list(subgraph.edges()),
            "density": nx.density(subgraph),
            "components": len(list(nx.weakly_connected_components(subgraph))),
        }

    async def _calculate_impact_radius(self, opportunity: ImprovementOpportunity) -> Dict[str, int]:
        """Calculate how far the impact of this change might reach."""
        graph = self.dependency_manager.dependency_graph

        impact_radius = {"direct": 0, "indirect": 0, "distant": 0}

        for module in opportunity.affected_modules:
            matching_nodes = [node for node in graph.nodes() if module in node]

            for node in matching_nodes:
                if node in graph:
                    # Direct impact: immediate dependencies
                    direct = len(list(graph.successors(node))) + len(list(graph.predecessors(node)))
                    impact_radius["direct"] += direct

                    # Indirect impact: 2-hop dependencies
                    indirect = 0
                    for neighbor in graph.neighbors(node):
                        indirect += len(list(graph.neighbors(neighbor)))
                    impact_radius["indirect"] += indirect

        return impact_radius

    async def _find_critical_paths_through_target(
        self, opportunity: ImprovementOpportunity
    ) -> List[str]:
        """Find critical execution paths that go through the improvement target."""
        # This would use execution trace data to find critical paths
        # For now, simplified version
        return [f"Critical path through {module}" for module in opportunity.affected_modules[:3]]

    async def _get_coupling_metrics(self, opportunity: ImprovementOpportunity) -> Dict[str, float]:
        """Get coupling metrics for the improvement target."""
        graph = self.dependency_manager.dependency_graph

        coupling_metrics = {}

        for module in opportunity.affected_modules:
            matching_nodes = [node for node in graph.nodes() if module in node]

            for node in matching_nodes:
                if node in graph:
                    in_degree = graph.in_degree(node)
                    out_degree = graph.out_degree(node)
                    total = in_degree + out_degree

                    coupling_metrics[node] = {
                        "afferent_coupling": in_degree,
                        "efferent_coupling": out_degree,
                        "instability": out_degree / total if total > 0 else 0,
                    }

        return coupling_metrics

    async def _infer_usage_scenarios(self, opportunity: ImprovementOpportunity) -> List[str]:
        """Infer how the improvement target is typically used."""
        # This would analyze call patterns from trace data
        return [
            f"Used in {opportunity.category} context",
            "Called during validation process",
            "Part of analysis pipeline",
        ]

    # Additional helper methods for risk and impact analysis
    async def _calculate_indirect_benefits(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate indirect benefits of this improvement."""
        return {
            "improved_testability": 0.3,
            "reduced_maintenance_cost": 0.4,
            "better_onboarding_experience": 0.2,
            "increased_development_velocity": 0.3,
        }

    async def _identify_stakeholders(self, opportunity: ImprovementOpportunity) -> List[str]:
        """Identify who would be affected by this improvement."""
        return [
            "vibelint developers",
            "vibelint users",
            "code quality engineers",
            "future maintainers",
        ]

    async def _predict_cascade_effects(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> List[str]:
        """Predict cascade effects of this improvement."""
        return [
            f"May require updates to {len(opportunity.dependency_impact)} dependent modules",
            "Could improve overall code quality metrics",
            "May reduce future technical debt accumulation",
        ]

    async def _predict_metrics_improvement(
        self, opportunity: ImprovementOpportunity, analysis_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Predict how metrics will improve."""
        return {
            "code_quality_score": f"+{opportunity.estimated_impact.get('maintainability', 0.5) * 10:.1f}%",
            "technical_debt_reduction": f"-{opportunity.estimated_impact.get('maintainability', 0.5) * 15:.1f}%",
            "development_velocity": f"+{opportunity.estimated_impact.get('testability', 0.3) * 5:.1f}%",
        }

    async def _assess_breaking_changes(self, opportunity: ImprovementOpportunity) -> str:
        """Assess risk of breaking changes."""
        if opportunity.category == "architecture":
            return "high" if len(opportunity.dependency_impact) > 5 else "medium"
        elif opportunity.severity == "high":
            return "medium"
        else:
            return "low"

    async def _assess_performance_impact(self, opportunity: ImprovementOpportunity) -> str:
        """Assess performance impact risk."""
        if "performance" in opportunity.current_state.lower():
            return "medium"
        elif opportunity.execution_criticality > 0.7:
            return "medium"
        else:
            return "low"

    async def _assess_maintenance_burden(self, opportunity: ImprovementOpportunity) -> str:
        """Assess maintenance burden risk."""
        if len(opportunity.implementation_steps) > 5:
            return "high"
        elif opportunity.category == "architecture":
            return "medium"
        else:
            return "low"

    async def _suggest_risk_mitigation(self, opportunity: ImprovementOpportunity) -> List[str]:
        """Suggest risk mitigation strategies."""
        return [
            "Implement changes incrementally",
            "Add comprehensive tests before refactoring",
            "Use feature flags for gradual rollout",
            "Monitor performance metrics during implementation",
            "Maintain backwards compatibility where possible",
        ]

    async def route_to_optimal_llm(self, context: LLMContext, task_type: str) -> str:
        """Route the request to the optimal LLM based on context and task."""

        # Calculate context size
        context_size = len(json.dumps(asdict(context), default=str))

        # Route based on complexity and context size
        if context_size > 20000:  # Large context
            return "claude_cli"  # Has largest context window
        elif task_type in ["architectural_changes", "complex_refactoring"]:
            return "chip"  # Best for deep reasoning
        elif task_type in ["simple_fixes", "code_generation"]:
            return "claudia"  # Fast for simple tasks
        else:
            return "chip"  # Default to reasoning model

    def export_context_for_llm(self, context: LLMContext, output_path: Path) -> Dict[str, Any]:
        """Export context in LLM-optimized format."""
        llm_context = {
            "vibelint_context_v1": {
                "metadata": {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "context_completeness": context.context_completeness,
                    "confidence_score": sum(context.confidence_indicators.values())
                    / len(context.confidence_indicators),
                },
                "project_understanding": context.project_overview,
                "improvement_focus": (
                    asdict(context.improvement_opportunity)
                    if context.improvement_opportunity
                    else {}
                ),
                "codebase_context": {
                    "filesystem": context.filesystem_context,
                    "semantics": context.semantic_context,
                    "dependencies": context.dependency_context,
                    "execution": context.execution_context,
                },
                "decision_support": {
                    "similar_patterns": context.similar_patterns,
                    "impact_analysis": context.impact_analysis,
                    "suggested_approach": context.suggested_approach,
                    "risk_assessment": context.risk_assessment,
                    "implementation_guidance": context.implementation_guidance,
                },
                "llm_instructions": context.llm_instructions,
            }
        }

        with open(output_path, "w") as f:
            json.dump(llm_context, f, indent=2, default=str)

        print(f"📋 LLM context exported to: {output_path}")
        return llm_context


# Main integration function
async def engineer_context_for_vibelint_improvement(
    project_root: Path,
    improvement_opportunity: ImprovementOpportunity,
    analysis_results: Dict[str, Any],
    multi_rep_analyzer: MultiRepresentationAnalyzer,
    dependency_manager: DependencyGraphManager,
) -> LLMContext:
    """
    Main entry point: Engineer rich context for vibelint self-improvement.

    This is the culmination of vibelint's context engineering - taking raw codebase
    analysis and transforming it into the perfect context for foundation models
    to make intelligent code improvements.
    """

    print(f"🚀 Engineering LLM context for improvement: {improvement_opportunity.opportunity_id}")

    context_engineer = VibelintContextEngineer(project_root, multi_rep_analyzer, dependency_manager)

    # Engineer the context
    context = await context_engineer.engineer_context_for_improvement(
        improvement_opportunity, analysis_results
    )

    # Export for LLM consumption
    export_path = (
        project_root
        / ".vibelint-self-improvement"
        / f"llm_context_{improvement_opportunity.opportunity_id}.json"
    )
    export_path.parent.mkdir(exist_ok=True)

    context_engineer.export_context_for_llm(context, export_path)

    print("✅ Context engineering complete:")
    print(f"  - Completeness: {context.context_completeness:.1%}")
    print(
        f"  - Confidence: {sum(context.confidence_indicators.values()) / len(context.confidence_indicators):.1%}"
    )
    print(
        f"  - Optimal LLM: {await context_engineer.route_to_optimal_llm(context, improvement_opportunity.category)}"
    )

    return context


if __name__ == "__main__":
    print(
        "🧠 Vibelint Context Engineering - The bridge between codebase analysis and LLM intelligence"
    )
    print(
        "This module transforms raw code analysis into rich context that enables foundation models"
    )
    print(
        "to make sophisticated architectural and code quality improvements beyond simple linting."
    )
