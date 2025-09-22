"""
Vibelint: Self-Improving Code Analysis Engine

A sophisticated code analysis system that combines filesystem, vector, and graph
representations to enable foundation models to understand and improve codebases
autonomously. Follows MCP-style capability assignment patterns.

Key Components:
- Multi-representation analysis (filesystem, vector, graph, runtime)
- Hybrid dependency graphs with NetworkX + Qdrant embeddings
- Context engineering for foundation model consumption
- Self-improvement and autonomous evolution
- Zero-config auto-discovery scaling

Usage:
    import vibelint

    # Create engine with auto-discovery
    engine = vibelint.VibelintEngine()

    # Assign capabilities
    engine.assign_capability("embedding", "vanguard_ensemble")
    engine.assign_capability("reasoning", "claude")
    engine.assign_capability("memory", "qdrant_enhanced")
    engine.assign_capability("tracing", "runtime_mock")

    # Analyze and improve
    results = await engine.analyze_and_improve("path/to/project")
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .auto_discovery import AutoDiscovery
from .config import Config as VibelintConfig
from .dependency_graph_manager import DependencyGraphManager
from .llm_context_engineer import VibelintContextEngineer as LLMContextEngineer
from .multi_representation_analyzer import MultiRepresentationAnalyzer
from .runtime_tracer import RuntimeTracer
from .self_improvement import VibelintSelfImprover as SelfImprovementSystem

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResults:
    """Results from vibelint analysis and improvement."""

    filesystem_analysis: Dict[str, Any]
    vector_analysis: Dict[str, Any]
    graph_analysis: Dict[str, Any]
    runtime_analysis: Dict[str, Any]
    improvement_suggestions: List[Dict[str, Any]]
    context_engineered: Dict[str, Any]
    self_improvements: Optional[Dict[str, Any]] = None


class CapabilityRegistry:
    """Registry for managing assigned capabilities."""

    def __init__(self):
        self.capabilities = {
            "embedding": "vanguard_ensemble",  # Default
            "reasoning": "claude",  # Default
            "memory": "qdrant_enhanced",  # Default
            "tracing": "runtime_mock",  # Default
            "discovery": "auto_zero_config",  # Default
        }

        # Available capability implementations
        self.available_capabilities = {
            "embedding": [
                "vanguard_ensemble",  # VanguardOne + VanguardTwo
                "vanguard_one",  # VanguardOne only
                "vanguard_two",  # VanguardTwo only
                "openai_ada",  # OpenAI embeddings
                "sentence_transformers",  # Local embeddings
            ],
            "reasoning": [
                "claude",  # Claude API
                "gpt_oss_120b",  # GPT-OSS 120B
                "claudia",  # Local Claudia model
                "chip",  # Local Chip model
                "ollama",  # Local Ollama models
            ],
            "memory": [
                "qdrant_enhanced",  # Qdrant with EBR
                "qdrant_basic",  # Basic Qdrant
                "chromadb",  # ChromaDB
                "in_memory",  # RAM-only
            ],
            "tracing": [
                "runtime_mock",  # Mock value injection
                "static_analysis",  # AST-only analysis
                "hybrid",  # Mock + static
                "disabled",  # No tracing
            ],
            "discovery": [
                "auto_zero_config",  # Automatic discovery
                "config_file",  # Explicit config
                "manual",  # Manual specification
            ],
        }

    def assign(self, capability_type: str, implementation: str) -> bool:
        """Assign a specific implementation to a capability type."""
        if capability_type not in self.available_capabilities:
            logger.error(f"Unknown capability type: {capability_type}")
            return False

        if implementation not in self.available_capabilities[capability_type]:
            logger.error(f"Unknown implementation '{implementation}' for '{capability_type}'")
            logger.info(f"Available: {self.available_capabilities[capability_type]}")
            return False

        self.capabilities[capability_type] = implementation
        logger.info(f"Assigned {capability_type} -> {implementation}")
        return True

    def get(self, capability_type: str) -> str:
        """Get the assigned implementation for a capability type."""
        return self.capabilities.get(capability_type, "unknown")

    def list_available(self, capability_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List available implementations for capabilities."""
        if capability_type:
            return {capability_type: self.available_capabilities.get(capability_type, [])}
        return self.available_capabilities.copy()


class VibelintEngine:
    """
    Main orchestrating engine for vibelint analysis and improvement.

    Provides MCP-style capability assignment and orchestrates all analysis
    components to deliver comprehensive code understanding and improvement.
    """

    def __init__(self, project_path: Optional[Union[str, Path]] = None):
        """Initialize the vibelint engine."""
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.capability_registry = CapabilityRegistry()

        # Initialize core components (lazy-loaded)
        self._config: Optional[VibelintConfig] = None
        self._auto_discovery: Optional[AutoDiscovery] = None
        self._analyzer: Optional[MultiRepresentationAnalyzer] = None
        self._graph_manager: Optional[DependencyGraphManager] = None
        self._context_engineer: Optional[LLMContextEngineer] = None
        self._runtime_tracer: Optional[RuntimeTracer] = None
        self._self_improvement: Optional[SelfImprovementSystem] = None

        logger.info(f"VibelintEngine initialized for project: {self.project_path}")

    def assign_capability(self, capability_type: str, implementation: str) -> bool:
        """
        Assign a specific implementation to a capability type.

        Args:
            capability_type: Type of capability (embedding, reasoning, memory, tracing, discovery)
            implementation: Specific implementation to use

        Returns:
            True if assignment successful, False otherwise
        """
        return self.capability_registry.assign(capability_type, implementation)

    def list_capabilities(self, capability_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List available capability implementations."""
        return self.capability_registry.list_available(capability_type)

    def get_assigned_capabilities(self) -> Dict[str, str]:
        """Get currently assigned capabilities."""
        return self.capability_registry.capabilities.copy()

    async def _ensure_components_initialized(self):
        """Lazy-load and initialize all components based on assigned capabilities."""
        if self._config is None:
            discovery_type = self.capability_registry.get("discovery")
            if discovery_type == "auto_zero_config":
                self._auto_discovery = AutoDiscovery(self.project_path)
                discovered_config = await self._auto_discovery.discover_configuration()
                self._config = VibelintConfig(**discovered_config)
            else:
                self._config = VibelintConfig.from_project_path(self.project_path)

        if self._analyzer is None:
            embedding_type = self.capability_registry.get("embedding")
            self._analyzer = MultiRepresentationAnalyzer(
                config=self._config, embedding_strategy=embedding_type
            )

        if self._graph_manager is None:
            self._graph_manager = DependencyGraphManager(
                config=self._config, embedding_strategy=self.capability_registry.get("embedding")
            )

        if self._context_engineer is None:
            reasoning_type = self.capability_registry.get("reasoning")
            self._context_engineer = LLMContextEngineer(
                config=self._config,
                reasoning_strategy=reasoning_type,
                embedding_strategy=self.capability_registry.get("embedding"),
            )

        tracing_type = self.capability_registry.get("tracing")
        if self._runtime_tracer is None and tracing_type != "disabled":
            self._runtime_tracer = RuntimeTracer(config=self._config, tracing_strategy=tracing_type)

        if self._self_improvement is None:
            self._self_improvement = SelfImprovementSystem(
                config=self._config, reasoning_strategy=self.capability_registry.get("reasoning")
            )

    async def analyze(
        self,
        target_path: Optional[Union[str, Path]] = None,
        include_runtime: bool = True,
        include_improvements: bool = True,
    ) -> AnalysisResults:
        """
        Perform comprehensive analysis of the codebase.

        Args:
            target_path: Specific path to analyze (defaults to project_path)
            include_runtime: Whether to include runtime tracing analysis
            include_improvements: Whether to generate improvement suggestions

        Returns:
            AnalysisResults containing all analysis data
        """
        await self._ensure_components_initialized()

        analysis_path = Path(target_path) if target_path else self.project_path
        logger.info(f"Starting comprehensive analysis of: {analysis_path}")

        # Perform multi-representation analysis
        analysis_result = await self._analyzer.analyze_comprehensive(analysis_path)

        # Build and analyze dependency graph
        await self._graph_manager.build_project_graph(analysis_path)
        graph_analysis = await self._graph_manager.analyze_dependencies()

        # Runtime analysis (if enabled)
        runtime_analysis = {}
        if include_runtime and self._runtime_tracer:
            runtime_analysis = await self._runtime_tracer.trace_project_execution(analysis_path)

        # Generate improvement suggestions
        improvement_suggestions = []
        if include_improvements:
            improvement_suggestions = await self._analyzer.generate_improvements(
                analysis_result, graph_analysis, runtime_analysis
            )

        # Context engineering for LLM consumption
        context_engineered = await self._context_engineer.engineer_context(
            filesystem_analysis=analysis_result.get("filesystem", {}),
            vector_analysis=analysis_result.get("vector", {}),
            graph_analysis=graph_analysis,
            runtime_analysis=runtime_analysis,
            improvements=improvement_suggestions,
        )

        return AnalysisResults(
            filesystem_analysis=analysis_result.get("filesystem", {}),
            vector_analysis=analysis_result.get("vector", {}),
            graph_analysis=graph_analysis,
            runtime_analysis=runtime_analysis,
            improvement_suggestions=improvement_suggestions,
            context_engineered=context_engineered,
        )

    async def improve(self, analysis_results: AnalysisResults) -> Dict[str, Any]:
        """
        Apply improvements based on analysis results.

        Args:
            analysis_results: Results from analyze() call

        Returns:
            Dictionary containing improvement outcomes
        """
        await self._ensure_components_initialized()

        logger.info("Applying improvements based on analysis")

        # Use context engineering to create LLM-ready improvement requests
        improvement_context = analysis_results.context_engineered

        # Apply improvements through the reasoning capability with kaia-guardrails protection
        try:
            # Import kaia-guardrails for safety verification
            from kaia_guardrails.behavior_guardian import get_behavior_capture
            from kaia_guardrails.safety_rails import SafetyRails

            # Create safety rails for this improvement session
            safety_rails = SafetyRails(self.project_path, get_behavior_capture())

            # Use safety rails to verify improvements
            async with safety_rails.protection_mode():
                improvement_results = await self._context_engineer.execute_improvements(
                    improvement_context=improvement_context, target_path=self.project_path
                )

            return improvement_results

        except ImportError:
            logger.warning("kaia-guardrails not available - proceeding without safety verification")
            # Fallback to direct execution without safety rails
            improvement_results = await self._context_engineer.execute_improvements(
                improvement_context=improvement_context, target_path=self.project_path
            )
            return improvement_results

    async def analyze_and_improve(
        self,
        target_path: Optional[Union[str, Path]] = None,
        include_runtime: bool = True,
        auto_apply: bool = False,
    ) -> AnalysisResults:
        """
        Perform comprehensive analysis and optionally apply improvements.

        Args:
            target_path: Specific path to analyze (defaults to project_path)
            include_runtime: Whether to include runtime tracing analysis
            auto_apply: Whether to automatically apply improvements

        Returns:
            AnalysisResults with improvement outcomes included
        """
        # Perform analysis
        results = await self.analyze(
            target_path=target_path, include_runtime=include_runtime, include_improvements=True
        )

        # Apply improvements if requested
        if auto_apply and results.improvement_suggestions:
            improvement_outcomes = await self.improve(results)
            results.context_engineered["improvement_outcomes"] = improvement_outcomes

        return results

    async def self_improve(self) -> Dict[str, Any]:
        """
        Run self-improvement analysis on vibelint itself.

        Returns:
            Dictionary containing self-improvement results
        """
        await self._ensure_components_initialized()

        logger.info("Starting vibelint self-improvement analysis")

        # Analyze vibelint itself
        vibelint_path = Path(__file__).parent
        self_analysis = await self.analyze(
            target_path=vibelint_path, include_runtime=True, include_improvements=True
        )

        # Run self-improvement system
        self_improvements = await self._self_improvement.analyze_and_improve(vibelint_path)

        # Combine results
        self_analysis.self_improvements = self_improvements

        return {
            "analysis": self_analysis,
            "improvements": self_improvements,
            "meta": {
                "analyzed_path": str(vibelint_path),
                "capabilities": self.get_assigned_capabilities(),
                "timestamp": self._context_engineer.get_natural_timestamp(),
            },
        }

    async def get_project_context(
        self, query: Optional[str] = None, max_context_size: int = 32000
    ) -> Dict[str, Any]:
        """
        Get engineered context for the project suitable for LLM consumption.

        Args:
            query: Optional query to focus context generation
            max_context_size: Maximum context size in tokens

        Returns:
            Dictionary containing engineered context
        """
        await self._ensure_components_initialized()

        # Perform lightweight analysis for context
        analysis = await self.analyze(include_runtime=False, include_improvements=False)

        # Engineer context with optional query focus
        context = await self._context_engineer.engineer_focused_context(
            analysis_results=analysis, query=query, max_tokens=max_context_size
        )

        return context


# Convenience functions for common use cases
async def quick_analyze(
    project_path: Union[str, Path], embedding: str = "vanguard_ensemble", reasoning: str = "claude"
) -> AnalysisResults:
    """Quick analysis with common defaults."""
    engine = VibelintEngine(project_path)
    engine.assign_capability("embedding", embedding)
    engine.assign_capability("reasoning", reasoning)
    return await engine.analyze()


async def self_improve_vibelint() -> Dict[str, Any]:
    """Run self-improvement on vibelint itself."""
    engine = VibelintEngine()
    return await engine.self_improve()


def create_engine(**capabilities) -> VibelintEngine:
    """Create engine with specified capabilities."""
    engine = VibelintEngine()
    for cap_type, implementation in capabilities.items():
        engine.assign_capability(cap_type, implementation)
    return engine


# Export main classes and functions
__all__ = [
    "VibelintEngine",
    "AnalysisResults",
    "CapabilityRegistry",
    "quick_analyze",
    "self_improve_vibelint",
    "create_engine",
]

__version__ = "0.1.0"
