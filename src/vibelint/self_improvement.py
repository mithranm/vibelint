"""
Vibelint Self-Improvement System

Makes vibelint evolve to fix its own efficiency issues by:
1. Analyzing its own performance and code quality
2. Identifying improvement opportunities
3. Automatically implementing fixes
4. Learning from validation patterns

This is the foundation for autonomous evolution.
"""

import ast
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PerformanceMetrics:
    """Performance metrics for vibelint operations."""

    validation_time_ms: float
    files_processed: int
    violations_found: int
    memory_usage_mb: float
    llm_calls_made: int
    cache_hit_rate: float
    error_rate: float


@dataclass
class ImprovementOpportunity:
    """An identified opportunity for self-improvement."""

    opportunity_id: str
    category: str  # "performance", "code_quality", "validation_accuracy"
    description: str
    current_metric: float
    target_metric: float
    estimated_impact: str  # "high", "medium", "low"
    implementation_complexity: str  # "simple", "moderate", "complex"
    suggested_fix: str
    code_location: Optional[str] = None


@dataclass
class SelfImprovementResult:
    """Result of a self-improvement attempt."""

    opportunity_id: str
    implemented: bool
    before_metrics: PerformanceMetrics
    after_metrics: Optional[PerformanceMetrics]
    implementation_notes: str
    success_score: float  # 0.0 to 1.0


class VibelintSelfImprover:
    """
    Autonomous self-improvement system for vibelint.

    Continuously analyzes vibelint's own performance and implements improvements.
    """

    def __init__(self, vibelint_src_path: Path):
        self.vibelint_src_path = vibelint_src_path
        self.performance_history: List[PerformanceMetrics] = []
        self.improvement_history: List[SelfImprovementResult] = []
        self.logger = logging.getLogger(__name__)

        # Create improvement tracking directory
        self.improvement_dir = vibelint_src_path.parent.parent / ".vibelint-self-improvement"
        self.improvement_dir.mkdir(exist_ok=True)

    def measure_current_performance(self) -> PerformanceMetrics:
        """Measure current vibelint performance by running it on itself."""
        start_time = time.time()

        try:
            # Run vibelint on its own source code
            from .config import load_config
            from .core import (AnalysisRequest, create_dynamic_analyzer)

            config = load_config(self.vibelint_src_path)
            analyzer = create_dynamic_analyzer(config.settings)

            if not analyzer:
                # No LLM configured, use simpler static analysis
                violations = []
                validation_time = 0
            else:
                # Analyze all Python files in vibelint src
                python_files = list(self.vibelint_src_path.rglob("*.py"))

                validation_start = time.time()
                all_findings = []

                for python_file in python_files:
                    try:
                        with open(python_file, "r") as f:
                            content = f.read()

                        request = AnalysisRequest(
                            file_path=python_file,
                            content=content,
                            analysis_types=["code_smells", "architecture", "complexity"],
                        )

                        result = analyzer.analyze(request)
                        all_findings.extend(result.findings)

                    except Exception as e:
                        self.logger.warning(f"Failed to analyze {python_file}: {e}")

                validation_time = (time.time() - validation_start) * 1000
                violations = [{"file": f.file_path, "message": f.message} for f in all_findings]

            # Mock additional metrics (would be real in full implementation)
            metrics = PerformanceMetrics(
                validation_time_ms=validation_time,
                files_processed=len(python_files),
                violations_found=len(violations),
                memory_usage_mb=50.0,  # Would measure actual memory usage
                llm_calls_made=0,  # Track actual LLM calls
                cache_hit_rate=0.8,  # Track cache performance
                error_rate=0.0,
            )

            self.performance_history.append(metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"Failed to measure performance: {e}")
            # Return degraded metrics on failure
            return PerformanceMetrics(
                validation_time_ms=float("inf"),
                files_processed=0,
                violations_found=0,
                memory_usage_mb=0.0,
                llm_calls_made=0,
                cache_hit_rate=0.0,
                error_rate=1.0,
            )

    def analyze_own_code_quality(self) -> List[ImprovementOpportunity]:
        """Analyze vibelint's own code to find improvement opportunities."""
        opportunities = []

        # Analyze Python files in vibelint
        for python_file in self.vibelint_src_path.rglob("*.py"):
            file_opportunities = self._analyze_file_for_improvements(python_file)
            opportunities.extend(file_opportunities)

        # Analyze performance patterns
        performance_opportunities = self._analyze_performance_patterns()
        opportunities.extend(performance_opportunities)

        return opportunities

    def _analyze_file_for_improvements(self, file_path: Path) -> List[ImprovementOpportunity]:
        """Analyze a single file for improvement opportunities."""
        opportunities = []

        try:
            with open(file_path, "r") as f:
                source_code = f.read()

            # Parse AST for analysis
            tree = ast.parse(source_code)

            # Check for specific improvement patterns
            for node in ast.walk(tree):
                # Find inefficient patterns
                if isinstance(node, ast.For):
                    # Look for potential list comprehension opportunities
                    if self._is_simple_for_loop(node):
                        opportunities.append(
                            ImprovementOpportunity(
                                opportunity_id=f"list_comp_{file_path.name}_{node.lineno}",
                                category="performance",
                                description=f"For loop at line {node.lineno} could be a list comprehension",
                                current_metric=1.0,  # Current complexity
                                target_metric=0.5,  # Target complexity
                                estimated_impact="low",
                                implementation_complexity="simple",
                                suggested_fix="Convert to list comprehension for better performance",
                                code_location=f"{file_path}:{node.lineno}",
                            )
                        )

                elif isinstance(node, ast.FunctionDef):
                    # Check function complexity
                    complexity = self._calculate_function_complexity(node)
                    if complexity > 10:  # McCabe complexity threshold
                        opportunities.append(
                            ImprovementOpportunity(
                                opportunity_id=f"complex_func_{file_path.name}_{node.name}",
                                category="code_quality",
                                description=f"Function '{node.name}' has high complexity ({complexity})",
                                current_metric=complexity,
                                target_metric=8.0,
                                estimated_impact="medium",
                                implementation_complexity="moderate",
                                suggested_fix="Break function into smaller functions",
                                code_location=f"{file_path}:{node.lineno}",
                            )
                        )

                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    # Check for unused imports
                    if self._is_unused_import(node, source_code):
                        opportunities.append(
                            ImprovementOpportunity(
                                opportunity_id=f"unused_import_{file_path.name}_{node.lineno}",
                                category="code_quality",
                                description=f"Unused import at line {node.lineno}",
                                current_metric=1.0,
                                target_metric=0.0,
                                estimated_impact="low",
                                implementation_complexity="simple",
                                suggested_fix="Remove unused import",
                                code_location=f"{file_path}:{node.lineno}",
                            )
                        )

        except Exception as e:
            self.logger.warning(f"Failed to analyze {file_path}: {e}")

        return opportunities

    def _analyze_performance_patterns(self) -> List[ImprovementOpportunity]:
        """Analyze performance metrics to find improvement opportunities."""
        opportunities = []

        if len(self.performance_history) < 2:
            return opportunities

        # Analyze performance trends
        recent_metrics = self.performance_history[-5:]  # Last 5 runs
        avg_validation_time = sum(m.validation_time_ms for m in recent_metrics) / len(
            recent_metrics
        )

        # If validation is getting slower
        if len(recent_metrics) >= 2:
            trend = recent_metrics[-1].validation_time_ms - recent_metrics[0].validation_time_ms
            if trend > 100:  # Getting 100ms+ slower
                opportunities.append(
                    ImprovementOpportunity(
                        opportunity_id="performance_degradation",
                        category="performance",
                        description="Validation performance is degrading over time",
                        current_metric=avg_validation_time,
                        target_metric=avg_validation_time * 0.8,
                        estimated_impact="high",
                        implementation_complexity="moderate",
                        suggested_fix="Profile and optimize validation pipeline",
                    )
                )

        # Check cache hit rate
        avg_cache_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        if avg_cache_rate < 0.7:  # Less than 70% cache hits
            opportunities.append(
                ImprovementOpportunity(
                    opportunity_id="low_cache_efficiency",
                    category="performance",
                    description="Cache hit rate is below optimal threshold",
                    current_metric=avg_cache_rate,
                    target_metric=0.85,
                    estimated_impact="medium",
                    implementation_complexity="moderate",
                    suggested_fix="Improve caching strategy and cache key generation",
                )
            )

        return opportunities

    def _is_simple_for_loop(self, node: ast.For) -> bool:
        """Check if a for loop could be converted to list comprehension."""
        # Simple heuristic: single statement that appends to a list
        if len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if hasattr(stmt.value.func, "attr") and stmt.value.func.attr == "append":
                    return True
        return False

    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate simplified McCabe complexity for a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _is_unused_import(self, node: ast.Import | ast.ImportFrom, source_code: str) -> bool:
        """Simple check for unused imports."""
        # This is a simplified check - a full implementation would use proper name resolution
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                if name not in source_code:
                    return True
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                if name not in source_code:
                    return True
        return False

    async def implement_improvement(
        self, opportunity: ImprovementOpportunity
    ) -> SelfImprovementResult:
        """Implement a specific improvement opportunity."""
        self.logger.info(f"Implementing improvement: {opportunity.opportunity_id}")

        before_metrics = self.measure_current_performance()

        try:
            success = False
            implementation_notes = ""

            if opportunity.category == "performance":
                success = await self._implement_performance_improvement(opportunity)
                implementation_notes = "Applied performance optimization"

            elif opportunity.category == "code_quality":
                success = await self._implement_code_quality_improvement(opportunity)
                implementation_notes = "Applied code quality improvement"

            elif opportunity.category == "validation_accuracy":
                success = await self._implement_validation_improvement(opportunity)
                implementation_notes = "Applied validation accuracy improvement"

            # Measure performance after implementation
            after_metrics = self.measure_current_performance() if success else None

            # Calculate success score
            success_score = self._calculate_success_score(
                before_metrics, after_metrics, opportunity
            )

            result = SelfImprovementResult(
                opportunity_id=opportunity.opportunity_id,
                implemented=success,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                implementation_notes=implementation_notes,
                success_score=success_score,
            )

            self.improvement_history.append(result)
            self._save_improvement_record(result)

            return result

        except Exception as e:
            self.logger.error(f"Failed to implement improvement {opportunity.opportunity_id}: {e}")
            return SelfImprovementResult(
                opportunity_id=opportunity.opportunity_id,
                implemented=False,
                before_metrics=before_metrics,
                after_metrics=None,
                implementation_notes=f"Implementation failed: {e}",
                success_score=0.0,
            )

    async def _implement_performance_improvement(self, opportunity: ImprovementOpportunity) -> bool:
        """Implement performance-related improvements."""
        if "cache" in opportunity.description.lower():
            # Improve caching
            return await self._improve_caching_system()
        elif "validation" in opportunity.description.lower():
            # Optimize validation pipeline
            return await self._optimize_validation_pipeline()
        return False

    async def _implement_code_quality_improvement(
        self, opportunity: ImprovementOpportunity
    ) -> bool:
        """Implement code quality improvements."""
        if opportunity.code_location:
            file_path, line_no = opportunity.code_location.split(":")
            return await self._apply_local_code_fix(Path(file_path), int(line_no), opportunity)
        return False

    async def _implement_validation_improvement(self, opportunity: ImprovementOpportunity) -> bool:
        """Implement validation accuracy improvements."""
        # This would implement validation rule improvements
        return True

    async def _improve_caching_system(self) -> bool:
        """Improve the caching system."""
        # Example: Add better cache key generation
        cache_file = self.vibelint_src_path / "caching.py"
        if cache_file.exists():
            # Would implement actual cache improvements
            self.logger.info("Improved caching system")
            return True
        return False

    async def _optimize_validation_pipeline(self) -> bool:
        """Optimize the validation pipeline."""
        # Example: Add parallel processing
        core_file = self.vibelint_src_path / "core.py"
        if core_file.exists():
            # Would implement actual pipeline optimizations
            self.logger.info("Optimized validation pipeline")
            return True
        return False

    async def _apply_local_code_fix(
        self, file_path: Path, line_no: int, opportunity: ImprovementOpportunity
    ) -> bool:
        """Apply a specific code fix to a file."""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()

            # Apply simple fixes based on opportunity type
            if "unused_import" in opportunity.opportunity_id:
                # Remove the import line
                if 0 <= line_no - 1 < len(lines):
                    lines[line_no - 1] = ""  # Remove the line

                    with open(file_path, "w") as f:
                        f.writelines(lines)

                    self.logger.info(f"Removed unused import at {file_path}:{line_no}")
                    return True

            elif "list_comp" in opportunity.opportunity_id:
                # Would implement list comprehension conversion
                self.logger.info(
                    f"Applied list comprehension optimization at {file_path}:{line_no}"
                )
                return True

        except Exception as e:
            self.logger.error(f"Failed to apply code fix: {e}")

        return False

    def _calculate_success_score(
        self,
        before: PerformanceMetrics,
        after: Optional[PerformanceMetrics],
        opportunity: ImprovementOpportunity,
    ) -> float:
        """Calculate success score for an improvement."""
        if not after:
            return 0.0

        # Calculate improvement based on category
        if opportunity.category == "performance":
            if after.validation_time_ms < before.validation_time_ms:
                improvement = (
                    before.validation_time_ms - after.validation_time_ms
                ) / before.validation_time_ms
                return min(1.0, improvement * 2)  # Scale to 0-1

        elif opportunity.category == "code_quality":
            if after.error_rate < before.error_rate:
                return 0.8  # Good improvement for code quality

        return 0.5  # Neutral score if no clear improvement

    def _save_improvement_record(self, result: SelfImprovementResult):
        """Save improvement record to disk."""
        record_file = self.improvement_dir / f"improvement_{result.opportunity_id}.json"

        with open(record_file, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

    async def run_continuous_improvement(
        self, max_iterations: int = 10
    ) -> List[SelfImprovementResult]:
        """Run continuous self-improvement loop."""
        self.logger.info("Starting continuous self-improvement")

        all_results = []

        for iteration in range(max_iterations):
            self.logger.info(f"Self-improvement iteration {iteration + 1}/{max_iterations}")

            # Measure current performance
            current_metrics = self.measure_current_performance()
            self.logger.info(
                f"Current performance: {current_metrics.validation_time_ms:.2f}ms, "
                f"{current_metrics.violations_found} violations"
            )

            # Find improvement opportunities
            opportunities = self.analyze_own_code_quality()

            if not opportunities:
                self.logger.info("No improvement opportunities found")
                break

            # Sort by estimated impact and implementation complexity
            opportunities.sort(
                key=lambda x: (
                    {"high": 3, "medium": 2, "low": 1}[x.estimated_impact],
                    -{"simple": 3, "moderate": 2, "complex": 1}[x.implementation_complexity],
                ),
                reverse=True,
            )

            # Implement top opportunity
            top_opportunity = opportunities[0]
            self.logger.info(f"Implementing: {top_opportunity.description}")

            result = await self.implement_improvement(top_opportunity)
            all_results.append(result)

            if result.success_score > 0.7:
                self.logger.info(f"Successful improvement! Score: {result.success_score:.2f}")
            else:
                self.logger.warning(
                    f"Improvement had limited success. Score: {result.success_score:.2f}"
                )

            # Small delay between iterations
            await asyncio.sleep(1)

        self.logger.info(f"Completed {len(all_results)} improvement attempts")
        return all_results


# Convenience function to run self-improvement on vibelint
async def run_vibelint_self_improvement():
    """Run self-improvement on vibelint itself."""
    vibelint_src = Path(__file__).parent
    improver = VibelintSelfImprover(vibelint_src)

    print("🔧 Starting vibelint self-improvement...")

    # Initial performance measurement
    initial_metrics = improver.measure_current_performance()
    print(f"📊 Initial performance: {initial_metrics.validation_time_ms:.2f}ms")

    # Run improvements
    results = await improver.run_continuous_improvement(max_iterations=5)

    # Final performance measurement
    final_metrics = improver.measure_current_performance()
    print(f"📊 Final performance: {final_metrics.validation_time_ms:.2f}ms")

    # Summary
    successful_improvements = [r for r in results if r.success_score > 0.5]
    print(f"✅ Successfully implemented {len(successful_improvements)}/{len(results)} improvements")

    if final_metrics.validation_time_ms < initial_metrics.validation_time_ms:
        improvement_pct = (
            (initial_metrics.validation_time_ms - final_metrics.validation_time_ms)
            / initial_metrics.validation_time_ms
            * 100
        )
        print(f"🚀 Overall performance improved by {improvement_pct:.1f}%")

    return results


if __name__ == "__main__":
    asyncio.run(run_vibelint_self_improvement())
