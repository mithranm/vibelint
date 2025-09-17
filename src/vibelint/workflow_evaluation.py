"""
Workflow evaluation framework for measuring effectiveness and performance.

Provides metrics collection, benchmarking, and continuous improvement
capabilities for workflow analysis quality.

vibelint/src/vibelint/workflow_evaluation.py
"""

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

from .workflows.base import BaseWorkflow, WorkflowResult, WorkflowStatus

logger = logging.getLogger(__name__)

__all__ = ["WorkflowEvaluator", "EvaluationResult", "PerformanceMetrics", "QualityMetrics"]


@dataclass
class PerformanceMetrics:
    """Performance evaluation metrics."""
    execution_time_seconds: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    throughput_files_per_second: Optional[float] = None

    def meets_performance_criteria(self, criteria: Dict[str, float]) -> bool:
        """Check if performance meets specified criteria."""
        if "max_execution_time" in criteria:
            if self.execution_time_seconds > criteria["max_execution_time"]:
                return False

        if "max_memory_usage" in criteria and self.memory_usage_mb:
            if self.memory_usage_mb > criteria["max_memory_usage"]:
                return False

        return True


@dataclass
class QualityMetrics:
    """Quality evaluation metrics."""
    confidence_score: float
    accuracy_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    coverage_percentage: Optional[float] = None
    false_positive_rate: Optional[float] = None

    def meets_quality_criteria(self, criteria: Dict[str, float]) -> bool:
        """Check if quality meets specified criteria."""
        if "min_confidence_score" in criteria:
            if self.confidence_score < criteria["min_confidence_score"]:
                return False

        if "min_coverage_percentage" in criteria and self.coverage_percentage:
            if self.coverage_percentage < criteria["min_coverage_percentage"]:
                return False

        return True


@dataclass
class EvaluationResult:
    """Result of workflow evaluation."""
    workflow_id: str
    timestamp: str
    overall_score: float  # 0.0 to 1.0

    # Detailed metrics
    performance: PerformanceMetrics
    quality: QualityMetrics

    # Compliance checks
    meets_criteria: bool
    criteria_violations: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    improvement_opportunities: List[str] = field(default_factory=list)

    # Comparison data
    baseline_comparison: Optional[Dict[str, float]] = None
    trend_analysis: Optional[Dict[str, Any]] = None


class WorkflowEvaluator:
    """Evaluates workflow effectiveness and performance."""

    def __init__(self):
        self.evaluation_history: Dict[str, List[EvaluationResult]] = {}
        self.baselines: Dict[str, Dict[str, float]] = {}

    def evaluate_workflow_execution(
        self,
        workflow: BaseWorkflow,
        result: WorkflowResult
    ) -> EvaluationResult:
        """Evaluate a completed workflow execution."""

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Extract performance metrics
        performance = PerformanceMetrics(
            execution_time_seconds=result.metrics.execution_time_seconds or 0.0,
            memory_usage_mb=result.metrics.memory_usage_mb,
            cpu_usage_percent=result.metrics.cpu_usage_percent,
            throughput_files_per_second=self._calculate_throughput(result.metrics)
        )

        # Extract quality metrics
        quality = QualityMetrics(
            confidence_score=result.metrics.confidence_score,
            accuracy_score=result.metrics.accuracy_score,
            coverage_percentage=result.metrics.coverage_percentage
        )

        # Get evaluation criteria
        criteria = workflow.get_evaluation_criteria()

        # Check compliance
        meets_criteria = True
        violations = []

        if not performance.meets_performance_criteria(criteria.get("performance", {})):
            meets_criteria = False
            violations.append("Performance criteria not met")

        if not quality.meets_quality_criteria(criteria.get("quality", {})):
            meets_criteria = False
            violations.append("Quality criteria not met")

        # Calculate overall score
        overall_score = self._calculate_overall_score(performance, quality, result)

        # Generate recommendations
        recommendations = self._generate_recommendations(workflow, performance, quality, result)

        # Create evaluation result
        evaluation = EvaluationResult(
            workflow_id=workflow.workflow_id,
            timestamp=timestamp,
            overall_score=overall_score,
            performance=performance,
            quality=quality,
            meets_criteria=meets_criteria,
            criteria_violations=violations,
            recommendations=recommendations
        )

        # Add baseline comparison if available
        if workflow.workflow_id in self.baselines:
            evaluation.baseline_comparison = self._compare_to_baseline(
                workflow.workflow_id, performance, quality
            )

        # Store evaluation
        if workflow.workflow_id not in self.evaluation_history:
            self.evaluation_history[workflow.workflow_id] = []
        self.evaluation_history[workflow.workflow_id].append(evaluation)

        # Update baseline if this is a good execution
        if overall_score > 0.8 and meets_criteria:
            self._update_baseline(workflow.workflow_id, performance, quality)

        return evaluation

    def _calculate_throughput(self, metrics) -> Optional[float]:
        """Calculate files processed per second."""
        if metrics.execution_time_seconds and metrics.files_processed:
            if metrics.execution_time_seconds > 0:
                return metrics.files_processed / metrics.execution_time_seconds
        return None

    def _calculate_overall_score(
        self,
        performance: PerformanceMetrics,
        quality: QualityMetrics,
        result: WorkflowResult
    ) -> float:
        """Calculate overall workflow score."""

        # Base score from execution status
        if result.status == WorkflowStatus.COMPLETED:
            base_score = 0.6
        elif result.status == WorkflowStatus.FAILED:
            return 0.0
        else:
            base_score = 0.3

        # Quality contribution (40% weight)
        quality_score = quality.confidence_score * 0.4

        # Performance contribution (20% weight)
        # Penalize slow execution
        performance_score = 0.2
        if performance.execution_time_seconds > 0:
            # Assume target is 30 seconds, linear penalty after that
            if performance.execution_time_seconds <= 30:
                performance_score = 0.2
            else:
                performance_score = max(0.0, 0.2 * (60 - performance.execution_time_seconds) / 30)

        # Findings contribution (20% weight)
        findings_score = 0.0
        if result.metrics.findings_generated > 0:
            # More findings generally better, up to a point
            findings_score = min(0.2, result.metrics.findings_generated * 0.02)

        return min(1.0, base_score + quality_score + performance_score + findings_score)

    def _generate_recommendations(
        self,
        workflow: BaseWorkflow,
        performance: PerformanceMetrics,
        quality: QualityMetrics,
        result: WorkflowResult
    ) -> List[str]:
        """Generate improvement recommendations."""

        recommendations = []

        # Performance recommendations
        if performance.execution_time_seconds > 60:
            recommendations.append("Consider optimizing workflow execution time")

        if performance.memory_usage_mb and performance.memory_usage_mb > 1000:
            recommendations.append("Consider reducing memory usage")

        # Quality recommendations
        if quality.confidence_score < 0.7:
            recommendations.append("Consider improving analysis confidence through better heuristics")

        if result.metrics.errors_encountered > 0:
            recommendations.append("Address error handling to improve reliability")

        # Findings recommendations
        if result.metrics.findings_generated == 0:
            recommendations.append("Verify workflow is detecting issues appropriately")

        if quality.coverage_percentage and quality.coverage_percentage < 80:
            recommendations.append("Improve analysis coverage of target files")

        return recommendations

    def _compare_to_baseline(
        self,
        workflow_id: str,
        performance: PerformanceMetrics,
        quality: QualityMetrics
    ) -> Dict[str, float]:
        """Compare current metrics to baseline."""

        baseline = self.baselines[workflow_id]
        comparison = {}

        if "execution_time" in baseline:
            comparison["execution_time_ratio"] = (
                performance.execution_time_seconds / baseline["execution_time"]
            )

        if "confidence_score" in baseline:
            comparison["confidence_improvement"] = (
                quality.confidence_score - baseline["confidence_score"]
            )

        return comparison

    def _update_baseline(
        self,
        workflow_id: str,
        performance: PerformanceMetrics,
        quality: QualityMetrics
    ):
        """Update baseline metrics for workflow."""

        if workflow_id not in self.baselines:
            self.baselines[workflow_id] = {}

        baseline = self.baselines[workflow_id]

        # Update with exponential smoothing
        alpha = 0.3  # Smoothing factor

        if "execution_time" in baseline:
            baseline["execution_time"] = (
                alpha * performance.execution_time_seconds +
                (1 - alpha) * baseline["execution_time"]
            )
        else:
            baseline["execution_time"] = performance.execution_time_seconds

        if "confidence_score" in baseline:
            baseline["confidence_score"] = (
                alpha * quality.confidence_score +
                (1 - alpha) * baseline["confidence_score"]
            )
        else:
            baseline["confidence_score"] = quality.confidence_score

        logger.debug(f"Updated baseline for workflow {workflow_id}")

    def get_workflow_trends(self, workflow_id: str, days: int = 30) -> Optional[Dict[str, Any]]:
        """Get trend analysis for workflow over specified period."""

        if workflow_id not in self.evaluation_history:
            return None

        evaluations = self.evaluation_history[workflow_id]
        if len(evaluations) < 2:
            return None

        # Simple trend analysis
        recent_evaluations = evaluations[-min(days, len(evaluations)):]

        execution_times = [e.performance.execution_time_seconds for e in recent_evaluations]
        confidence_scores = [e.quality.confidence_score for e in recent_evaluations]
        overall_scores = [e.overall_score for e in recent_evaluations]

        trends = {
            "evaluation_count": len(recent_evaluations),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "avg_confidence_score": sum(confidence_scores) / len(confidence_scores),
            "avg_overall_score": sum(overall_scores) / len(overall_scores),
            "performance_trend": self._calculate_trend(execution_times),
            "quality_trend": self._calculate_trend(confidence_scores),
            "overall_trend": self._calculate_trend(overall_scores)
        }

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate simple trend direction."""
        if len(values) < 2:
            return "insufficient_data"

        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        if second_avg > first_avg * 1.05:
            return "improving"
        elif second_avg < first_avg * 0.95:
            return "degrading"
        else:
            return "stable"

    def generate_evaluation_report(self, workflow_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""

        if workflow_ids is None:
            workflow_ids = list(self.evaluation_history.keys())

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "workflows_evaluated": len(workflow_ids),
            "workflow_summaries": {}
        }

        for workflow_id in workflow_ids:
            if workflow_id in self.evaluation_history:
                evaluations = self.evaluation_history[workflow_id]
                latest = evaluations[-1] if evaluations else None

                if latest:
                    trends = self.get_workflow_trends(workflow_id)

                    report["workflow_summaries"][workflow_id] = {
                        "latest_score": latest.overall_score,
                        "meets_criteria": latest.meets_criteria,
                        "execution_count": len(evaluations),
                        "recommendations": latest.recommendations,
                        "trends": trends
                    }

        return report

    def export_evaluation_data(self, output_path: Path):
        """Export evaluation data for analysis."""
        import json

        export_data = {
            "evaluation_history": {},
            "baselines": self.baselines,
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Convert evaluation history to serializable format
        for workflow_id, evaluations in self.evaluation_history.items():
            export_data["evaluation_history"][workflow_id] = []
            for eval_result in evaluations:
                eval_dict = {
                    "workflow_id": eval_result.workflow_id,
                    "timestamp": eval_result.timestamp,
                    "overall_score": eval_result.overall_score,
                    "meets_criteria": eval_result.meets_criteria,
                    "performance": {
                        "execution_time_seconds": eval_result.performance.execution_time_seconds,
                        "memory_usage_mb": eval_result.performance.memory_usage_mb,
                        "throughput_files_per_second": eval_result.performance.throughput_files_per_second
                    },
                    "quality": {
                        "confidence_score": eval_result.quality.confidence_score,
                        "coverage_percentage": eval_result.quality.coverage_percentage
                    },
                    "recommendations": eval_result.recommendations
                }
                export_data["evaluation_history"][workflow_id].append(eval_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Evaluation data exported to {output_path}")
