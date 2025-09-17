"""
Simple diagnostics for dual LLM setup with context probing.

Discovers actual context limits for both primary and orchestrator LLMs
using systematic testing approaches.

tools/vibelint/src/vibelint/diagnostics.py
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .context_probing import run_context_probing, ProbeConfig, ProbeResult
from .llm_manager import create_llm_manager, LLMRole

logger = logging.getLogger(__name__)

__all__ = ["run_diagnostics", "run_benchmark"]


class DualLLMDiagnostics:
    """Simple diagnostics for dual LLM setup."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with vibelint configuration."""
        self.config = config
        self.llm_manager = create_llm_manager(config)

    def _extract_llm_configs(self) -> Dict[str, Dict[str, Any]]:
        """Extract LLM configurations for probing."""
        llm_config = self.config.get("llm", {})
        configs = {}

        # Primary LLM
        if llm_config.get("primary_api_url"):
            configs["primary"] = {
                "api_base_url": llm_config["primary_api_url"],
                "model": llm_config["primary_model"],
                "temperature": llm_config.get("primary_temperature", 0.1)
            }

        # Orchestrator LLM
        if llm_config.get("orchestrator_api_url"):
            configs["orchestrator"] = {
                "api_base_url": llm_config["orchestrator_api_url"],
                "model": llm_config["orchestrator_model"],
                "temperature": llm_config.get("orchestrator_temperature", 0.2)
            }

        return configs

    async def run_context_probing(self) -> Dict[str, ProbeResult]:
        """Run context probing for both LLMs."""
        llm_configs = self._extract_llm_configs()

        if not llm_configs:
            logger.error("No LLM configurations found")
            return {}

        # Simple probe config optimized for both inference engines
        probe_config = ProbeConfig(
            max_tokens_to_test=50000,  # Test up to 50k tokens
            token_increment_strategy="exponential",
            enable_niah_testing=True,
            enable_performance_testing=True,
            max_probe_duration_minutes=10  # 10 min max per LLM
        )

        results = await run_context_probing(
            llm_configs,
            probe_config=probe_config,
            save_results=True,
            results_file=Path("llm_probe_results.json")
        )

        # Save calibration report
        if results:
            await self._save_calibration_results(results)

        return results

    async def _save_calibration_results(self, probe_results: Dict[str, ProbeResult]):
        """Save simple calibration report."""
        report = "# LLM Calibration Results\n\n"
        report += f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "**System:** vibelint dual LLM diagnostics\n\n"

        for llm_name, result in probe_results.items():
            report += f"## {llm_name.title()} LLM\n\n"
            report += f"- **Model:** {result.model}\n"
            report += f"- **API:** {result.api_base_url}\n"
            report += f"- **Engine:** {result.inference_engine.value}\n"
            report += f"- **Max Context:** {result.max_context_tokens:,} tokens\n"
            report += f"- **Effective Context:** {result.effective_context_tokens:,} tokens\n"
            report += f"- **Success Rate:** {result.success_rate:.1%}\n"
            report += f"- **Avg Latency:** {result.avg_latency_ms:.0f}ms\n\n"

        report += "## Recommended Configuration\n\n"
        report += "Update your `pyproject.toml`:\n\n"
        report += "```toml\n"
        report += "[tool.vibelint.llm]\n"

        for llm_name, result in probe_results.items():
            prefix = f"{llm_name}_"
            report += f"# Calibrated limits for {llm_name} LLM\n"
            report += f"{prefix}max_context_tokens = {result.effective_context_tokens}\n"
            report += f"{prefix}max_prompt_tokens = {result.recommended_max_prompt_tokens}\n"

        report += "enable_context_probing = false  # Disable after calibration\n"
        report += "```\n"

        Path("LLM_CALIBRATION_RESULTS.md").write_text(report, encoding="utf-8")

    async def benchmark_routing(self) -> Dict[str, Any]:
        """Simple benchmark of LLM routing logic."""
        if not self.llm_manager:
            return {"error": "No LLM manager configured"}

        # Test scenarios
        scenarios = [
            {"content": "Short docstring task", "task_type": "docstring", "expected": LLMRole.PRIMARY},
            {"content": "x" * 5000, "task_type": "analysis", "expected": LLMRole.ORCHESTRATOR},
            {"content": "Architecture analysis", "task_type": "architecture", "expected": LLMRole.ORCHESTRATOR},
        ]

        results = {"scenarios": [], "routing_accuracy": 0.0}
        correct = 0

        for scenario in scenarios:
            from .llm_manager import LLMRequest
            request = LLMRequest(scenario["content"], scenario["task_type"])
            selected = self.llm_manager.select_llm(request)

            is_correct = selected == scenario["expected"]
            if is_correct:
                correct += 1

            results["scenarios"].append({
                "task": scenario["task_type"],
                "content_size": len(scenario["content"]),
                "expected": scenario["expected"].value,
                "selected": selected.value,
                "correct": is_correct
            })

        results["routing_accuracy"] = correct / len(scenarios)
        return results


async def run_diagnostics(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run diagnostics with context probing.

    Args:
        config: Vibelint configuration dictionary

    Returns:
        Diagnostics results
    """
    diagnostics = DualLLMDiagnostics(config)

    try:
        print("=== LLM Context Probing ===")
        probe_results = await diagnostics.run_context_probing()

        if not probe_results:
            print("❌ No LLM configurations found or probing failed")
            return {"error": "Context probing failed"}

        # Print results
        for llm_name, result in probe_results.items():
            print(f"\n{llm_name.upper()} LLM:")
            print(f"  ✓ Max Context: {result.max_context_tokens:,} tokens")
            print(f"  ✓ Success Rate: {result.success_rate:.1%}")
            print(f"  ✓ Latency: {result.avg_latency_ms:.0f}ms")

        print(f"\n✅ Context probing completed")
        print(f"📄 Results saved to LLM_CALIBRATION_RESULTS.md")

        return {"probe_results": probe_results, "success": True}

    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")
        return {"error": str(e), "success": False}


async def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run LLM routing benchmark.

    Args:
        config: Vibelint configuration dictionary

    Returns:
        Benchmark results
    """
    diagnostics = DualLLMDiagnostics(config)

    try:
        print("=== LLM Routing Benchmark ===")
        results = await diagnostics.benchmark_routing()

        if "error" in results:
            print(f"❌ {results['error']}")
            return results

        print(f"Routing Accuracy: {results['routing_accuracy']:.1%}")

        for scenario in results["scenarios"]:
            status = "✓" if scenario["correct"] else "✗"
            print(f"  {status} {scenario['task']}: {scenario['selected']} (expected: {scenario['expected']})")

        return results

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Simple test
    test_config = {
        "llm": {
            "primary_api_url": "http://100.94.250.88:8001",
            "primary_model": "openai/gpt-oss-20b",
            "orchestrator_api_url": "http://100.116.54.128:11434",
            "orchestrator_model": "llama3.2:latest",
            "context_threshold": 3000
        }
    }

    asyncio.run(run_diagnostics(test_config))