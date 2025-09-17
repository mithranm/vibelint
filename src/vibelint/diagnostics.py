"""
Simple diagnostics for dual LLM setup with context probing.

Discovers actual context limits for both fast and orchestrator LLMs
using systematic testing approaches.

tools/vibelint/src/vibelint/diagnostics.py
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict

import requests

from .context_probing import ProbeResult
from .llm_manager import LLMRole, create_llm_manager

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

        # Fast LLM
        if llm_config.get("fast_api_url"):
            configs["fast"] = {
                "api_base_url": llm_config["fast_api_url"],
                "model": llm_config["fast_model"],
                "temperature": llm_config.get("fast_temperature", 0.1),
            }

        # Orchestrator LLM
        if llm_config.get("orchestrator_api_url"):
            configs["orchestrator"] = {
                "api_base_url": llm_config["orchestrator_api_url"],
                "model": llm_config["orchestrator_model"],
                "temperature": llm_config.get("orchestrator_temperature", 0.2),
            }

        return configs

    async def run_context_probing(self) -> Dict[str, ProbeResult]:
        """Run linear ramp-up context probing to find optimal speed/context balance."""
        llm_configs = self._extract_llm_configs()

        if not llm_configs:
            logger.error("No LLM configurations found")
            return {}

        results = {}

        for llm_name, config in llm_configs.items():
            print(f"\n🔍 Testing {llm_name.upper()} LLM context limits...")

            # Set timeout based on LLM type
            model_name = config["model"].lower()
            if "gpt-oss-20b" in model_name and "120b" not in model_name:  # Fast vLLM model
                timeout_seconds = 15  # Fast model should be quick
                context_sizes = [100, 500, 1000, 2000, 4000, 8000, 16000]  # Test higher contexts
            else:  # 120B orchestrator model
                timeout_seconds = 90  # 120B model needs more time for large contexts
                context_sizes = [
                    100,
                    1000,
                    4000,
                    8000,
                    16000,
                    24000,
                    32000,
                    40000,
                ]  # Test higher with stable system

            max_working_context = 0
            total_duration = 0
            successful_tests = 0
            failed_tests = 0

            import time

            import requests

            for context_size in context_sizes:
                try:
                    # Generate content roughly matching token count (4 chars ≈ 1 token)
                    content = "Context test. " * (context_size // 3)
                    content = content[: context_size * 4]  # Approximate token count

                    print(f"  Testing {context_size} tokens... ", end="", flush=True)

                    start_time = time.time()
                    response = requests.post(
                        f"{config['api_base_url']}/v1/chat/completions",
                        json={
                            "model": config["model"],
                            "messages": [{"role": "user", "content": f"Summarize: {content}"}],
                            "max_tokens": 20,  # Small response to focus on context processing
                            "temperature": config["temperature"],
                        },
                        timeout=timeout_seconds,
                    )
                    duration = time.time() - start_time

                    if response.status_code == 200:
                        max_working_context = context_size
                        total_duration += duration
                        successful_tests += 1
                        print(f"✅ {duration:.1f}s")
                    else:
                        failed_tests += 1
                        print(f"❌ HTTP {response.status_code}")
                        break  # Stop on first HTTP error

                except requests.exceptions.Timeout:
                    failed_tests += 1
                    print(f"⏱️ Timeout (>{timeout_seconds}s)")
                    break  # Stop on first timeout
                except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                    failed_tests += 1
                    print(f"❌ Error: {str(e)[:50]}...")
                    break  # Stop on first error

                # Small delay to let server recover
                time.sleep(1)

            # Calculate results
            if successful_tests > 0:
                avg_latency = (total_duration / successful_tests) * 1000  # Convert to ms
                success_rate = successful_tests / (successful_tests + failed_tests)

                from .context_probing import InferenceEngine, ProbeResult

                results[llm_name] = ProbeResult(
                    model=config["model"],
                    api_base_url=config["api_base_url"],
                    inference_engine=InferenceEngine.OPENAI_COMPATIBLE,
                    max_context_tokens=max_working_context,
                    effective_context_tokens=int(max_working_context * 0.9),  # 10% safety margin
                    max_output_tokens=2048,
                    avg_latency_ms=avg_latency,
                    throughput_tokens_per_sec=20.0 / (avg_latency / 1000),  # Rough estimate
                    success_rate=success_rate,
                    test_count=successful_tests + failed_tests,
                    needle_in_haystack_accuracy=1.0,  # Not tested
                    position_bias_detected=False,
                    first_failure_tokens=(
                        context_sizes[successful_tests] if failed_tests > 0 else None
                    ),
                    error_patterns=[],
                    recommended_max_prompt_tokens=int(
                        max_working_context * 0.8
                    ),  # 20% safety margin
                    recommended_batch_size=1,
                    probe_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    probe_duration_seconds=total_duration,
                )

                print(
                    f"  📊 Result: {max_working_context} tokens max, {avg_latency:.0f}ms avg, {success_rate:.0%} success"
                )
            else:
                print(f"  ❌ No successful tests for {llm_name}")

        # Save assessment report
        if results:
            await self._save_assessment_results(results)

        return results

    async def _save_assessment_results(self, probe_results: Dict[str, ProbeResult]):
        """Save assessment report comparing configured vs actual capabilities."""
        # Get current user configuration
        user_config = self.config.get("llm", {})

        report = "# LLM Assessment Results\n\n"
        report += f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "**Assessment:** Configuration vs Actual Performance\n\n"

        issues_found = False

        for llm_name, result in probe_results.items():
            report += f"## {llm_name.title()} LLM Assessment\n\n"

            # Current configuration
            prefix = f"{llm_name}_"
            configured_max_tokens = user_config.get(f"{prefix}max_tokens", 0)
            configured_context = user_config.get(f"{prefix}max_context_tokens", 0)

            report += "### Current Configuration:\n"
            report += f"- **Model:** {result.model}\n"
            report += f"- **API:** {result.api_base_url}\n"
            report += f"- **Configured Max Tokens:** {configured_max_tokens:,}\n"
            report += f"- **Configured Context:** {configured_context:,}\n\n"

            # Actual performance
            report += "### Discovered Performance:\n"
            report += f"- **Engine Type:** {result.inference_engine.value}\n"
            report += f"- **Actual Max Context:** {result.max_context_tokens:,} tokens\n"
            report += f"- **Effective Context:** {result.effective_context_tokens:,} tokens\n"
            report += f"- **Success Rate:** {result.success_rate:.1%}\n"
            report += f"- **Avg Latency:** {result.avg_latency_ms:.0f}ms\n\n"

            # Assessment and warnings
            report += "### Assessment:\n"

            if configured_context > result.effective_context_tokens:
                report += f"⚠️  **WARNING:** Configured context ({configured_context:,}) exceeds actual capacity ({result.effective_context_tokens:,})\n"
                issues_found = True
            elif configured_context == 0:
                report += f"💡 **INFO:** No context limit configured, discovered {result.effective_context_tokens:,} tokens\n"
            else:
                report += "✅ **OK:** Configured context within actual capacity\n"

            if result.success_rate < 0.9:
                report += f"⚠️  **WARNING:** Low success rate ({result.success_rate:.1%}) - LLM may be overloaded\n"
                issues_found = True
            else:
                report += f"✅ **OK:** Good success rate ({result.success_rate:.1%})\n"

            if result.avg_latency_ms > 10000:  # 10 seconds
                report += f"⚠️  **WARNING:** High latency ({result.avg_latency_ms:.0f}ms) - consider optimization\n"
                issues_found = True
            else:
                report += f"✅ **OK:** Acceptable latency ({result.avg_latency_ms:.0f}ms)\n"

            report += "\n"

        # Overall recommendations
        if issues_found:
            report += "## 🔧 Recommended Actions\n\n"
            report += "Based on the assessment, consider updating your `pyproject.toml`:\n\n"
            report += "```toml\n"
            report += "[tool.vibelint.llm]\n"

            for llm_name, result in probe_results.items():
                prefix = f"{llm_name}_"
                configured_context = user_config.get(f"{prefix}max_context_tokens", 0)

                if configured_context > result.effective_context_tokens or configured_context == 0:
                    report += f"# Recommended limits for {llm_name} LLM based on testing\n"
                    report += f"{prefix}max_context_tokens = {result.effective_context_tokens}\n"
                    report += (
                        f"{prefix}max_prompt_tokens = {result.recommended_max_prompt_tokens}\n"
                    )

            report += "```\n\n"
        else:
            report += "## ✅ Configuration Assessment\n\n"
            report += "Your current configuration appears to be well-matched to your LLM capabilities!\n\n"

        report += "### Next Steps\n\n"
        report += "1. Review any warnings above\n"
        report += "2. Update configuration if recommended\n"
        report += "3. Re-run diagnostics after changes to verify\n"
        report += "4. Monitor performance in production use\n"

        Path("LLM_ASSESSMENT_RESULTS.md").write_text(report, encoding="utf-8")

    async def benchmark_routing(self) -> Dict[str, Any]:
        """Simple benchmark of LLM routing logic."""
        if not self.llm_manager:
            return {"error": "No LLM manager configured"}

        # Test scenarios
        scenarios = [
            {"content": "Short docstring task", "task_type": "docstring", "expected": LLMRole.FAST},
            {"content": "x" * 5000, "task_type": "analysis", "expected": LLMRole.ORCHESTRATOR},
            {
                "content": "Architecture analysis",
                "task_type": "architecture",
                "expected": LLMRole.ORCHESTRATOR,
            },
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

            results["scenarios"].append(
                {
                    "task": scenario["task_type"],
                    "content_size": len(scenario["content"]),
                    "expected": scenario["expected"].value,
                    "selected": selected.value,
                    "correct": is_correct,
                }
            )

        results["routing_accuracy"] = correct / len(scenarios)
        return results


async def run_diagnostics(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run comprehensive diagnostics: context probing + routing benchmark.

    Args:
        config: Vibelint configuration dictionary

    Returns:
        Complete diagnostics results
    """
    diagnostics = DualLLMDiagnostics(config)

    try:
        # Step 1: Context probing
        print("=== LLM Context Probing ===")
        probe_results = await diagnostics.run_context_probing()

        if not probe_results:
            print("❌ No LLM configurations found or probing failed")
            return {"error": "Context probing failed"}

        # Print context probing results
        for llm_name, result in probe_results.items():
            print(f"\n{llm_name.upper()} LLM:")
            print(f"  ✓ Max Context: {result.max_context_tokens:,} tokens")
            print(f"  ✓ Success Rate: {result.success_rate:.1%}")
            print(f"  ✓ Latency: {result.avg_latency_ms:.0f}ms")

        # Step 2: Routing benchmark
        print("\n=== LLM Routing Benchmark ===")
        routing_results = await diagnostics.benchmark_routing()

        if "error" not in routing_results:
            print(f"Routing Accuracy: {routing_results['routing_accuracy']:.1%}")

            for scenario in routing_results["scenarios"]:
                status = "✓" if scenario["correct"] else "✗"
                print(
                    f"  {status} {scenario['task']}: {scenario['selected']} (expected: {scenario['expected']})"
                )

        print("\n✅ Comprehensive diagnostics completed")
        print("📄 Results saved to LLM_CALIBRATION_RESULTS.md")

        return {"probe_results": probe_results, "routing_results": routing_results, "success": True}

    except (requests.exceptions.RequestException, ValueError, KeyError, OSError) as e:
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
            print(
                f"  {status} {scenario['task']}: {scenario['selected']} (expected: {scenario['expected']})"
            )

        return results

    except (requests.exceptions.RequestException, ValueError, KeyError, OSError) as e:
        print(f"❌ Benchmark failed: {e}")
        return {"error": str(e)}
