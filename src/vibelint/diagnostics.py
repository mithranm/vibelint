"""
Vibelint diagnostics and benchmarking utilities.

This module provides diagnostic tools for testing vibelint's performance,
LLM integration, and analysis quality.

tools/vibelint/src/vibelint/diagnostics.py
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

__all__ = ["LLMBenchmark", "VibelintDiagnostics"]


class LLMBenchmark:
    """Benchmark vibelint's LLM performance and generation parameters."""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "vibelint_benchmark"
        self.temp_dir.mkdir(exist_ok=True)

    def create_test_files(self) -> List[Tuple[str, Path, int]]:
        """Create test files of various sizes for benchmarking."""
        test_files = []

        # Small file (~500 chars)
        small_content = '''
def small_function(x: int) -> str:
    """Small test function."""
    return f"result: {x}"

class TestClass:
    def method(self):
        pass
'''
        small_file = self.temp_dir / "small_test.py"
        small_file.write_text(small_content)
        test_files.append(("small", small_file, len(small_content)))

        # Medium file (~2000 chars)
        medium_content = (
            small_content * 4
            + '''
def medium_function(data: Dict[str, Any]) -> List[str]:
    """Process data and return results."""
    results = []
    for key, value in data.items():
        if isinstance(value, str):
            results.append(f"{key}: {value}")
    return results
'''
        )
        medium_file = self.temp_dir / "medium_test.py"
        medium_file.write_text(medium_content)
        test_files.append(("medium", medium_file, len(medium_content)))

        return test_files

    def benchmark_llm_speed(self, categories: List[str] = None) -> Dict[str, Any]:
        """Benchmark LLM analysis speed across different file sizes."""
        if categories is None:
            categories = ["ai"]

        test_files = self.create_test_files()
        results = {"benchmark_results": [], "summary": {}}

        for size_name, file_path, char_count in test_files:
            print(f"Benchmarking {size_name} file ({char_count} chars)...")

            start_time = time.time()
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "vibelint",
                    "check",
                    str(file_path),
                    "--categories",
                    ",".join(categories),
                    "--format",
                    "json",
                ]

                env = {"PYTHONPATH": "tools/vibelint/src"}
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)

                end_time = time.time()
                duration = end_time - start_time

                benchmark_result = {
                    "file_size": size_name,
                    "char_count": char_count,
                    "duration_seconds": duration,
                    "success": result.returncode == 0,
                    "stdout_length": len(result.stdout),
                    "stderr_length": len(result.stderr),
                }

                results["benchmark_results"].append(benchmark_result)
                print(f"  Duration: {duration:.2f}s")

            except subprocess.TimeoutExpired:
                print("  TIMEOUT after 120s")
                results["benchmark_results"].append(
                    {
                        "file_size": size_name,
                        "char_count": char_count,
                        "duration_seconds": 120,
                        "success": False,
                        "error": "timeout",
                    }
                )

        # Calculate summary stats
        successful_runs = [r for r in results["benchmark_results"] if r["success"]]
        if successful_runs:
            durations = [r["duration_seconds"] for r in successful_runs]
            results["summary"] = {
                "total_runs": len(results["benchmark_results"]),
                "successful_runs": len(successful_runs),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
            }

        return results

    def cleanup(self):
        """Clean up temporary test files."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class VibelintDiagnostics:
    """Diagnostic tools for testing vibelint functionality."""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "vibelint_diagnostics"
        self.temp_dir.mkdir(exist_ok=True)

    def create_test_file(self, content: str, name: str) -> Path:
        """Create a temporary test file with given content."""
        test_file = self.temp_dir / f"{name}.py"
        test_file.write_text(content)
        return test_file

    def test_ai_analysis(self, test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Test AI analysis on various code patterns."""
        results = {"test_results": [], "summary": {}}

        for test_name, test_content in test_cases:
            print(f"Testing AI analysis: {test_name}")

            test_file = self.create_test_file(test_content, test_name)

            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "vibelint",
                    "check",
                    str(test_file),
                    "--categories",
                    "ai",
                    "--format",
                    "json",
                ]

                env = {"PYTHONPATH": "tools/vibelint/src"}
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)

                test_result = {
                    "test_name": test_name,
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "has_findings": "findings" in result.stdout.lower(),
                }

                results["test_results"].append(test_result)
                print(f"  Result: {'PASS' if test_result['success'] else 'FAIL'}")

            except subprocess.TimeoutExpired:
                print("  TIMEOUT")
                results["test_results"].append(
                    {"test_name": test_name, "success": False, "error": "timeout"}
                )

        # Summary
        successful_tests = [r for r in results["test_results"] if r["success"]]
        results["summary"] = {
            "total_tests": len(results["test_results"]),
            "successful_tests": len(successful_tests),
            "success_rate": (
                len(successful_tests) / len(results["test_results"])
                if results["test_results"]
                else 0
            ),
        }

        return results

    def cleanup(self):
        """Clean up temporary test files."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def run_benchmark():
    """Run LLM benchmark and print results."""
    benchmark = LLMBenchmark()
    try:
        results = benchmark.benchmark_llm_speed()

        print("\n=== LLM Benchmark Results ===")
        for result in results["benchmark_results"]:
            print(
                f"{result['file_size']}: {result['duration_seconds']:.2f}s ({'SUCCESS' if result['success'] else 'FAILED'})"
            )

        if results["summary"]:
            print(
                f"\nSummary: {results['summary']['successful_runs']}/{results['summary']['total_runs']} successful"
            )
            print(f"Average duration: {results['summary']['avg_duration']:.2f}s")

        return results
    finally:
        benchmark.cleanup()


def run_diagnostics():
    """Run diagnostic tests and print results."""
    diagnostics = VibelintDiagnostics()

    test_cases = [
        (
            "simple_function",
            """
def test_function():
    pass
""",
        ),
        (
            "complex_logic",
            """
def complex_function(data):
    if not data:
        return None

    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)

    return result
""",
        ),
    ]

    try:
        results = diagnostics.test_ai_analysis(test_cases)

        print("\n=== Diagnostic Test Results ===")
        for result in results["test_results"]:
            print(f"{result['test_name']}: {'PASS' if result['success'] else 'FAIL'}")

        print(f"\nSuccess rate: {results['summary']['success_rate']:.1%}")

        return results
    finally:
        diagnostics.cleanup()


if __name__ == "__main__":
    print("Running vibelint diagnostics...")
    run_benchmark()
    run_diagnostics()
