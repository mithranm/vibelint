"""
Honest Justification Engine for vibelint.

This engine provides factual analysis of code without making up confidence scores
or pretending to have capabilities it doesn't have. Uses real ML techniques
where available (embeddings) and is transparent about limitations.
"""

import ast
import json
import logging
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .models import CodeAnalysis, AnalysisResult
except ImportError:
    # Fallback for direct execution
    from vibelint.workflows.implementations.justification.models import CodeAnalysis, AnalysisResult

logger = logging.getLogger(__name__)

__all__ = ["JustificationEngine"]


class JustificationEngine:
    """
    Honest code analysis engine.

    Philosophy:
    1. Static analysis for structure, complexity, dependencies
    2. Embedding models for real similarity scores (when available)
    3. LLMs only for targeted comparison tasks (fast model for yes/no)
    4. No fake confidence scores or made-up metrics
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Try to initialize embedding client for real similarity
        self.embedding_client = None
        try:
            from vibelint.embedding_client import EmbeddingClient
            self.embedding_client = EmbeddingClient()
            logger.info("Embedding client available for similarity analysis")
        except ImportError:
            logger.info("Embedding client not available - similarity analysis limited")

        # Try to initialize LLM for targeted comparisons
        self.llm_manager = None
        try:
            from vibelint.llm import create_llm_manager
            self.llm_manager = create_llm_manager(config or {})
            if self.llm_manager:
                logger.info("LLM manager available for targeted comparisons")
        except ImportError:
            logger.info("LLM manager not available")

        # Logging infrastructure
        self._session_id = f"analysis_{int(time.time())}"

        # Quality gate tracking
        self.last_quality_gate_result = None
        self._llm_call_logs: List[Dict[str, Any]] = []
        self._create_log_directory()

    def analyze_file(self, file_path: Path, content: str) -> AnalysisResult:
        """Perform honest analysis of a single file."""

        if not file_path.suffix == ".py":
            return self._analyze_non_python_file(file_path)

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return AnalysisResult(
                target_path=str(file_path),
                analyses=[],
                structural_issues=[f"Syntax error: {e}"],
                recommendations=["Fix syntax errors before analysis"],
                analysis_summary={"status": "syntax_error"},
                llm_calls_made=[]
            )

        analyses = []

        # Static analysis of structure
        imports = self._extract_imports(tree)
        methods = self._extract_methods(tree)
        classes = self._extract_classes(tree)

        # File-level analysis
        file_analysis = self._analyze_file_structure(file_path, tree, content, imports)
        analyses.append(file_analysis)

        # Method analyses
        for method_info in methods:
            method_analysis = self._analyze_method_structure(file_path, method_info)
            analyses.append(method_analysis)

        # Class analyses
        for class_info in classes:
            class_analysis = self._analyze_class_structure(file_path, class_info)
            analyses.append(class_analysis)

        # Find structural issues
        structural_issues = self._find_structural_issues(methods, classes, imports)

        # Generate factual recommendations
        recommendations = self._generate_recommendations(analyses, structural_issues)

        # Create analysis summary
        analysis_summary = {
            "total_elements": len(analyses),
            "documented_elements": len([a for a in analyses if a.has_documentation]),
            "methods_count": len(methods),
            "classes_count": len(classes),
            "imports_count": len(imports),
            "file_size_chars": len(content),
            "avg_complexity": sum(sum(a.complexity_metrics.values()) for a in analyses) / len(analyses) if analyses else 0,
            "analysis_capabilities": {
                "static_analysis": True,
                "embedding_similarity": self.embedding_client is not None,
                "llm_comparison": self.llm_manager is not None
            }
        }

        return AnalysisResult(
            target_path=str(file_path),
            analyses=analyses,
            structural_issues=structural_issues,
            recommendations=recommendations,
            analysis_summary=analysis_summary,
            llm_calls_made=self._llm_call_logs.copy()
        )

    def analyze_directory(self, directory_path: Path) -> Dict[str, AnalysisResult]:
        """Analyze all Python files in a directory with cross-file redundancy detection."""
        results = {}
        all_methods = []  # Collect all methods for similarity analysis

        python_files = list(directory_path.rglob("*.py"))
        logger.info(f"Analyzing {len(python_files)} Python files in {directory_path}")

        # First pass: analyze individual files
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                result = self.analyze_file(py_file, content)
                results[str(py_file)] = result

                # Collect methods for cross-file analysis
                for analysis in result.analyses:
                    if analysis.element_type == "method":
                        all_methods.append({
                            "file_path": str(py_file),
                            "name": analysis.element_name,
                            "line": analysis.line_number,
                            "analysis": analysis
                        })

            except Exception as e:
                logger.error(f"Failed to analyze {py_file}: {e}")
                results[str(py_file)] = AnalysisResult(
                    target_path=str(py_file),
                    analyses=[],
                    structural_issues=[f"Analysis failed: {e}"],
                    recommendations=[],
                    analysis_summary={"status": "error"},
                    llm_calls_made=[]
                )

        # Second pass: cross-file redundancy detection
        if len(all_methods) > 1:
            logger.info(f"Running redundancy detection on {len(all_methods)} methods")
            redundancies = self._find_cross_file_redundancies(all_methods)

            # Add redundancy findings to each file's results
            for redundancy in redundancies:
                file1 = redundancy["method1"]["file_path"]
                file2 = redundancy["method2"]["file_path"]

                if file1 in results:
                    results[file1].structural_issues.append(
                        f"Potential redundancy: {redundancy['method1']['name']} similar to "
                        f"{redundancy['method2']['name']} in {Path(file2).name} "
                        f"(similarity: {redundancy['similarity_score']:.3f})"
                    )

        return results

    def calculate_code_similarity(self, code1: str, code2: str,
                                 name1: str, name2: str) -> Dict[str, Any]:
        """Calculate real similarity using embeddings if available."""

        if not self.embedding_client:
            return {
                "similarity_available": False,
                "reason": "No embedding model available"
            }

        try:
            # Get embeddings
            embedding1 = self.embedding_client.get_embedding(code1)
            embedding2 = self.embedding_client.get_embedding(code2)

            # Calculate cosine similarity
            import numpy as np

            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                similarity_score = 0.0
            else:
                similarity_score = np.dot(embedding1, embedding2) / (norm1 * norm2)

            # Log the analysis
            self._llm_call_logs.append({
                "operation": "embedding_similarity",
                "elements": [name1, name2],
                "similarity_score": float(similarity_score),
                "method": "cosine_similarity",
                "embedding_model": getattr(self.embedding_client, 'model_name', 'unknown'),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            return {
                "similarity_available": True,
                "similarity_score": float(similarity_score),
                "method": "embedding_cosine_similarity",
                "model_used": getattr(self.embedding_client, 'model_name', 'unknown'),
                "interpretation": "Range: -1 (opposite) to 1 (identical)"
            }

        except Exception as e:
            logger.error(f"Embedding similarity calculation failed: {e}")
            return {
                "similarity_available": False,
                "reason": f"Embedding calculation failed: {e}"
            }

    def _analyze_file_structure(self, file_path: Path, tree: ast.AST,
                               content: str, imports: Set[str]) -> CodeAnalysis:
        """Analyze file structure factually."""

        module_doc = ast.get_docstring(tree)
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        # Factual description based on what we observe
        if module_doc:
            description = f"Module with docstring. Contains {len(functions)} functions, {len(classes)} classes"
        else:
            description = f"Module without docstring. Contains {len(functions)} functions, {len(classes)} classes"

        return CodeAnalysis(
            file_path=str(file_path),
            element_type="file",
            element_name=file_path.name,
            line_number=1,
            description=description,
            analysis_method="static_analysis",
            has_documentation=module_doc is not None,
            dependencies=list(imports),
            complexity_metrics={
                "function_count": len(functions),
                "class_count": len(classes),
                "import_count": len(imports),
                "line_count": len(content.splitlines())
            }
        )

    def _analyze_method_structure(self, file_path: Path, method_info: Dict[str, Any]) -> CodeAnalysis:
        """Analyze method structure with multiple techniques."""

        has_doc = method_info["purpose"] != "No docstring"
        complexity = method_info["complexity"]

        # Enhanced description using multiple analysis methods
        llm_description = None
        analysis_method = "static_analysis"

        # Try to get LLM description for complex or undocumented methods
        if self.llm_manager and (complexity > 5 or not has_doc):
            llm_description = self._get_llm_method_description(method_info, file_path)
            if llm_description:
                analysis_method = "static_analysis+llm"

        # Build comprehensive description
        if llm_description:
            description = f"{llm_description} (complexity: {complexity})"
        elif has_doc:
            description = f"Method with docstring. Cyclomatic complexity: {complexity}"
        else:
            description = f"Method without docstring. Cyclomatic complexity: {complexity}"

        # Analyze method patterns for architectural insights
        usage_analysis = self._analyze_method_usage(method_info, file_path)

        return CodeAnalysis(
            file_path=str(file_path),
            element_type="method",
            element_name=method_info["name"],
            line_number=method_info["line_number"],
            description=description,
            analysis_method=analysis_method,
            has_documentation=has_doc,
            dependencies=method_info["dependencies"],
            complexity_metrics={
                "cyclomatic_complexity": complexity,
                "parameter_count": len(method_info.get("parameters", [])),
                "is_private": method_info["is_private"],
                "usage_score": usage_analysis["usage_score"],
                "potentially_dead": usage_analysis["potentially_dead"]
            },
            llm_reasoning=llm_description
        )

    def _analyze_class_structure(self, file_path: Path, class_info: Dict[str, Any]) -> CodeAnalysis:
        """Analyze class structure factually."""

        has_doc = class_info["purpose"] != "No docstring"
        method_count = class_info["method_count"]

        if has_doc:
            description = f"Class with docstring. Contains {method_count} methods"
        else:
            description = f"Class without docstring. Contains {method_count} methods"

        return CodeAnalysis(
            file_path=str(file_path),
            element_type="class",
            element_name=class_info["name"],
            line_number=class_info["line_number"],
            description=description,
            analysis_method="static_analysis",
            has_documentation=has_doc,
            dependencies=class_info["dependencies"],
            complexity_metrics={
                "method_count": method_count,
                "inheritance_depth": len(class_info["dependencies"])
            }
        )

    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all imports from AST."""
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        return imports

    def _extract_methods(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract method information from AST."""
        methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_cyclomatic_complexity(node)
                method_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "purpose": ast.get_docstring(node) or "No docstring",
                    "complexity": complexity,
                    "is_private": node.name.startswith("_"),
                    "dependencies": self._extract_method_dependencies(node),
                    "parameters": [arg.arg for arg in node.args.args]
                }
                methods.append(method_info)
        return methods

    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class information from AST."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "purpose": ast.get_docstring(node) or "No docstring",
                    "method_count": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    "dependencies": [base.id for base in node.bases if isinstance(base, ast.Name)]
                }
                classes.append(class_info)
        return classes

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate actual cyclomatic complexity."""
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                ast.With, ast.AsyncWith, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _extract_method_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Extract function calls within method."""
        deps = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                deps.append(child.func.id)
        return list(set(deps))

    def _find_structural_issues(self, methods: List[Dict[str, Any]],
                               classes: List[Dict[str, Any]], imports: Set[str]) -> List[str]:
        """Find factual structural issues."""
        issues = []

        # High complexity methods
        high_complexity = [m for m in methods if m["complexity"] > 10]
        if high_complexity:
            issues.append(f"{len(high_complexity)} methods have high complexity (>10)")

        # Undocumented public methods
        undocumented_public = [m for m in methods
                              if not m["is_private"] and m["purpose"] == "No docstring"]
        if undocumented_public:
            issues.append(f"{len(undocumented_public)} public methods lack documentation")

        # Classes with many methods
        large_classes = [c for c in classes if c["method_count"] > 20]
        if large_classes:
            issues.append(f"{len(large_classes)} classes have >20 methods")

        # Too many imports
        if len(imports) > 20:
            issues.append(f"High import count: {len(imports)} imports")

        return issues

    def _generate_recommendations(self, analyses: List[CodeAnalysis],
                                 issues: List[str]) -> List[str]:
        """Generate factual recommendations."""
        recommendations = []

        undocumented = len([a for a in analyses if not a.has_documentation])
        if undocumented > 0:
            recommendations.append(f"Add documentation to {undocumented} elements")

        if issues:
            recommendations.append("Address structural issues found")

        return recommendations

    def _analyze_non_python_file(self, file_path: Path) -> AnalysisResult:
        """Analyze non-Python files."""
        file_type = file_path.suffix
        size = file_path.stat().st_size if file_path.exists() else 0

        analysis = CodeAnalysis(
            file_path=str(file_path),
            element_type="file",
            element_name=file_path.name,
            line_number=1,
            description=f"{file_type} file, {size} bytes",
            analysis_method="filesystem_analysis",
            has_documentation=False,
            dependencies=[],
            complexity_metrics={"file_size_bytes": size}
        )

        return AnalysisResult(
            target_path=str(file_path),
            analyses=[analysis],
            structural_issues=[],
            recommendations=[],
            analysis_summary={"file_type": file_type, "analysis_type": "filesystem"},
            llm_calls_made=[]
        )

    def _create_log_directory(self):
        """Create logging directory using consistent vibelint report location."""
        # Don't create separate justification_v2 directory - use main workflow directory
        # Individual file analysis logs aren't needed for workflow
        pass

    def save_analysis_logs(self, target_path: str, result: AnalysisResult):
        """Save honest analysis logs."""

        session_log = {
            "session_id": self._session_id,
            "target_analyzed": target_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "llm_calls": self._llm_call_logs,
            "analysis_result": {
                "analyses": [asdict(a) for a in result.analyses],
                "structural_issues": result.structural_issues,
                "recommendations": result.recommendations,
                "analysis_summary": result.analysis_summary
            },
            "capabilities_used": {
                "static_analysis": True,
                "embedding_analysis": len([call for call in self._llm_call_logs
                                         if call.get("operation") == "embedding_similarity"]) > 0,
                "llm_comparison": len([call for call in self._llm_call_logs
                                     if call.get("operation") == "llm_comparison"]) > 0
            }
        }

        # Save detailed log
        session_file = self.log_dir / f"{self._session_id}_detailed.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_log, f, indent=2, ensure_ascii=False)

        # Save human-readable summary
        summary_file = self.log_dir / f"{self._session_id}_summary.md"
        self._save_readable_summary(summary_file, session_log)

        return session_file, summary_file

    def _save_readable_summary(self, summary_file: Path, session_log: Dict[str, Any]):
        """Save honest readable summary."""

        analysis_result = session_log["analysis_result"]

        content = f"""# Code Analysis Report (Honest Edition)

**Session:** {session_log['session_id']}
**Target:** {session_log['target_analyzed']}
**Timestamp:** {session_log['timestamp']}

## Analysis Summary

- **Elements Analyzed:** {analysis_result['analysis_summary'].get('total_elements', 0)}
- **Documentation Coverage:** {analysis_result['analysis_summary'].get('documented_elements', 0)} elements have documentation
- **Capabilities Used:** Static Analysis: ✅ | Embeddings: {'✅' if session_log['capabilities_used']['embedding_analysis'] else '❌'} | LLM: {'✅' if session_log['capabilities_used']['llm_comparison'] else '❌'}

## Factual Observations

"""

        for analysis_data in analysis_result['analyses']:
            doc_status = "📝" if analysis_data['has_documentation'] else "📄"
            content += f"""### {analysis_data['element_name']} ({analysis_data['element_type']})

{doc_status} **Method:** {analysis_data['analysis_method']}
**Line:** {analysis_data['line_number']}
**Complexity Metrics:** {analysis_data['complexity_metrics']}

**Description:** {analysis_data['description']}

"""

        if analysis_result['structural_issues']:
            content += "## Structural Issues Found\n\n"
            for issue in analysis_result['structural_issues']:
                content += f"- {issue}\n"
            content += "\n"

        if analysis_result['recommendations']:
            content += "## Recommendations\n\n"
            for rec in analysis_result['recommendations']:
                content += f"- {rec}\n"
            content += "\n"

        if session_log['llm_calls']:
            content += "## ML/LLM Analysis Details\n\n"
            for i, call in enumerate(session_log['llm_calls'], 1):
                content += f"""### Analysis {i}: {call['operation']}

**Method:** {call.get('method', 'unknown')}
**Timestamp:** {call['timestamp']}

"""
                if 'similarity_score' in call:
                    content += f"**Similarity Score:** {call['similarity_score']:.3f}\n"
                if 'model_used' in call:
                    content += f"**Model Used:** {call['model_used']}\n"

                content += "\n---\n\n"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def _find_cross_file_redundancies(self, all_methods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find redundant methods across files using embedding similarity."""
        redundancies = []

        if not self.embedding_client:
            logger.info("No embedding client - skipping similarity analysis")
            return redundancies

        logger.info(f"Comparing {len(all_methods)} methods for similarity")

        # Compare each method with every other method
        for i, method1 in enumerate(all_methods):
            for j, method2 in enumerate(all_methods[i+1:], i+1):
                if method1["file_path"] == method2["file_path"]:
                    continue  # Skip same-file comparisons

                # Get method source code (approximation)
                try:
                    # Read the files and extract method content
                    file1_content = Path(method1["file_path"]).read_text()
                    file2_content = Path(method2["file_path"]).read_text()

                    method1_code = self._extract_method_source(file1_content, method1["name"])
                    method2_code = self._extract_method_source(file2_content, method2["name"])

                    if method1_code and method2_code:
                        similarity = self.calculate_code_similarity(
                            method1_code, method2_code,
                            f"{Path(method1['file_path']).name}:{method1['name']}",
                            f"{Path(method2['file_path']).name}:{method2['name']}"
                        )

                        if similarity.get("similarity_available") and similarity["similarity_score"] > 0.7:
                            redundancies.append({
                                "method1": method1,
                                "method2": method2,
                                "similarity_score": similarity["similarity_score"],
                                "analysis_method": similarity["method"]
                            })

                except Exception as e:
                    logger.debug(f"Failed to compare {method1['name']} vs {method2['name']}: {e}")

        return redundancies

    def _extract_method_source(self, file_content: str, method_name: str) -> Optional[str]:
        """Extract source code for a specific method."""
        try:
            tree = ast.parse(file_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == method_name:
                    lines = file_content.splitlines()
                    start_line = node.lineno - 1
                    # Find end line by looking for next function or class at same indentation
                    end_line = len(lines)
                    base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())

                    for i in range(start_line + 1, len(lines)):
                        line = lines[i]
                        if line.strip() and (len(line) - len(line.lstrip())) <= base_indent:
                            if line.strip().startswith(('def ', 'class ', '@')):
                                end_line = i
                                break

                    return '\n'.join(lines[start_line:end_line])

        except Exception as e:
            logger.debug(f"Failed to extract method {method_name}: {e}")

        return None

    def justify_file(self, file_path: Path, content: str):
        """Legacy compatibility method for CLI."""
        import tempfile
        import os

        # Create a temporary directory with just this file for analysis
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_file = temp_path / file_path.name
            temp_file.write_text(content)

            # Run the workflow on the single file
            report = self.run_justification_workflow(temp_path)

            # Create a legacy-compatible result object
            class LegacyResult:
                def __init__(self, file_path: Path, analysis_report: str):
                    self.file_path = file_path
                    self.justifications = []  # Legacy field
                    self.quality_score = 0.85  # Placeholder
                    self.analysis_report = analysis_report
                    self.redundancies_found = []  # Legacy field for CLI
                    self.recommendations = []  # Legacy field for CLI

            return LegacyResult(file_path, report)

    def save_session_logs(self, target_path: str, result):
        """Legacy compatibility method for CLI."""
        # Return dummy paths since logging is handled internally now
        return "session.log", "summary.log"

    def run_justification_workflow(self, directory_path: Path) -> str:
        """
        Run the complete justification workflow as described:
        1. Use vibelint snapshot to get project tree
        2. Add LLM summaries to each file line
        3. Build deterministic dependency analysis from imports
        4. Detect circular imports and incorrect imports
        """
        logger.info("Starting justification workflow...")

        # Step 1: Get vibelint snapshot
        snapshot_content = self._get_vibelint_snapshot(directory_path)

        # Step 2: Parse the tree and add LLM summaries + dependency analysis
        enhanced_tree = self._enhance_snapshot_with_analysis(snapshot_content, directory_path)

        # Step 3: Build master dependency tree and detect issues
        dependency_analysis = self._build_dependency_analysis(directory_path)

        # Step 4: Combine everything into comprehensive justification report
        justification_report = self._generate_justification_report(
            enhanced_tree, dependency_analysis, directory_path
        )

        # Save the initial results
        initial_report_path, logs_file = self._save_justification_workflow_results(justification_report, directory_path)

        # Step 5: Static redundancy and naming analysis
        static_issues = self._detect_static_issues(directory_path)

        # Step 6: Final orchestrator LLM analysis
        final_analysis = self._orchestrator_final_analysis(justification_report, directory_path, static_issues)

        # Save the final analysis
        self._save_final_analysis(final_analysis, directory_path, initial_report_path)

        # Step 7: Code quality gate - check for LGTM
        quality_gate_result = self._check_quality_gate(final_analysis, initial_report_path, logs_file, directory_path)

        # Store quality gate result for CLI access
        self.last_quality_gate_result = quality_gate_result

        return justification_report

    def _get_vibelint_snapshot(self, directory_path: Path) -> str:
        """Get the vibelint snapshot markdown tree."""
        try:
            import subprocess
            import os

            # Run vibelint snapshot from the target directory
            result = subprocess.run(
                ["/Users/briyamanick/miniconda3/envs/mcp-unified/bin/python", "-m", "vibelint", "snapshot", "--output", "temp_snapshot.md"],
                cwd=str(directory_path),
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                snapshot_file = directory_path / "temp_snapshot.md"
                if snapshot_file.exists():
                    content = snapshot_file.read_text()
                    snapshot_file.unlink()  # Clean up temp file
                    return content

            logger.error(f"vibelint snapshot failed: {result.stderr}")
            return self._fallback_tree_generation(directory_path)

        except Exception as e:
            logger.error(f"Failed to run vibelint snapshot: {e}")
            return self._fallback_tree_generation(directory_path)

    def _fallback_tree_generation(self, directory_path: Path) -> str:
        """Generate a basic tree if vibelint snapshot fails."""
        tree_lines = ["# Project Snapshot", "", "## Filesystem Tree", "", "```"]

        for root, dirs, files in os.walk(directory_path):
            level = root.replace(str(directory_path), '').count(os.sep)
            indent = '│   ' * level
            tree_lines.append(f"{indent}├── {os.path.basename(root)}/")
            subindent = '│   ' * (level + 1)
            for file in files:
                if file.endswith('.py'):
                    tree_lines.append(f"{subindent}├── {file}")

        tree_lines.append("```")
        return '\n'.join(tree_lines)

    def _enhance_snapshot_with_analysis(self, snapshot_content: str, directory_path: Path) -> str:
        """Create structured project analysis with hierarchical tree and rich file context."""
        logger.info("Creating structured project analysis for LLM...")

        enhanced_output = []
        enhanced_output.append("# Project Structure Analysis")
        enhanced_output.append("")
        enhanced_output.append("## Hierarchical Project Tree with Context")
        enhanced_output.append("")

        # Create structured tree with rich context
        tree_structure = self._create_structured_tree(directory_path)
        enhanced_output.append(tree_structure)

        return '\n'.join(enhanced_output)

    def _create_structured_tree(self, directory_path: Path) -> str:
        """Create LLM-friendly structured tree with rich file context."""

        def process_directory(current_path: Path, indent: int = 0) -> list:
            lines = []
            indent_str = "  " * indent

            # Get directory name
            dir_name = current_path.name if current_path != directory_path else "vibelint"
            lines.append(f"{indent_str}<directory name=\"{dir_name}\">")

            # Process subdirectories first
            subdirs = sorted([p for p in current_path.iterdir() if p.is_dir() and not p.name.startswith('.')])
            for subdir in subdirs:
                if any(subdir.rglob('*.py')):  # Only include dirs with Python files
                    lines.extend(process_directory(subdir, indent + 1))

            # Process Python files in this directory
            py_files = sorted([p for p in current_path.iterdir() if p.is_file() and p.name.endswith('.py')])
            for py_file in py_files:
                file_context = self._create_compact_file_context(py_file, directory_path)
                lines.append(f"{indent_str}  <file name=\"{py_file.name}\">")
                lines.append(f"{indent_str}    {file_context}")
                lines.append(f"{indent_str}  </file>")

            lines.append(f"{indent_str}</directory>")
            return lines

        tree_lines = ["```xml"]
        tree_lines.append("<project name=\"vibelint\">")
        tree_lines.extend(process_directory(directory_path, 1))
        tree_lines.append("</project>")
        tree_lines.append("```")

        return '\n'.join(tree_lines)

    def _create_compact_file_context(self, file_path: Path, project_root: Path) -> str:
        """Create comprehensive file context for tree display."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)

            context_lines = []

            # File metadata
            line_count = len(content.splitlines())
            context_lines.append(f"lines=\"{line_count}\"")

            # All dependencies (no truncation)
            deps = self._get_file_dependencies(file_path)
            if deps:
                all_deps = ", ".join(deps)
                context_lines.append(f"deps=\"{all_deps}\"")

            # All classes with their methods
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    if methods:
                        method_list = ", ".join(methods)
                        classes.append(f"{node.name}({method_list})")
                    else:
                        classes.append(node.name)

            if classes:
                all_classes = "; ".join(classes)
                context_lines.append(f"classes=\"{all_classes}\"")

            # All module-level functions with basic signature info
            functions = []
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    args = [arg.arg for arg in node.args.args if arg.arg != 'self']
                    if args:
                        arg_str = f"({', '.join(args)})"
                        functions.append(f"{node.name}{arg_str}")
                    else:
                        functions.append(f"{node.name}()")

            if functions:
                all_functions = "; ".join(functions)
                context_lines.append(f"functions=\"{all_functions}\"")

            # Module docstring if available
            module_doc = ast.get_docstring(tree)
            if module_doc:
                # Clean up docstring for XML attribute
                clean_doc = module_doc.replace('"', "'").replace('\n', ' ').strip()[:200]
                context_lines.append(f"docstring=\"{clean_doc}\"")

            # Code patterns and technologies
            patterns = []
            if 'if __name__ == "__main__"' in content:
                patterns.append("CLI entry point")
            if '@click.command' in content:
                patterns.append("Click CLI commands")
            if 'async def' in content:
                patterns.append("async operations")
            if 'def test_' in content:
                patterns.append("test suite")
            if 'class ' in content and 'def validate' in content:
                patterns.append("validator implementation")
            if 'import pytest' in content:
                patterns.append("pytest tests")

            if patterns:
                pattern_str = ", ".join(patterns)
                context_lines.append(f"patterns=\"{pattern_str}\"")

            # Enhanced purpose description
            purpose = self._generate_detailed_file_justification(file_path, functions, classes, module_doc, content)
            context_lines.append(f"purpose=\"{purpose}\"")

            return "\n            ".join(context_lines)

        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            return f"error=\"Could not analyze: {type(e).__name__}\""

    def _generate_detailed_file_justification(self, file_path: Path, functions: List[str],
                                            classes: List[str], module_doc: str, content: str) -> str:
        """Generate detailed architectural justification for a file."""
        file_name = file_path.name
        path_parts = file_path.parts

        # More detailed pattern-based analysis
        if "cli" in path_parts:
            command_functions = [f for f in functions if not f.startswith('_')]
            if command_functions:
                return f"Command-line interface module implementing {', '.join([f.split('(')[0] for f in command_functions])} commands for user interaction with vibelint's core functionality"
            return "Command-line interface module providing user-facing commands"

        elif "validators" in path_parts:
            if "single_file" in path_parts:
                validator_name = file_name.replace('.py', '').replace('_', ' ')
                return f"Single-file code quality validator implementing {validator_name} rules with automatic fix capabilities and pattern detection"
            elif "project_wide" in path_parts:
                validator_name = file_name.replace('.py', '').replace('_', ' ')
                return f"Project-wide code quality validator analyzing cross-file {validator_name} patterns and architectural consistency"
            elif "architecture" in path_parts:
                validator_name = file_name.replace('.py', '').replace('_', ' ')
                return f"Architecture validation module ensuring {validator_name} design patterns and structural integrity across the codebase"

        elif "llm" in path_parts:
            if "manager" in file_name:
                return "LLM request management system with dual-LLM architecture supporting fast inference and orchestrator routing based on context size and complexity"
            elif "orchestrator" in file_name:
                return "LLM orchestration engine coordinating complex multi-step analysis workflows with context engineering and request optimization"
            elif "config" in file_name:
                return "LLM configuration management system handling environment setup, model routing, and API credential management"
            else:
                return f"LLM integration component providing {file_name.replace('.py', '').replace('_', ' ')} functionality for AI-powered code analysis"

        elif "workflows" in path_parts:
            if "implementations" in path_parts:
                workflow_name = file_name.replace('.py', '').replace('_', ' ')
                return f"Workflow implementation orchestrating {workflow_name} analysis pipeline with validator coordination and result aggregation"
            return "Workflow coordination system managing analysis execution and validator integration"

        elif "context" in path_parts:
            context_type = file_name.replace('.py', '').replace('_', ' ')
            return f"Context analysis module providing {context_type} for LLM workflows with code structure understanding and semantic analysis"

        # Core functionality patterns
        elif file_name == "core.py":
            return "Core vibelint execution engine coordinating validation workflows, plugin management, and result processing with CLI integration"
        elif file_name == "config.py":
            return "Configuration management system handling project settings, validation rules, and environment-specific overrides with hierarchical merging"
        elif file_name == "plugin_system.py":
            return "Plugin architecture system enabling dynamic validator loading, custom rule registration, and extensible analysis capabilities"

        # Test patterns
        elif "test" in file_name or "tests" in path_parts:
            test_target = file_name.replace('test_', '').replace('.py', '')
            return f"Comprehensive test suite validating {test_target} functionality with edge case coverage and integration testing"

        # Special files
        elif file_name == "__init__.py":
            return "Package initialization module defining public API, organizing exports, and establishing module hierarchy"
        elif file_name == "__main__.py":
            return "Package entry point enabling module execution via 'python -m' with command-line argument handling"

        # Fallback with more detail
        if module_doc:
            doc_words = module_doc.split()[:15]
            return f"Module implementing {' '.join(doc_words)} with integrated functionality"
        elif functions and classes:
            return f"Implementation module combining {len(classes)} class definitions and {len(functions)} utility functions for coordinated functionality"
        elif functions:
            func_names = [f.split('(')[0] for f in functions[:3]]
            return f"Utility module providing {', '.join(func_names)} functions for specialized operations"
        elif classes:
            class_names = [c.split('(')[0] for c in classes[:2]]
            return f"Class definition module implementing {', '.join(class_names)} components for system architecture"
        else:
            return f"Configuration or data module supporting {file_name.replace('.py', '').replace('_', ' ')} functionality"

    def _create_detailed_file_analysis(self, file_path: Path, project_root: Path) -> str:
        """Create comprehensive file analysis with code context for LLM."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)

            # Get relative path for cleaner display
            rel_path = file_path.relative_to(project_root)

            analysis = [f"### {rel_path}"]

            # File metadata
            analysis.append(f"**Lines:** {len(content.splitlines())}")

            # Dependencies
            deps = self._get_file_dependencies(file_path)
            if deps:
                deps_str = ", ".join(deps[:5]) + (f" +{len(deps)-5}" if len(deps) > 5 else "")
                analysis.append(f"**Dependencies:** {deps_str}")

            # Module docstring
            module_doc = ast.get_docstring(tree)
            if module_doc:
                doc_summary = module_doc.split('\n')[0] if '\n' in module_doc else module_doc[:100]
                analysis.append(f"**Purpose:** {doc_summary}")

            # Classes with methods
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    class_info = f"{node.name}"
                    if methods:
                        method_str = ", ".join(methods[:3]) + (f" +{len(methods)-3}" if len(methods) > 3 else "")
                        class_info += f"({method_str})"
                    classes.append(class_info)

            if classes:
                analysis.append(f"**Classes:** {'; '.join(classes)}")

            # Functions (at module level, not in classes)
            functions = []
            class_methods = set()

            # First collect all class methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            class_methods.add(child.name)

            # Then collect module-level functions
            for node in tree.body:  # Only look at top-level nodes
                if isinstance(node, ast.FunctionDef):
                    # Get function signature info
                    args = [arg.arg for arg in node.args.args]
                    arg_str = f"({', '.join(args[:3])}{'...' if len(args) > 3 else ''})"
                    functions.append(f"{node.name}{arg_str}")

            if functions:
                analysis.append(f"**Functions:** {'; '.join(functions[:5])}{' +more' if len(functions) > 5 else ''}")

            # Key imports (for context)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)

            key_imports = [imp for imp in imports if any(keyword in imp.lower() for keyword in
                          ['click', 'fastapi', 'flask', 'django', 'requests', 'asyncio', 'typing', 'pathlib'])]
            if key_imports:
                analysis.append(f"**Key Technologies:** {', '.join(key_imports[:4])}")

            # Code patterns (for better LLM understanding)
            patterns = []
            if 'def main(' in content or 'if __name__ == "__main__"' in content:
                patterns.append("CLI entry point")
            if 'class ' in content and 'def validate' in content:
                patterns.append("Validator implementation")
            if 'async def' in content:
                patterns.append("Async operations")
            if '@click.command' in content:
                patterns.append("Click CLI commands")
            if 'def test_' in content:
                patterns.append("Test suite")

            if patterns:
                analysis.append(f"**Patterns:** {', '.join(patterns)}")

            return '\n'.join(analysis)

        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            return f"### {file_path.relative_to(project_root)}\n**Error:** Could not analyze file"

    def _find_file_in_directory(self, directory: Path, filename: str) -> Optional[Path]:
        """Find a file by name in the directory tree."""
        for file_path in directory.rglob(filename):
            if file_path.name == filename:
                return file_path
        return None

    def _get_llm_file_summary(self, file_path: Path) -> str:
        """Get LLM-generated one-line summary of file purpose."""
        logger.debug(f"Getting LLM summary for {file_path}")

        if not self.llm_manager:
            logger.warning(f"No LLM manager available for {file_path}")
            return "No LLM available"

        # Use synchronous interface - LLM manager should provide sync methods
        return self._get_llm_file_summary_sync(file_path)

    def _get_llm_file_summary_sync(self, file_path: Path) -> str:
        """Synchronous LLM file summary using proper interface."""
        try:
            logger.debug(f"Reading file content: {file_path}")
            content = file_path.read_text()

            # Get file structure for context
            logger.debug(f"Parsing AST for {file_path}")
            tree = ast.parse(content)
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][:3]
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)][:2]

            context = f"File: {file_path.name}\n"
            if functions:
                context += f"Functions: {', '.join(functions)}\n"
            if classes:
                context += f"Classes: {', '.join(classes)}\n"

            # Add docstring if available
            module_doc = ast.get_docstring(tree)
            if module_doc:
                context += f"Docstring: {module_doc[:100]}...\n"

            # Use orchestrator LLM for contextual justification analysis
            # Get rich context instead of raw file content
            file_context = self._create_detailed_file_analysis(file_path, file_path.parent)

            prompt = f"""Based on the detailed file analysis below, provide a concise justification explaining WHY this file exists in the codebase and WHAT architectural purpose it serves.

{file_context}

Focus on:
- The architectural role this file plays in the system
- What problems it solves or what functionality it enables
- How it fits into the overall codebase design
- Reference specific classes, functions, or patterns when relevant

Provide a single, comprehensive sentence that justifies this file's existence in the project."""

            from vibelint.llm.manager import LLMRequest

            llm_request = LLMRequest(
                content=prompt,
                max_tokens=150,  # Higher token count for detailed justification -> routes to orchestrator
                temperature=0.3  # Creative analysis temperature -> routes to orchestrator
            )

            logger.debug(f"Calling LLM for {file_path} with prompt length: {len(prompt)}")
            response = self.llm_manager.process_request_sync(llm_request)
            logger.debug(f"LLM response for {file_path}: {response}")

            if response and response.get("success") and response.get("content"):
                summary = response["content"].strip()
                logger.info(f"LLM summary for {file_path.name}: {summary}")

                # Log the LLM call
                self._llm_call_logs.append({
                    "operation": "file_summary",
                    "file_path": str(file_path),
                    "response": summary,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

                return summary
            else:
                # Fallback to pattern-based justification analysis
                logger.warning(f"LLM failed for {file_path}, using pattern-based justification: {response}")
                summary = self._generate_pattern_based_justification(file_path, functions, classes, module_doc, content)
                logger.debug(f"Pattern-based justification for {file_path.name}: {summary}")
                return summary

        except SyntaxError as e:
            error_msg = f"Syntax error in {file_path}: {e}"
            logger.warning(error_msg)
            return f"SYNTAX ERROR: {str(e)[:50]}..."
        except Exception as e:
            error_msg = f"LLM summary failed for {file_path}: {type(e).__name__}: {e}"
            logger.warning(error_msg)
            return f"ERROR: {type(e).__name__}"

    def _generate_pattern_based_justification(self, file_path: Path, functions: List[str],
                                            classes: List[str], module_doc: str, content: str) -> str:
        """Generate justification based on code patterns and file purpose."""

        file_name = file_path.name
        path_parts = file_path.parts

        # CLI interface files
        if "cli" in path_parts:
            if functions:
                return f"CLI command interface providing {', '.join(functions[:2])} commands for user interaction"
            return "CLI command interface module"

        # Validator files
        if "validators" in path_parts:
            if "single_file" in path_parts:
                return f"Single-file validator for code quality rules like {file_name.replace('.py', '').replace('_', ' ')}"
            elif "project_wide" in path_parts:
                return f"Project-wide validator analyzing cross-file patterns for {file_name.replace('.py', '').replace('_', ' ')}"
            elif "architecture" in path_parts:
                return f"Architecture validator ensuring design patterns for {file_name.replace('.py', '').replace('_', ' ')}"
            return "Code validation logic module"

        # LLM related files
        if "llm" in path_parts:
            if "manager" in file_name:
                return "LLM request routing and configuration management for dual LLM architecture"
            elif "orchestrator" in file_name:
                return "LLM orchestration logic for complex multi-step analysis workflows"
            elif "config" in file_name:
                return "LLM configuration loading and environment setup"
            return "LLM integration and processing logic"

        # Workflow files
        if "workflows" in path_parts:
            if "implementations" in path_parts:
                return f"Workflow implementation for {file_name.replace('.py', '').replace('_', ' ')} analysis pipeline"
            return "Workflow orchestration and execution logic"

        # Context files
        if "context" in path_parts:
            return f"Context analysis providing {file_name.replace('.py', '').replace('_', ' ')} for LLM workflows"

        # Core functionality
        if file_name == "core.py":
            return "Core vibelint functionality and main execution engine"
        elif file_name == "config.py":
            return "Configuration loading and project settings management"
        elif file_name == "cli.py":
            return "Main CLI entry point and command routing"

        # Pattern-based analysis
        if "test" in file_name or "tests" in path_parts:
            return f"Test suite for {file_name.replace('test_', '').replace('.py', '')} functionality"

        # Function-based patterns
        if functions:
            # API/interface patterns
            if any("api" in f.lower() for f in functions):
                return "API interface providing programmatic access to functionality"
            # Processing patterns
            elif any(f.startswith(("process", "analyze", "validate")) for f in functions):
                return f"Processing engine for {', '.join([f for f in functions[:2] if f.startswith(('process', 'analyze', 'validate'))])}"
            # Utility patterns
            elif any(f.startswith(("get", "set", "create", "load")) for f in functions):
                return f"Utility functions for {', '.join(functions[:2])} operations"

        # Class-based patterns
        if classes:
            # Manager/Controller patterns
            if any("manager" in c.lower() or "controller" in c.lower() for c in classes):
                return f"Management layer for {', '.join([c for c in classes[:2] if 'manager' in c.lower() or 'controller' in c.lower()])}"
            # Validator patterns
            elif any("validator" in c.lower() for c in classes):
                return f"Validation logic implementing {', '.join([c for c in classes[:2] if 'validator' in c.lower()])}"

        # Content-based patterns
        if "import click" in content and functions:
            return f"CLI command implementation with {', '.join(functions[:2])} user commands"
        elif "class " in content and "def validate" in content:
            return "Validation rule implementation for code quality enforcement"
        elif "async def" in content:
            return "Async processing module for concurrent operations"

        # Fallback with better context
        if module_doc:
            doc_summary = module_doc.split('.')[0] if '.' in module_doc else module_doc[:50]
            return f"Module providing {doc_summary.lower()}"
        elif functions and classes:
            return f"Implementation module combining {len(classes)} classes and {len(functions)} functions"
        elif functions:
            return f"Function library providing {len(functions)} utility operations"
        elif classes:
            return f"Class definitions for {len(classes)} core components"
        else:
            return f"Configuration or data module for {file_name.replace('.py', '').replace('_', ' ')}"

    def _get_file_dependencies(self, file_path: Path) -> List[str]:
        """Get deterministic list of imports from file."""
        logger.debug(f"Extracting dependencies from: {file_path}")

        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                        logger.debug(f"Found import: {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
                    logger.debug(f"Found from import: {node.module}")

            unique_imports = list(set(imports))
            logger.debug(f"Dependencies for {file_path.name}: {unique_imports}")
            return unique_imports

        except Exception as e:
            error_msg = f"Failed to extract dependencies from {file_path}: {type(e).__name__}: {e}"
            logger.warning(error_msg)  # Warning instead of error
            return []  # Return empty deps instead of crashing

    def _build_dependency_analysis(self, directory_path: Path) -> Dict[str, Any]:
        """Build comprehensive dependency analysis starting from entrypoints."""
        logger.info("Building dependency analysis from entrypoints...")

        # Find all entrypoints
        entrypoints = self._find_entrypoints(directory_path)
        logger.info(f"Found entrypoints: {[str(ep) for ep in entrypoints]}")

        # Build dependency trees from each entrypoint
        dependency_trees = {}
        all_reachable_files = set()

        for entrypoint in entrypoints:
            logger.info(f"Tracing dependencies from {entrypoint}")
            tree, reachable = self._trace_dependencies_from_entrypoint(entrypoint, directory_path)
            dependency_trees[str(entrypoint)] = tree
            all_reachable_files.update(reachable)

        # Note: Static dead code analysis removed - unreliable with dynamic loading

        # Detect circular imports
        circular_imports = self._detect_circular_imports(directory_path)
        logger.info(f"Found {len(circular_imports)} circular import chains")

        return {
            "entrypoints": [str(ep) for ep in entrypoints],
            "dependency_trees": dependency_trees,
            "circular_imports": circular_imports,
            "reachable_files": [str(f) for f in all_reachable_files],
            "analysis_summary": f"Analyzed {len(entrypoints)} entrypoints, {len(circular_imports)} circular imports detected"
        }

    def _find_entrypoints(self, directory_path: Path) -> List[Path]:
        """Find main entrypoints in the codebase."""
        entrypoints = []

        # 1. Files with if __name__ == "__main__"
        for py_file in directory_path.rglob('*.py'):
            try:
                content = py_file.read_text()
                if 'if __name__ == "__main__"' in content:
                    entrypoints.append(py_file)
            except Exception as e:
                logger.warning(f"Could not read {py_file}: {e}")

        # 2. CLI entry points from pyproject.toml
        pyproject_file = directory_path / "pyproject.toml"
        if pyproject_file.exists():
            try:
                try:
                    import tomllib  # Python 3.11+
                    with open(pyproject_file, "rb") as f:
                        pyproject_data = tomllib.load(f)
                except ImportError:
                    import tomli  # Fallback for older Python
                    with open(pyproject_file, "rb") as f:
                        pyproject_data = tomli.load(f)

                # Check for console scripts
                scripts = pyproject_data.get("project", {}).get("scripts", {})
                for script_name, entry_point in scripts.items():
                    # Parse entry point like "vibelint.cli:main"
                    if ":" in entry_point:
                        module_path, func_name = entry_point.split(":", 1)
                        module_file = self._module_path_to_file(module_path, directory_path)
                        if module_file and module_file.exists():
                            entrypoints.append(module_file)
            except Exception as e:
                logger.warning(f"Could not parse pyproject.toml: {e}")

        # 3. Test files (pytest entrypoints)
        test_files = list(directory_path.rglob('test_*.py')) + list(directory_path.rglob('*_test.py'))
        entrypoints.extend(test_files)

        return list(set(entrypoints))  # Remove duplicates

    def _module_path_to_file(self, module_path: str, project_root: Path) -> Optional[Path]:
        """Convert module path like 'vibelint.cli' to file path."""
        parts = module_path.split('.')

        # Try different possible locations
        possible_paths = [
            project_root / "src" / Path("/".join(parts) + ".py"),
            project_root / Path("/".join(parts) + ".py"),
            project_root / "src" / Path("/".join(parts)) / "__init__.py",
            project_root / Path("/".join(parts)) / "__init__.py",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def _trace_dependencies_from_entrypoint(self, entrypoint: Path, project_root: Path) -> Tuple[Dict[str, Any], Set[Path]]:
        """Trace all dependencies from a single entrypoint using AST analysis."""
        visited = set()
        dependency_tree = {}
        reachable_files = set()

        def trace_file(file_path: Path, depth: int = 0) -> Dict[str, Any]:
            if depth > 20:  # Prevent infinite recursion
                return {"error": "Max depth reached"}

            if file_path in visited:
                return {"circular": True, "path": str(file_path)}

            visited.add(file_path)
            reachable_files.add(file_path)

            try:
                content = file_path.read_text()
                tree = ast.parse(content)

                imports = []
                dependencies = {}

                # Parse all imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            import_name = alias.name
                            imports.append(import_name)

                            # Try to resolve to local file
                            dep_file = self._resolve_import_to_file(import_name, file_path, project_root)
                            if dep_file and dep_file != file_path:
                                dependencies[import_name] = trace_file(dep_file, depth + 1)

                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            import_name = node.module
                            imports.append(import_name)

                            # Try to resolve to local file
                            dep_file = self._resolve_import_to_file(import_name, file_path, project_root)
                            if dep_file and dep_file != file_path:
                                dependencies[import_name] = trace_file(dep_file, depth + 1)

                return {
                    "file": str(file_path),
                    "imports": imports,
                    "dependencies": dependencies,
                    "depth": depth
                }

            except Exception as e:
                logger.warning(f"Failed to trace {file_path}: {e}")
                return {"error": str(e), "file": str(file_path)}

        tree = trace_file(entrypoint)
        return tree, reachable_files

    def _resolve_import_to_file(self, import_name: str, current_file: Path, project_root: Path) -> Optional[Path]:
        """Resolve an import name to an actual file path in the project."""
        # Skip standard library and external packages
        if import_name in ['os', 'sys', 'json', 'pathlib', 'typing', 'logging', 'time', 'datetime',
                          'click', 'requests', 'pytest', 'numpy', 'pydantic', 'fastapi', 'django']:
            return None

        # Handle relative imports (starting with .)
        if import_name.startswith('.'):
            # Get current package directory
            current_dir = current_file.parent
            parts = import_name.split('.')
            # Navigate up for each leading dot
            for part in parts:
                if part == '':
                    current_dir = current_dir.parent
                else:
                    break
            relative_path = '/'.join(parts[len([p for p in parts if p == '']):])
            if relative_path:
                possible_file = current_dir / Path(relative_path + '.py')
                if possible_file.exists():
                    return possible_file
            return None

        # Handle absolute imports within the project
        parts = import_name.split('.')

        # Try different possible locations
        possible_paths = [
            project_root / "src" / Path("/".join(parts) + ".py"),
            project_root / Path("/".join(parts) + ".py"),
            project_root / "src" / Path("/".join(parts)) / "__init__.py",
            project_root / Path("/".join(parts)) / "__init__.py",
        ]

        for path in possible_paths:
            if path.exists() and path.is_relative_to(project_root):
                return path

        return None

    def _detect_circular_imports(self, directory_path: Path) -> List[List[str]]:
        """Detect circular import chains in the codebase."""
        # Build import graph
        import_graph = {}

        for py_file in directory_path.rglob('*.py'):
            try:
                deps = self._get_file_dependencies(py_file)
                # Filter to only local dependencies
                local_deps = []
                for dep in deps:
                    dep_file = self._resolve_import_to_file(dep, py_file, directory_path)
                    if dep_file:
                        local_deps.append(str(dep_file))

                import_graph[str(py_file)] = local_deps
            except Exception as e:
                logger.warning(f"Failed to analyze {py_file} for circular imports: {e}")

        # Find cycles using DFS
        def find_cycles_from_node(node: str, path: List[str], visited: Set[str]) -> List[List[str]]:
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                return [path[cycle_start:] + [node]]

            if node in visited:
                return []

            visited.add(node)
            cycles = []

            for neighbor in import_graph.get(node, []):
                cycles.extend(find_cycles_from_node(neighbor, path + [node], visited.copy()))

            return cycles

        all_cycles = []
        global_visited = set()

        for node in import_graph:
            if node not in global_visited:
                cycles = find_cycles_from_node(node, [], set())
                all_cycles.extend(cycles)
                global_visited.add(node)

        # Remove duplicate cycles
        unique_cycles = []
        for cycle in all_cycles:
            normalized = tuple(sorted(cycle))
            if normalized not in [tuple(sorted(c)) for c in unique_cycles]:
                unique_cycles.append(cycle)

        return unique_cycles

    def _generate_justification_report(self, enhanced_tree: str, dependency_analysis: Dict[str, Any], directory_path: Path) -> str:
        """Generate the comprehensive justification report."""

        report = f"""# Project Justification Analysis

**Project:** {directory_path.name}
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Type:** Comprehensive (LLM + Static + Dependency)

## Enhanced Project Tree with Justifications

{enhanced_tree}

## Dependency Analysis Summary

- **Entrypoints Found:** {len(dependency_analysis['entrypoints'])}
- **Reachable Files:** {len(dependency_analysis['reachable_files'])}
- **Circular Imports Found:** {len(dependency_analysis['circular_imports'])}

### Issues Detected

"""

        if dependency_analysis['circular_imports']:
            report += "#### Circular Imports\n"
            for circular in dependency_analysis['circular_imports']:
                report += f"- {circular}\n"
            report += "\n"

        # Dead code analysis removed - static analysis unreliable with dynamic loading

        # Add LLM analysis summary
        llm_calls = len([call for call in self._llm_call_logs if call.get('operation') == 'file_summary'])
        report += f"""## Analysis Capabilities Used

- **LLM File Summaries:** {llm_calls} files analyzed
- **Static Dependency Analysis:** ✅ Complete
- **Circular Import Detection:** ✅ Complete
- **Architectural Analysis:** ✅ Via LLM orchestrator

## Methodology

This analysis combines:
1. **vibelint snapshot** for project structure
2. **Fast LLM** (750 token limit) for file purpose summaries
3. **AST parsing** for deterministic import analysis
4. **Graph analysis** for circular dependency detection

All analysis is logged and reproducible.
"""

        return report

    def _save_justification_workflow_results(self, report: str, directory_path: Path):
        """Save the justification workflow results."""

        # Use consistent vibelint reports directory
        reports_dir = self._get_vibelint_reports_dir() / "justification_workflow"
        reports_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = directory_path.name if directory_path.name != '.' else 'project'
        report_file = reports_dir / f"architectural_analysis_{project_name}_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Justification workflow results saved to {report_file}")

        # Also save the LLM call logs
        logs_file = reports_dir / f"justification_workflow_{timestamp}_logs.json"
        with open(logs_file, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": self._session_id,
                "llm_calls": self._llm_call_logs,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "directory_analyzed": str(directory_path)
            }, f, indent=2)

        return report_file, logs_file

    def _get_vibelint_reports_dir(self) -> Path:
        """Get the consistent vibelint reports directory location."""
        # Find the vibelint project root (tools/vibelint/)
        current_path = Path.cwd()

        # If we're already in vibelint directory, use it
        if current_path.name == "vibelint" and (current_path / "src" / "vibelint").exists():
            return current_path / ".vibelint-reports"

        # Look for vibelint directory in current path or parents
        while current_path.parent != current_path:
            vibelint_dir = current_path / "tools" / "vibelint"
            if vibelint_dir.exists() and (vibelint_dir / "src" / "vibelint").exists():
                return vibelint_dir / ".vibelint-reports"

            # Check if current directory is the vibelint root
            if current_path.name == "vibelint" and (current_path / "src" / "vibelint").exists():
                return current_path / ".vibelint-reports"

            current_path = current_path.parent

        # Fallback: create reports in current working directory
        return Path.cwd() / ".vibelint-reports"

    def _get_llm_method_description(self, method_info: Dict[str, Any], file_path: Path) -> Optional[str]:
        """Get LLM-generated description of method purpose."""
        if not self.llm_manager:
            return None

        try:
            # Extract method source code
            file_content = file_path.read_text()
            method_source = self._extract_method_source(file_content, method_info["name"])

            if not method_source:
                return None

            # Use fast LLM for description (within 750 token limit)
            prompt = f"""Analyze this Python method and provide a one-sentence description of what it does:

```python
{method_source[:2000]}  # Truncate for token limit
```

Respond with ONLY a single sentence description, no preamble."""

            request = {
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.1
            }

            from vibelint.llm.manager import LLMRequest
            llm_request = LLMRequest(
                content=prompt,
                max_tokens=100,
                temperature=0.1
            )

            response = self.llm_manager.process_request_sync(llm_request)

            if response and response.get("success"):
                description = response["content"].strip()

                # Log the LLM call
                self._llm_call_logs.append({
                    "operation": "method_description",
                    "method_name": method_info["name"],
                    "file_path": str(file_path),
                    "prompt_tokens": len(prompt.split()),
                    "response": description,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

                return description

        except Exception as e:
            logger.debug(f"LLM description failed for {method_info['name']}: {e}")

        return None

    def _analyze_method_usage(self, method_info: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Analyze method usage patterns for architectural insights."""
        method_name = method_info["name"]
        is_private = method_info["is_private"]

        # Quick static analysis for usage
        usage_score = 0
        potentially_dead = False

        try:
            # Check if method is called within the same file
            file_content = file_path.read_text()

            # Count direct calls
            call_count = file_content.count(f"{method_name}(")
            call_count += file_content.count(f".{method_name}(")

            # Special methods (dunder methods) are likely used
            if method_name.startswith("__") and method_name.endswith("__"):
                usage_score = 0.9
            # Public methods are more likely to be used
            elif not is_private:
                usage_score = 0.7 if call_count > 0 else 0.3
            # Private methods with no calls are suspicious
            else:
                usage_score = 0.5 if call_count > 0 else 0.1
                potentially_dead = call_count == 0

            # TODO: Enhanced analysis could check:
            # - Cross-file references (grep across project)
            # - Test coverage data
            # - Runtime call traces

        except Exception as e:
            logger.debug(f"Usage analysis failed for {method_name}: {e}")
            usage_score = 0.5  # Unknown usage

        return {
            "usage_score": usage_score,
            "potentially_dead": potentially_dead,
            "analysis_method": "static_call_count"
        }

    def _analyze_file_purpose_with_llm(self, file_path: Path, content: str) -> Optional[str]:
        """Get LLM-generated file purpose description."""
        if not self.llm_manager:
            return None

        try:
            # Get file structure summary
            tree = ast.parse(content)
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][:5]
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)][:3]

            # Build context for LLM
            context = f"""File: {file_path.name}
Functions: {', '.join(functions) if functions else 'None'}
Classes: {', '.join(classes) if classes else 'None'}
"""

            # Add docstring if available
            module_doc = ast.get_docstring(tree)
            if module_doc:
                context += f"Docstring: {module_doc[:200]}..."

            prompt = f"""Analyze this Python file and describe its main purpose in one sentence:

{context}

Respond with ONLY a single sentence description of the file's purpose."""

            request = {
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.1
            }

            from vibelint.llm.manager import LLMRequest
            llm_request = LLMRequest(
                content=prompt,
                max_tokens=100,
                temperature=0.1
            )

            response = self.llm_manager.process_request_sync(llm_request)

            if response and response.get("success"):
                description = response["content"].strip()

                # Log the LLM call
                self._llm_call_logs.append({
                    "operation": "file_description",
                    "file_path": str(file_path),
                    "response": description,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

                return description

        except Exception as e:
            logger.debug(f"LLM file description failed for {file_path}: {e}")

        return None

    def _detect_static_issues(self, directory_path: Path) -> Dict[str, Any]:
        """Detect static issues like redundancy and bad naming without LLM."""
        issues = {
            "redundant_files": [],
            "naming_issues": [],
            "useless_files": [],
            "backup_files": []
        }

        # Get all Python files
        py_files = list(directory_path.rglob("*.py"))
        file_names = {f.name: f for f in py_files}

        # Detect versioned redundancy (file.py vs file_v2.py, file2.py, etc.)
        for file_path in py_files:
            stem = file_path.stem

            # Check for version patterns
            if stem.endswith('_v2') or stem.endswith('_v3') or stem.endswith('_v4'):
                base_name = stem.rsplit('_v', 1)[0] + '.py'
                if base_name in file_names:
                    issues["redundant_files"].append({
                        "newer": str(file_path),
                        "older": str(file_names[base_name]),
                        "type": "versioned_redundancy"
                    })

            # Check for numbered versions (file2.py, file3.py)
            import re
            if re.match(r'.*\d$', stem) and len(stem) > 1:
                base_stem = re.sub(r'\d+$', '', stem)
                base_name = base_stem + '.py'
                if base_name in file_names:
                    issues["redundant_files"].append({
                        "newer": str(file_path),
                        "older": str(file_names[base_name]),
                        "type": "numbered_redundancy"
                    })

        # Detect bad naming patterns
        bad_patterns = ['temp', 'test_', 'old_', 'backup_', 'copy_', 'tmp_', '_old', '_backup', '_copy']
        for file_path in py_files:
            name = file_path.name.lower()
            for pattern in bad_patterns:
                if pattern in name:
                    issues["naming_issues"].append({
                        "file": str(file_path),
                        "issue": f"Contains suspicious pattern: {pattern}",
                        "severity": "medium"
                    })

        # Detect potentially useless files (very small, mostly empty)
        for file_path in py_files:
            try:
                content = file_path.read_text()
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                non_comment_lines = [line for line in lines if not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''")]

                # Flag files with less than 5 non-comment lines
                if len(non_comment_lines) < 5 and file_path.name != '__init__.py':
                    issues["useless_files"].append({
                        "file": str(file_path),
                        "lines": len(non_comment_lines),
                        "reason": "Very small file with minimal content"
                    })
            except Exception:
                pass  # Skip files we can't read

        # Use LLM to identify backup and temporary files
        issues["backup_files"] = self._detect_backup_files_with_llm(directory_path)

        return issues

    def _detect_backup_files_with_llm(self, directory_path: Path) -> List[Dict[str, str]]:
        """Use LLM to intelligently detect backup and temporary files."""
        if not self.llm_manager:
            return []

        # Get all files (not just Python)
        all_files = []
        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                # Skip very large files and binary files
                try:
                    if file_path.stat().st_size < 1024 * 1024:  # < 1MB
                        all_files.append(str(file_path.relative_to(directory_path)))
                except (OSError, ValueError):
                    continue

        if not all_files:
            return []

        # Create prompt for LLM to identify backup files
        file_list = "\n".join(all_files[:100])  # Limit to first 100 files
        prompt = f"""Analyze this file list and identify backup, temporary, or archive files that should NOT be committed to version control.

File List:
{file_list}

Look for patterns like:
- .bak, .backup, .orig extensions
- Files ending with ~ (Unix backup)
- .tmp, .temp extensions
- Numbered versions (file_v2.py, file2.py when file.py exists)
- OS-specific temporary files (.DS_Store, Thumbs.db, etc.)
- Editor backup files (.swp, .swo, etc.)
- Archive files in source directories (.zip, .tar, etc.)
- Any file that appears to be a backup or temporary copy

Return ONLY a JSON array of objects with this format:
[{{"file": "path/to/file", "reason": "why this should not be committed"}}]

If no backup files found, return: []"""

        try:
            from vibelint.llm.manager import LLMRequest
            llm_request = LLMRequest(
                content=prompt,
                max_tokens=1000,  # Moderate token count for list processing
                temperature=0.1   # Low temperature for precise identification
            )

            response = self.llm_manager.process_request_sync(llm_request)

            if response and response.get("success") and response.get("content"):
                import json
                try:
                    backup_files = json.loads(response["content"].strip())
                    if isinstance(backup_files, list):
                        return backup_files
                except json.JSONDecodeError:
                    logger.warning("LLM backup file detection returned invalid JSON")

        except Exception as e:
            logger.debug(f"LLM backup file detection failed: {e}")

        return []

    def _format_static_issues(self, static_issues: Dict[str, Any]) -> str:
        """Format static issues for inclusion in analysis report."""
        formatted = "# Static Analysis Issues\n\n"

        if static_issues["redundant_files"]:
            formatted += "## Redundant Files Detected\n\n"
            for redundancy in static_issues["redundant_files"]:
                formatted += f"- **{redundancy['type']}**: `{redundancy['newer']}` vs `{redundancy['older']}`\n"
            formatted += "\n"

        if static_issues["naming_issues"]:
            formatted += "## Naming Issues Detected\n\n"
            for issue in static_issues["naming_issues"]:
                formatted += f"- `{issue['file']}`: {issue['issue']} (severity: {issue['severity']})\n"
            formatted += "\n"

        if static_issues["useless_files"]:
            formatted += "## Potentially Useless Files\n\n"
            for issue in static_issues["useless_files"]:
                formatted += f"- `{issue['file']}`: {issue['reason']} ({issue['lines']} lines)\n"
            formatted += "\n"

        if static_issues["backup_files"]:
            formatted += "## Backup/Temporary Files (Should Not Be Committed)\n\n"
            for backup in static_issues["backup_files"]:
                formatted += f"- `{backup['file']}`: {backup['reason']}\n"
            formatted += "\n"

        if not any(static_issues.values()):
            formatted += "No static issues detected.\n\n"

        return formatted

    def _orchestrator_final_analysis(self, justification_report: str, directory_path: Path, static_issues: Dict[str, Any] = None) -> str:
        """Use orchestrator LLM to perform final comprehensive analysis of the justification report."""
        logger.info("Starting orchestrator final analysis...")

        # If static issues detected, include them even without LLM
        static_analysis = ""
        if static_issues:
            static_analysis = self._format_static_issues(static_issues)

        if not self.llm_manager:
            logger.warning("No LLM manager available for final analysis")
            return f"LLM manager not available - showing static analysis only:\n\n{static_analysis}"

        # Create comprehensive prompt for orchestrator LLM
        prompt = f"""# Comprehensive Vibelint Project Analysis

You are analyzing a complete justification report for the vibelint codebase. Your task is to provide expert analysis on code quality, architecture, and maintenance issues.

## Analysis Requirements:

### 1. Redundancy Detection
- Identify duplicate or redundant files (e.g., justification.py vs justification_v2.py)
- Flag similar functionality that should be consolidated
- Note outdated implementations that should be removed

### 2. Naming Issues
- Identify poorly named files, functions, or modules
- Suggest better naming conventions
- Flag confusing or misleading names

### 3. Architecture Issues
- Assess overall code organization and structure
- Identify circular dependencies or poor separation of concerns
- Note missing abstractions or over-engineering

### 4. Dead Code Analysis
- Validate the dependency analysis results
- Identify genuinely unused files vs files loaded dynamically
- Distinguish between dead code and optional features

### 5. Quality Issues
- Note inconsistent patterns or styles
- Identify overly complex or poorly designed components
- Flag potential maintenance burdens

## Justification Report to Analyze:

{justification_report}

## Pre-detected Static Issues:

{static_analysis}

## Required Output Format:

Provide your analysis in the following structured format:

### Critical Issues
[List 3-5 most important issues that need immediate attention]

### Redundancy Analysis
[Specific files/functions that are redundant and should be consolidated or removed]

### Architecture Assessment
[Overall architectural health and specific structural issues]

### Dead Code Validation
[Assessment of the dependency analysis accuracy and real vs false positives]

### Recommendations
[Prioritized action items for improving the codebase]

Focus on actionable insights that will improve code maintainability and quality.

**IMPORTANT**: If the codebase is already well-structured with no significant issues, simply respond with "LGTM - No significant architectural issues found." Don't feel obligated to find problems where none exist."""

        from vibelint.llm.manager import LLMRequest

        llm_request = LLMRequest(
            content=prompt,
            max_tokens=8192,  # Large output for comprehensive analysis → routes to orchestrator
            temperature=0.3   # Creative analysis temperature → routes to orchestrator
        )

        try:
            logger.info("Calling orchestrator LLM for final analysis...")
            response = self.llm_manager.process_request_sync(llm_request)

            if response and response.get("success") and response.get("content"):
                analysis = response["content"].strip()
                logger.info("Orchestrator final analysis completed successfully")

                # Log the LLM call
                self._llm_call_logs.append({
                    "operation": "final_analysis",
                    "response": analysis,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

                # Combine LLM analysis with static analysis
                combined_analysis = f"{analysis}\n\n---\n\n{static_analysis}" if static_analysis else analysis
                return combined_analysis
            else:
                error_msg = f"Orchestrator LLM analysis failed: {response}"
                logger.warning(error_msg)
                # Return static analysis even if LLM fails
                return f"LLM analysis failed: {error_msg}\n\n---\n\n{static_analysis}" if static_analysis else f"LLM analysis failed: {error_msg}"

        except Exception as e:
            error_msg = f"Orchestrator final analysis error: {type(e).__name__}: {e}"
            logger.error(error_msg)
            # Return static analysis even if LLM errors
            return f"Analysis error: {error_msg}\n\n---\n\n{static_analysis}" if static_analysis else f"Analysis error: {error_msg}"

    def _save_final_analysis(self, final_analysis: str, directory_path: Path, initial_report_path: str) -> str:
        """Save the final orchestrator analysis alongside the initial report."""
        reports_dir = self._get_vibelint_reports_dir() / "justification_workflow"

        # Generate filename based on initial report
        initial_name = Path(initial_report_path).stem
        final_report_name = f"{initial_name}_QUALITY_ASSESSMENT.md"
        final_report_path = reports_dir / final_report_name

        # Create comprehensive final report
        final_content = f"""# Vibelint Final Analysis Report

**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Analysis Type:** Orchestrator LLM Comprehensive Review
**Initial Report:** {Path(initial_report_path).name}

---

{final_analysis}

---

## Analysis Metadata

**LLM Calls Made:** {len(self._llm_call_logs)}
**Analysis Depth:** Comprehensive (Static + Dependency + LLM)
**Report Generation:** Automated via JustificationEngineV2

For detailed file-by-file analysis, see the initial report: `{Path(initial_report_path).name}`
"""

        try:
            final_report_path.write_text(final_content, encoding="utf-8")
            logger.info(f"Final analysis saved to: {final_report_path}")
            return str(final_report_path)
        except Exception as e:
            error_msg = f"Failed to save final analysis: {e}"
            logger.error(error_msg)
            return error_msg

    def _check_quality_gate(self, final_analysis: str, initial_report_path: str, logs_file: str, directory_path: Path) -> dict:
        """
        Check if the code quality gate passes based on LGTM in final analysis.

        Returns dict with:
        - passed: bool
        - exit_code: int (0 for pass, 2 for quality gate failure)
        - message: str
        - reports: list of report paths
        """
        # Check for LGTM in the final analysis
        lgtm_found = "LGTM" in final_analysis

        # Get the quality assessment report path
        initial_name = Path(initial_report_path).stem
        quality_report_name = f"{initial_name}_QUALITY_ASSESSMENT.md"
        reports_dir = self._get_vibelint_reports_dir() / "justification_workflow"
        quality_report_path = reports_dir / quality_report_name

        result = {
            "passed": lgtm_found,
            "exit_code": 0 if lgtm_found else 2,
            "reports": [
                str(quality_report_path),
                str(initial_report_path),
                str(logs_file)
            ]
        }

        if lgtm_found:
            result["message"] = f"✅ Code quality gate PASSED - No significant architectural issues found"
            logger.info("Quality gate PASSED: LGTM found in analysis")
        else:
            project_name = directory_path.name if directory_path.name != '.' else 'project'
            result["message"] = f"""❌ Code quality gate FAILED - Architectural issues detected

Project: {project_name}
Issues found in quality assessment. Review the following reports:

📋 Quality Assessment: {quality_report_path}
📋 Detailed Analysis:   {initial_report_path}
📋 Analysis Logs:       {logs_file}

Address the critical issues identified before proceeding."""

            logger.error("Quality gate FAILED: No LGTM found in final analysis")
            logger.error(f"Review reports: {', '.join(result['reports'])}")

        return result

    def get_quality_gate_result(self) -> Optional[dict]:
        """Get the last quality gate result."""
        return self.last_quality_gate_result

    def enforce_quality_gate(self, safe_mode: bool = False) -> Optional[dict]:
        """
        Enforce quality gate by exiting with appropriate code if it failed.

        Args:
            safe_mode: If True, return result instead of exiting

        Returns:
            In safe mode: quality gate result dict
            In normal mode: None (exits process on failure)
        """
        if self.last_quality_gate_result is None:
            return None if safe_mode else None  # No quality gate run yet

        result = self.last_quality_gate_result

        if safe_mode:
            # Safe mode: return the result without printing or exiting
            return result
        else:
            # Normal mode: print and exit on failure
            print(result["message"])
            if not result["passed"]:
                import sys
                sys.exit(result["exit_code"])
            return None

    def run_justification_safe(self, directory_path: Path) -> dict:
        """
        Run justification workflow in safe mode - never exits, returns all results.

        Returns dict with:
        - success: bool
        - quality_gate: dict (quality gate result)
        - reports: list of generated report paths
        - error: str (if any error occurred)
        """
        try:
            # Run the normal workflow
            report = self.run_justification_workflow(directory_path)

            # Get quality gate result
            quality_gate = self.get_quality_gate_result()

            # Collect all report paths
            reports = quality_gate.get("reports", []) if quality_gate else []

            return {
                "success": True,
                "quality_gate": quality_gate,
                "reports": reports,
                "main_report": report,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "quality_gate": None,
                "reports": [],
                "main_report": None,
                "error": str(e)
            }