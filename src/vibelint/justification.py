"""
Justification Engine for vibelint.

Core justification workflow that uses static analysis and minimal LLM calls
to justify code decisions and identify redundancies.

This is the foundation of vibelint's software quality approach.
"""

import ast
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .llm import LLMRequest, LLMRole, create_llm_manager

logger = logging.getLogger(__name__)

__all__ = ["JustificationEngine", "JustificationResult", "CodeJustification"]


@dataclass
class CodeJustification:
    """A single justification for a piece of code."""

    file_path: str
    element_type: str  # 'file', 'method', 'class'
    element_name: str
    line_number: int
    justification: str
    confidence: float
    dependencies: List[str]
    complexity_score: int


@dataclass
class JustificationResult:
    """Result of justification analysis."""

    file_path: str
    justifications: List[CodeJustification]
    redundancies_found: List[str]
    recommendations: List[str]
    quality_score: float


class JustificationEngine:
    """
    Core justification engine using static analysis and minimal LLM usage.

    Philosophy:
    1. Static analysis for structure and dependencies
    2. Fast LLM for simple yes/no decisions on method similarity
    3. Orchestrator LLM for complex cross-file analysis only when needed
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.llm_manager = create_llm_manager(config or {})

        # Cache for static analysis results
        self._import_cache: Dict[str, Set[str]] = {}
        self._method_cache: Dict[str, List[Dict[str, Any]]] = {}

        # Logging infrastructure
        self._session_id = f"justification_{int(time.time())}"
        self._llm_call_logs: List[Dict[str, Any]] = []
        self._create_log_directory()

    def justify_file(self, file_path: Path, content: str) -> JustificationResult:
        """Justify a single file's existence and structure."""

        if not file_path.suffix == ".py":
            return self._justify_non_python_file(file_path)

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return JustificationResult(
                file_path=str(file_path),
                justifications=[],
                redundancies_found=[],
                recommendations=[f"Fix syntax error: {e}"],
                quality_score=0.0
            )

        # Static analysis first
        imports = self._extract_imports(tree)
        methods = self._extract_methods(tree)
        classes = self._extract_classes(tree)

        # Cache results
        self._import_cache[str(file_path)] = imports
        self._method_cache[str(file_path)] = methods

        # Generate justifications
        justifications = []

        # File-level justification
        file_justification = self._justify_file_purpose(file_path, tree, content)
        justifications.append(file_justification)

        # Method-level justifications
        for method_info in methods:
            if method_info["complexity"] >= 2:  # Only justify non-trivial methods
                method_justification = self._justify_method(file_path, method_info, tree)
                justifications.append(method_justification)

        # Class-level justifications
        for class_info in classes:
            class_justification = self._justify_class(file_path, class_info, tree)
            justifications.append(class_justification)

        # Find redundancies (static analysis only)
        redundancies = self._find_local_redundancies(methods, classes)

        # Generate recommendations
        recommendations = self._generate_recommendations(justifications, redundancies)

        # Calculate quality score
        quality_score = self._calculate_quality_score(justifications)

        return JustificationResult(
            file_path=str(file_path),
            justifications=justifications,
            redundancies_found=redundancies,
            recommendations=recommendations,
            quality_score=quality_score
        )

    def justify_method_comparison(self, method1_path: str, method1_name: str,
                                  method2_path: str, method2_name: str) -> Dict[str, Any]:
        """Use fast LLM to compare two methods for similarity (yes/no decision)."""

        if not self.llm_manager or not self.llm_manager.is_llm_available(LLMRole.FAST):
            logger.warning("Fast LLM not available for method comparison")
            return {"similar": False, "confidence": 0.0, "reasoning": "LLM unavailable"}

        # Get method content from cache
        method1_info = self._get_method_from_cache(method1_path, method1_name)
        method2_info = self._get_method_from_cache(method2_path, method2_name)

        if not method1_info or not method2_info:
            return {"similar": False, "confidence": 0.0, "reasoning": "Method not found"}

        # Create focused prompt for fast LLM (750 token limit)
        prompt = f"""Compare these two methods for functional similarity:

Method 1: {method1_name}
{method1_info.get('signature', 'Unknown signature')}
Purpose: {method1_info.get('purpose', 'No docstring')}

Method 2: {method2_name}
{method2_info.get('signature', 'Unknown signature')}
Purpose: {method2_info.get('purpose', 'No docstring')}

Answer: {{"similar": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

        request = LLMRequest(
            content=prompt,
            task_type="method_comparison",
            max_tokens=100,  # Short response
            temperature=0.1   # Deterministic
        )

        try:
            start_time = time.time()
            response = self.llm_manager.process_request(request)
            duration = time.time() - start_time

            # Log the LLM interaction
            self._log_llm_call({
                "operation": "method_comparison",
                "methods": [f"{method1_path}:{method1_name}", f"{method2_path}:{method2_name}"],
                "prompt": prompt,
                "response_raw": response.content,
                "llm_role": "fast",
                "duration_seconds": duration,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            # Parse structured JSON response
            result = json.loads(response.content)
            return result
        except Exception as e:
            logger.error(f"Fast LLM method comparison failed: {e}")
            # Log the failure
            self._log_llm_call({
                "operation": "method_comparison",
                "methods": [f"{method1_path}:{method1_name}", f"{method2_path}:{method2_name}"],
                "prompt": prompt,
                "error": str(e),
                "llm_role": "fast",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            return {"similar": False, "confidence": 0.0, "reasoning": f"Error: {e}"}

    def _justify_non_python_file(self, file_path: Path) -> JustificationResult:
        """Justify non-Python files using file system analysis."""

        purpose = self._infer_file_purpose_from_path(file_path)

        justification = CodeJustification(
            file_path=str(file_path),
            element_type="file",
            element_name=file_path.name,
            line_number=1,
            justification=purpose,
            confidence=0.8,
            dependencies=[],
            complexity_score=1
        )

        return JustificationResult(
            file_path=str(file_path),
            justifications=[justification],
            redundancies_found=[],
            recommendations=[],
            quality_score=0.8
        )

    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all imports from AST (static analysis)."""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

        return imports

    def _extract_methods(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract method information from AST (static analysis)."""
        methods = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                method_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "signature": self._extract_signature(node),
                    "purpose": ast.get_docstring(node) or "No docstring",
                    "complexity": self._calculate_complexity(node),
                    "is_private": node.name.startswith("_"),
                    "dependencies": self._extract_method_dependencies(node)
                }
                methods.append(method_info)

        return methods

    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class information from AST (static analysis)."""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "purpose": ast.get_docstring(node) or "No docstring",
                    "method_count": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    "dependencies": self._extract_class_dependencies(node)
                }
                classes.append(class_info)

        return classes

    def _justify_file_purpose(self, file_path: Path, tree: ast.AST, content: str) -> CodeJustification:
        """Justify file's existence using static analysis."""

        module_doc = ast.get_docstring(tree)

        # Analyze structure
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        # Infer purpose from filename and structure
        purpose = self._infer_file_purpose(file_path, functions, classes, module_doc)

        complexity = len(functions) + len(classes) * 2
        confidence = 0.9 if module_doc else 0.6

        return CodeJustification(
            file_path=str(file_path),
            element_type="file",
            element_name=file_path.name,
            line_number=1,
            justification=purpose,
            confidence=confidence,
            dependencies=list(self._import_cache.get(str(file_path), set())),
            complexity_score=complexity
        )

    def _justify_method(self, file_path: Path, method_info: Dict[str, Any], tree: ast.AST) -> CodeJustification:
        """Justify method's existence using static analysis."""

        purpose = method_info["purpose"]
        if purpose == "No docstring":
            # Infer from method name and structure
            purpose = f"Method {method_info['name']} - inferred from implementation"

        return CodeJustification(
            file_path=str(file_path),
            element_type="method",
            element_name=method_info["name"],
            line_number=method_info["line_number"],
            justification=purpose,
            confidence=0.8 if method_info["purpose"] != "No docstring" else 0.4,
            dependencies=method_info["dependencies"],
            complexity_score=method_info["complexity"]
        )

    def _justify_class(self, file_path: Path, class_info: Dict[str, Any], tree: ast.AST) -> CodeJustification:
        """Justify class's existence using static analysis."""

        purpose = class_info["purpose"]
        if purpose == "No docstring":
            purpose = f"Class {class_info['name']} with {class_info['method_count']} methods"

        return CodeJustification(
            file_path=str(file_path),
            element_type="class",
            element_name=class_info["name"],
            line_number=class_info["line_number"],
            justification=purpose,
            confidence=0.8 if class_info["purpose"] != "No docstring" else 0.5,
            dependencies=class_info["dependencies"],
            complexity_score=class_info["method_count"]
        )

    def _find_local_redundancies(self, methods: List[Dict[str, Any]],
                                classes: List[Dict[str, Any]]) -> List[str]:
        """Find potential redundancies within a single file (static analysis)."""
        redundancies = []

        # Check for similar method names
        method_names = [m["name"] for m in methods]
        for i, name1 in enumerate(method_names):
            for j, name2 in enumerate(method_names[i+1:], i+1):
                if self._names_similar(name1, name2):
                    redundancies.append(f"Similar method names: {name1}, {name2}")

        # Check for classes with single methods (possible over-engineering)
        for class_info in classes:
            if class_info["method_count"] == 1:
                redundancies.append(f"Single-method class: {class_info['name']}")

        return redundancies

    def _generate_recommendations(self, justifications: List[CodeJustification],
                                 redundancies: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Check documentation coverage
        undocumented = [j for j in justifications if j.confidence < 0.5]
        if undocumented:
            recommendations.append(f"Add documentation to {len(undocumented)} elements")

        # Check complexity
        complex_elements = [j for j in justifications if j.complexity_score > 10]
        if complex_elements:
            recommendations.append(f"Consider breaking down {len(complex_elements)} complex elements")

        # Add redundancy recommendations
        if redundancies:
            recommendations.append("Review potential redundancies for consolidation")

        return recommendations

    def _calculate_quality_score(self, justifications: List[CodeJustification]) -> float:
        """Calculate overall quality score for the file."""
        if not justifications:
            return 0.0

        # Average confidence weighted by complexity
        total_weight = 0
        weighted_confidence = 0

        for just in justifications:
            weight = max(just.complexity_score, 1)
            weighted_confidence += just.confidence * weight
            total_weight += weight

        return weighted_confidence / total_weight if total_weight > 0 else 0.0

    # Helper methods for static analysis
    def _extract_signature(self, node: ast.FunctionDef) -> str:
        """Extract method signature."""
        args = [arg.arg for arg in node.args.args]
        return f"{node.name}({', '.join(args)})"

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity

    def _extract_method_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Extract what the method depends on."""
        deps = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                deps.append(child.func.id)
        return list(set(deps))

    def _extract_class_dependencies(self, node: ast.ClassDef) -> List[str]:
        """Extract what the class depends on."""
        deps = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                deps.append(base.id)
        return deps

    def _infer_file_purpose(self, file_path: Path, functions: List[ast.FunctionDef],
                           classes: List[ast.ClassDef], module_doc: Optional[str]) -> str:
        """Infer file purpose from static analysis."""

        if module_doc:
            return f"Documented module: {module_doc.split('.')[0]}"

        filename = file_path.stem

        if filename.endswith("_test") or filename.startswith("test_"):
            return "Test module"
        elif filename == "__init__":
            return "Package initialization"
        elif filename in ["main", "__main__"]:
            return "Application entry point"
        elif len(classes) > len(functions):
            return f"Class definitions module ({len(classes)} classes)"
        elif len(functions) > 0:
            return f"Function definitions module ({len(functions)} functions)"
        else:
            return "Configuration or data module"

    def _infer_file_purpose_from_path(self, file_path: Path) -> str:
        """Infer non-Python file purpose from path and extension."""

        if file_path.name in ["README.md", "CHANGELOG.md", "LICENSE"]:
            return "Project documentation"
        elif file_path.suffix in [".toml", ".yaml", ".yml", ".json"]:
            return "Configuration file"
        elif file_path.suffix in [".sh", ".bat"]:
            return "Automation script"
        else:
            return f"Support file ({file_path.suffix})"

    def _names_similar(self, name1: str, name2: str) -> bool:
        """Check if two names are suspiciously similar."""
        # Simple heuristic: similar if they share >80% of characters
        if len(name1) < 3 or len(name2) < 3:
            return False

        common_chars = len(set(name1.lower()) & set(name2.lower()))
        max_chars = max(len(set(name1.lower())), len(set(name2.lower())))

        return common_chars / max_chars > 0.8

    def _get_method_from_cache(self, file_path: str, method_name: str) -> Optional[Dict[str, Any]]:
        """Get method info from cache."""
        methods = self._method_cache.get(file_path, [])
        for method in methods:
            if method["name"] == method_name:
                return method
        return None

    def _create_log_directory(self):
        """Create logging directory for justification outputs."""
        # Look for existing .vibelint-reports directory
        current_path = Path.cwd()
        while current_path.parent != current_path:
            reports_dir = current_path / ".vibelint-reports"
            if reports_dir.exists():
                self.log_dir = reports_dir / "justification"
                break
            current_path = current_path.parent
        else:
            # Fallback: create in current directory
            self.log_dir = Path.cwd() / ".vibelint-reports" / "justification"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Justification logs will be saved to: {self.log_dir}")

    def _log_llm_call(self, call_data: Dict[str, Any]):
        """Log an LLM call with full details."""
        self._llm_call_logs.append(call_data)

    def save_session_logs(self, file_path: str, result: JustificationResult):
        """Save comprehensive session logs including all LLM calls and final justification."""

        # Create session log with all details
        session_log = {
            "session_id": self._session_id,
            "file_analyzed": file_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "llm_calls": self._llm_call_logs,
            "final_justification": {
                "justifications": [asdict(j) for j in result.justifications],
                "redundancies_found": result.redundancies_found,
                "recommendations": result.recommendations,
                "quality_score": result.quality_score
            },
            "summary": {
                "total_llm_calls": len(self._llm_call_logs),
                "elements_analyzed": len(result.justifications),
                "static_analysis_only": len(self._llm_call_logs) == 0
            }
        }

        # Save detailed session log
        session_file = self.log_dir / f"{self._session_id}_detailed.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_log, f, indent=2, ensure_ascii=False)

        # Save human-readable summary
        summary_file = self.log_dir / f"{self._session_id}_summary.md"
        self._save_readable_summary(summary_file, session_log)

        logger.info(f"Justification session logs saved:")
        logger.info(f"  Detailed: {session_file}")
        logger.info(f"  Summary: {summary_file}")

        return session_file, summary_file

    def _save_readable_summary(self, summary_file: Path, session_log: Dict[str, Any]):
        """Save human-readable summary of justification analysis."""

        content = f"""# Justification Analysis Summary

**Session:** {session_log['session_id']}
**File:** {session_log['file_analyzed']}
**Timestamp:** {session_log['timestamp']}
**Quality Score:** {session_log['final_justification']['quality_score']:.1%}

## Analysis Summary

- **Elements Analyzed:** {session_log['summary']['elements_analyzed']}
- **LLM Calls Made:** {session_log['summary']['total_llm_calls']}
- **Analysis Type:** {'Static Analysis Only' if session_log['summary']['static_analysis_only'] else 'Static + LLM Analysis'}

## Code Justifications

"""

        for just_data in session_log['final_justification']['justifications']:
            confidence_emoji = "✅" if just_data['confidence'] > 0.7 else "⚠️" if just_data['confidence'] > 0.4 else "❌"
            content += f"""### {just_data['element_name']} ({just_data['element_type']})

{confidence_emoji} **Confidence:** {just_data['confidence']:.1%}
**Line:** {just_data['line_number']}
**Complexity:** {just_data['complexity_score']}

**Justification:** {just_data['justification']}

"""

        if session_log['final_justification']['redundancies_found']:
            content += "## ⚠️ Potential Redundancies\n\n"
            for redundancy in session_log['final_justification']['redundancies_found']:
                content += f"- {redundancy}\n"
            content += "\n"

        if session_log['final_justification']['recommendations']:
            content += "## 💡 Recommendations\n\n"
            for rec in session_log['final_justification']['recommendations']:
                content += f"- {rec}\n"
            content += "\n"

        if session_log['llm_calls']:
            content += "## 🤖 LLM Call Details\n\n"
            for i, call in enumerate(session_log['llm_calls'], 1):
                content += f"""### Call {i}: {call['operation']}

**LLM Role:** {call['llm_role']}
**Duration:** {call.get('duration_seconds', 'N/A')}s
**Timestamp:** {call['timestamp']}

**Prompt:**
```
{call['prompt'][:500]}{'...' if len(call['prompt']) > 500 else ''}
```

**Response:**
```
{call.get('response_raw', call.get('error', 'No response'))[:500]}{'...' if len(call.get('response_raw', '')) > 500 else ''}
```

---

"""

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(content)