#!/usr/bin/env python3
"""
File and Method Justification Analysis Workflow

A comprehensive workflow that analyzes every file and method to justify their existence,
then uses embeddings to find similar justifications and identify redundancies.

This is implemented as a workflow, not validators, because it requires:
1. Cross-file analysis and coordination
2. State collection across multiple files
3. Embedding generation and similarity analysis
4. Multi-phase processing (collection → analysis → reporting)

This file now serves as a legacy wrapper around the new modular JustificationEngineV2.
For direct justification analysis, use the JustificationEngineV2 instead.
"""

import ast
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. Install with: pip install sentence-transformers"
    )


@dataclass
class FileJustification:
    """Represents a file's existence justification."""

    file_path: str
    primary_purpose: str
    secondary_purposes: List[str]
    method_count: int
    class_count: int
    import_dependencies: List[str]
    exported_symbols: List[str]
    module_docstring: Optional[str]
    complexity_score: int
    purpose_embedding: Optional[List[float]] = None


@dataclass
class MethodJustification:
    """Represents a method's existence justification."""

    file_path: str
    method_name: str
    line_number: int
    docstring: Optional[str]
    complexity_score: int
    unique_functionality: str
    dependencies: List[str]
    return_type_hint: Optional[str]
    is_private: bool
    purpose_embedding: Optional[List[float]] = None


@dataclass
class RedundancyCluster:
    """Represents a cluster of redundant files or methods."""

    cluster_type: str  # 'file' or 'method'
    similarity_score: float
    items: List[str]  # file paths or method signatures
    common_purpose: str
    recommendation: str


class JustificationAnalysisWorkflow:
    """Workflow for comprehensive justification analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.similarity_threshold = self.config.get("similarity_threshold", 0.85)
        self.min_complexity_for_analysis = self.config.get("min_complexity", 2)

        # Initialize embedding model if available
        self.model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")

        # Data collection
        self.file_justifications: Dict[str, FileJustification] = {}
        self.method_justifications: List[MethodJustification] = []
        self.processed_files: Set[str] = set()

    def analyze_file(self, file_path: Path, content: str = None) -> Dict[str, Any]:
        """Analyze a single file for justification. This is the deterministic part."""

        # Determine if this is a Python file or other type
        if file_path.suffix == ".py" and content is not None:
            return self._analyze_python_file(file_path, content)
        else:
            return self._analyze_non_python_file(file_path)

    def _analyze_python_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Analyze a Python file for justification."""
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.debug(f"Syntax error in {file_path}: {e}")
            return {"error": f"Syntax error: {e}"}

        # Analyze file-level justification
        file_justification = self._analyze_file_justification(file_path, tree, content)

        # Analyze method-level justifications
        method_justifications = self._analyze_methods_in_file(file_path, tree, content)

        # Store for cross-file analysis
        self.file_justifications[str(file_path)] = file_justification
        self.method_justifications.extend(method_justifications)
        self.processed_files.add(str(file_path))

        return {
            "file_justification": asdict(file_justification),
            "method_justifications": [asdict(m) for m in method_justifications],
            "analysis_quality": self._assess_analysis_quality(
                file_justification, method_justifications
            ),
        }

    def _analyze_non_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a non-Python file for justification."""
        file_justification = self._analyze_generic_file_justification(file_path)

        # Store for cross-file analysis
        self.file_justifications[str(file_path)] = file_justification
        self.processed_files.add(str(file_path))

        return {
            "file_justification": asdict(file_justification),
            "method_justifications": [],
            "analysis_quality": self._assess_generic_file_quality(file_justification),
        }

    def _analyze_file_justification(
        self, file_path: Path, tree: ast.AST, content: str
    ) -> FileJustification:
        """Analyze what justifies this file's existence."""

        # Extract module docstring
        module_docstring = ast.get_docstring(tree)

        # Count structural elements
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        imports = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]

        # Infer primary purpose
        primary_purpose = self._infer_primary_purpose(
            file_path, classes, functions, module_docstring
        )

        # Identify secondary purposes
        secondary_purposes = self._identify_secondary_purposes(classes, functions, content)

        # Extract dependencies and exports
        import_dependencies = self._extract_import_dependencies(imports)
        exported_symbols = self._extract_exported_symbols(tree)

        # Calculate complexity
        complexity_score = len(functions) + len(classes) * 2 + len(imports)

        # Generate embedding for purpose if model available
        purpose_embedding = None
        if self.model:
            purpose_text = self._create_purpose_text(
                file_path, primary_purpose, secondary_purposes, module_docstring
            )
            if purpose_text:
                try:
                    embedding = self.model.encode([purpose_text])[0]
                    purpose_embedding = embedding.tolist()
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for {file_path}: {e}")

        return FileJustification(
            file_path=str(file_path),
            primary_purpose=primary_purpose,
            secondary_purposes=secondary_purposes,
            method_count=len(functions),
            class_count=len(classes),
            import_dependencies=import_dependencies,
            exported_symbols=exported_symbols,
            module_docstring=module_docstring,
            complexity_score=complexity_score,
            purpose_embedding=purpose_embedding,
        )

    def _analyze_methods_in_file(
        self, file_path: Path, tree: ast.AST, content: str
    ) -> List[MethodJustification]:
        """Analyze all methods in a file for their justification."""
        methods = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                method_justification = self._analyze_single_method(file_path, node, content)
                if method_justification:
                    methods.append(method_justification)

        return methods

    def _analyze_single_method(
        self, file_path: Path, node: ast.FunctionDef, content: str
    ) -> Optional[MethodJustification]:
        """Analyze a single method's justification for existence."""

        # Calculate complexity
        complexity = self._calculate_method_complexity(node)

        # Skip trivial methods unless configured otherwise
        if complexity < self.min_complexity_for_analysis and not node.name.startswith("__"):
            return None

        # Extract method information
        docstring = ast.get_docstring(node)
        dependencies = self._extract_method_dependencies(node)
        unique_functionality = self._describe_unique_functionality(node, docstring)
        return_type_hint = self._extract_return_type_hint(node)
        is_private = node.name.startswith("_")

        # Generate embedding for method purpose
        purpose_embedding = None
        if self.model:
            purpose_text = self._create_method_purpose_text(
                node.name, docstring, unique_functionality
            )
            if purpose_text:
                try:
                    embedding = self.model.encode([purpose_text])[0]
                    purpose_embedding = embedding.tolist()
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for {file_path}:{node.name}: {e}")

        return MethodJustification(
            file_path=str(file_path),
            method_name=node.name,
            line_number=node.lineno,
            docstring=docstring,
            complexity_score=complexity,
            unique_functionality=unique_functionality,
            dependencies=dependencies,
            return_type_hint=return_type_hint,
            is_private=is_private,
            purpose_embedding=purpose_embedding,
        )

    def _infer_primary_purpose(
        self,
        file_path: Path,
        classes: List[ast.ClassDef],
        functions: List[ast.FunctionDef],
        docstring: Optional[str],
    ) -> str:
        """Infer the primary purpose of a file."""

        filename = file_path.stem

        # Check for common patterns
        if filename.endswith("_test") or filename.startswith("test_"):
            return "testing"
        elif filename in ["__init__", "main", "__main__"]:
            return "module_initialization" if filename == "__init__" else "application_entry_point"
        elif filename.endswith("_config") or "config" in filename:
            return "configuration"
        elif filename.endswith("_utils") or filename == "utils":
            return "utilities"
        elif filename.endswith("_types") or filename == "types":
            return "type_definitions"
        elif len(classes) > len(functions) and classes:
            return "class_definitions"
        elif len(functions) > len(classes) and functions:
            return "functional_operations"
        elif docstring:
            return self._extract_purpose_from_docstring(docstring)
        else:
            return "unclear_purpose"

    def _identify_secondary_purposes(
        self, classes: List[ast.ClassDef], functions: List[ast.FunctionDef], content: str
    ) -> List[str]:
        """Identify secondary purposes beyond the primary file purpose."""
        secondary = []

        # Check for helper functions
        helper_funcs = [
            f for f in functions if f.name.startswith("_") and not f.name.startswith("__")
        ]
        if helper_funcs:
            secondary.append("helper_functions")

        # Check for error handling
        if "except" in content or "raise" in content:
            secondary.append("error_handling")

        # Check for constants
        if content.count("=") > len(functions) + len(classes):
            secondary.append("constant_definitions")

        # Check for decorators
        if "@" in content:
            secondary.append("decorators")

        return secondary

    def _calculate_method_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a method."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity += 1

        return complexity

    def _describe_unique_functionality(
        self, node: ast.FunctionDef, docstring: Optional[str]
    ) -> str:
        """Describe what makes this method unique."""

        if docstring:
            # Extract first sentence of docstring
            first_sentence = docstring.split(".")[0].strip()
            if first_sentence and len(first_sentence) > 10:
                return first_sentence

        # Analyze method body for patterns
        patterns = []

        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                patterns.append("returns_value")
            elif isinstance(child, ast.Yield):
                patterns.append("generator")
            elif isinstance(child, ast.Await):
                patterns.append("async_operation")
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id in ["print", "log"]:
                    patterns.append("logging")
                elif child.func.id in ["open", "read", "write"]:
                    patterns.append("file_io")

        base_desc = f"method_{node.name}"
        if patterns:
            base_desc += "_" + "_".join(patterns[:3])  # Limit to 3 patterns

        return base_desc

    def _analyze_generic_file_justification(self, file_path: Path) -> FileJustification:
        """Analyze what justifies a non-Python file's existence."""

        # Get file stats
        try:
            file_stats = file_path.stat()
            file_size = file_stats.st_size
        except (OSError, IOError):
            file_size = 0

        # Infer purpose from file type and location
        primary_purpose = self._infer_generic_file_purpose(file_path)
        secondary_purposes = self._identify_generic_secondary_purposes(file_path)

        # Try to read content for documentation/config files
        content_sample = self._get_content_sample(file_path)

        # Generate embedding if model available
        purpose_embedding = None
        if self.model and content_sample:
            purpose_text = self._create_generic_purpose_text(
                file_path, primary_purpose, content_sample
            )
            if purpose_text:
                try:
                    embedding = self.model.encode([purpose_text])[0]
                    purpose_embedding = embedding.tolist()
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for {file_path}: {e}")

        return FileJustification(
            file_path=str(file_path),
            primary_purpose=primary_purpose,
            secondary_purposes=secondary_purposes,
            method_count=0,
            class_count=0,
            import_dependencies=[],
            exported_symbols=[],
            module_docstring=content_sample[:200] if content_sample else None,
            complexity_score=self._calculate_file_complexity(file_path, file_size),
            purpose_embedding=purpose_embedding,
        )

    def _infer_generic_file_purpose(self, file_path: Path) -> str:
        """Infer the purpose of a non-Python file."""

        filename = file_path.name.lower()
        suffix = file_path.suffix.lower()
        parent_dir = file_path.parent.name.lower()

        # Configuration files
        if filename in [
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "pipfile",
            "poetry.lock",
            "package.json",
            "yarn.lock",
            "composer.json",
        ]:
            return "project_configuration"

        if filename in [
            ".gitignore",
            ".gitattributes",
            ".gitmodules",
            ".pre-commit-config.yaml",
            ".github",
            ".gitlab-ci.yml",
            "tox.ini",
            ".flake8",
            ".pylintrc",
        ]:
            return "development_tooling"

        if filename in ["dockerfile", "docker-compose.yml", "docker-compose.yaml", ".dockerignore"]:
            return "containerization"

        # Documentation
        if suffix in [".md", ".rst", ".txt"] and any(
            doc_word in filename
            for doc_word in [
                "readme",
                "changelog",
                "history",
                "license",
                "contributing",
                "install",
                "usage",
            ]
        ):
            return "documentation"

        if filename in ["license", "copying", "authors", "contributors", "notice"]:
            return "legal_compliance"

        # Data files
        if suffix in [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"]:
            if parent_dir in ["config", "configs", "settings"]:
                return "configuration_data"
            elif parent_dir in ["data", "datasets", "fixtures"]:
                return "test_data"
            else:
                return "structured_data"

        if suffix in [".csv", ".tsv", ".xlsx", ".parquet", ".h5", ".hdf5"]:
            return "dataset"

        if suffix in [".sql", ".db", ".sqlite", ".sqlite3"]:
            return "database_schema"

        # Templates and static files
        if suffix in [".html", ".htm", ".css", ".js", ".jsx", ".ts", ".tsx", ".vue"]:
            return "web_frontend"

        if suffix in [".j2", ".jinja", ".jinja2", ".template", ".tmpl"]:
            return "template"

        # CI/CD and deployment
        if (
            parent_dir in [".github", ".gitlab", "ci", "deploy", "deployment"]
            or "workflow" in filename
            or "pipeline" in filename
        ):
            return "cicd_automation"

        # Logs
        if suffix in [".log", ".logs"] or parent_dir in ["logs", "log"]:
            return "logs"

        # Tests
        if parent_dir in ["test", "tests", "testing"] or "test" in filename:
            return "test_fixtures"

        # Scripts
        if suffix in [".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd"]:
            return "automation_script"

        # Media/assets
        if suffix in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp"]:
            return "image_asset"

        if suffix in [".pdf", ".doc", ".docx", ".odt"]:
            return "document"

        # Fallback based on extension
        if suffix:
            return f"unknown_file_type_{suffix[1:]}"

        return "unknown_purpose"

    def _identify_generic_secondary_purposes(self, file_path: Path) -> List[str]:
        """Identify secondary purposes for generic files."""
        secondary = []

        # Check if it's in a hidden directory (development tooling)
        if any(part.startswith(".") for part in file_path.parts):
            secondary.append("hidden_development_file")

        # Check if it's a lock file (dependency management)
        if "lock" in file_path.name.lower():
            secondary.append("dependency_lock")

        # Check if it's executable
        try:
            if file_path.stat().st_mode & 0o111:  # Has execute permission
                secondary.append("executable")
        except (OSError, IOError):
            pass

        # Check if it's large (might be generated)
        try:
            if file_path.stat().st_size > 1024 * 1024:  # > 1MB
                secondary.append("large_file")
        except (OSError, IOError):
            pass

        return secondary

    def _get_content_sample(self, file_path: Path) -> Optional[str]:
        """Get a sample of file content for analysis."""
        try:
            # Only read text files, and only first 1KB
            if file_path.suffix.lower() in [
                ".md",
                ".txt",
                ".rst",
                ".yaml",
                ".yml",
                ".json",
                ".toml",
                ".ini",
                ".cfg",
                ".conf",
                ".xml",
                ".html",
            ]:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read(1024)  # First 1KB
        except (OSError, IOError, UnicodeDecodeError):
            pass
        return None

    def _create_generic_purpose_text(
        self, file_path: Path, primary_purpose: str, content_sample: str
    ) -> str:
        """Create text for embedding generation for generic files."""
        parts = [
            f"File: {file_path.name}",
            f"Type: {file_path.suffix}",
            f"Purpose: {primary_purpose}",
            f"Location: {file_path.parent.name}",
        ]

        if content_sample:
            # Extract meaningful content (first few lines)
            lines = content_sample.split("\n")[:5]
            content_preview = " ".join(line.strip() for line in lines if line.strip())
            if content_preview:
                parts.append(f"Content: {content_preview}")

        return " ".join(parts)

    def _calculate_file_complexity(self, file_path: Path, file_size: int) -> int:
        """Calculate complexity score for a generic file."""
        complexity = 0

        # Size-based complexity
        if file_size > 100 * 1024:  # > 100KB
            complexity += 3
        elif file_size > 10 * 1024:  # > 10KB
            complexity += 2
        elif file_size > 1024:  # > 1KB
            complexity += 1

        # Location-based complexity (deeper = more complex)
        depth = len(file_path.parts) - 1
        complexity += min(depth, 5)

        # Extension-based complexity
        if file_path.suffix.lower() in [".json", ".yaml", ".toml", ".xml"]:
            complexity += 2  # Structured data is more complex

        return complexity

    def _assess_generic_file_quality(self, file_justification: FileJustification) -> Dict[str, Any]:
        """Assess the quality of justification analysis for a generic file."""

        quality_score = 50  # Base score for non-Python files
        issues = []

        # Check for unclear purpose
        if file_justification.primary_purpose.startswith("unknown"):
            issues.append("unclear_file_purpose")
            quality_score -= 20

        # Check for appropriate location
        filename = Path(file_justification.file_path).name.lower()
        if (
            filename in ["readme.md", "license", "changelog.md"]
            and "root" not in str(file_justification.file_path).lower()
        ):
            issues.append("misplaced_project_file")
            quality_score -= 10

        # Check for configuration files without clear purpose
        if (
            "config" in file_justification.primary_purpose
            and not file_justification.module_docstring
        ):
            issues.append("undocumented_configuration")
            quality_score -= 15

        return {
            "quality_score": max(quality_score, 0),
            "issues": issues,
            "documentation_ratio": 1.0 if file_justification.module_docstring else 0.0,
            "complexity_assessment": (
                "high"
                if file_justification.complexity_score > 8
                else "medium" if file_justification.complexity_score > 4 else "low"
            ),
        }

    def _extract_method_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Extract what this method depends on."""
        dependencies = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                dependencies.append(child.func.id)
            elif isinstance(child, ast.Attribute):
                dependencies.append(child.attr)

        return list(set(dependencies))

    def _extract_return_type_hint(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type hint if present."""
        if node.returns:
            return ast.unparse(node.returns) if hasattr(ast, "unparse") else str(node.returns)
        return None

    def _extract_import_dependencies(self, imports: List[ast.stmt]) -> List[str]:
        """Extract import dependencies."""
        dependencies = []

        for imp in imports:
            if isinstance(imp, ast.Import):
                dependencies.extend([alias.name for alias in imp.names])
            elif isinstance(imp, ast.ImportFrom) and imp.module:
                dependencies.append(imp.module)

        return dependencies

    def _extract_exported_symbols(self, tree: ast.AST) -> List[str]:
        """Extract symbols that this file exports."""
        exports = []

        # Look for __all__ definition
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            exports.extend(
                                [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]
                            )

        # If no __all__, extract public functions and classes
        if not exports:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not node.name.startswith("_"):
                        exports.append(node.name)

        return exports

    def _extract_purpose_from_docstring(self, docstring: str) -> str:
        """Extract purpose from module docstring."""
        sentences = docstring.strip().split(".")
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if len(clean_sentence) > 10:
                return clean_sentence.lower().replace(" ", "_")
        return "documented_module"

    def _create_purpose_text(
        self,
        file_path: Path,
        primary_purpose: str,
        secondary_purposes: List[str],
        docstring: Optional[str],
    ) -> str:
        """Create text for embedding generation."""
        parts = [f"File: {file_path.name}", f"Purpose: {primary_purpose}"]

        if secondary_purposes:
            parts.append(f"Secondary: {', '.join(secondary_purposes)}")

        if docstring:
            parts.append(docstring[:200])  # Limit docstring length

        return " ".join(parts)

    def _create_method_purpose_text(
        self, method_name: str, docstring: Optional[str], unique_functionality: str
    ) -> str:
        """Create text for method embedding generation."""
        parts = [f"Method: {method_name}"]

        if docstring:
            parts.append(docstring[:100])  # Limit docstring length

        parts.append(f"Functionality: {unique_functionality}")

        return " ".join(parts)

    def _assess_analysis_quality(
        self,
        file_justification: FileJustification,
        method_justifications: List[MethodJustification],
    ) -> Dict[str, Any]:
        """Assess the quality of justification analysis for this file."""

        quality_score = 0
        issues = []

        # File-level assessment
        if file_justification.module_docstring:
            quality_score += 20
        else:
            issues.append("missing_module_docstring")

        if not file_justification.primary_purpose.startswith("unclear"):
            quality_score += 20
        else:
            issues.append("unclear_primary_purpose")

        if file_justification.exported_symbols:
            quality_score += 10
        else:
            issues.append("no_clear_exports")

        # Method-level assessment
        documented_methods = sum(1 for m in method_justifications if m.docstring)
        if method_justifications:
            doc_ratio = documented_methods / len(method_justifications)
            quality_score += int(doc_ratio * 30)

            if doc_ratio < 0.5:
                issues.append("insufficient_method_documentation")

        # Complexity assessment
        if file_justification.complexity_score > 50:
            issues.append("high_complexity_file")

        complex_methods = sum(1 for m in method_justifications if m.complexity_score > 10)
        if complex_methods > len(method_justifications) * 0.3:
            issues.append("many_complex_methods")

        return {
            "quality_score": min(quality_score, 100),
            "issues": issues,
            "documentation_ratio": (
                documented_methods / len(method_justifications) if method_justifications else 0
            ),
            "complexity_assessment": (
                "high"
                if file_justification.complexity_score > 50
                else "medium" if file_justification.complexity_score > 20 else "low"
            ),
        }

    def find_redundancies(self) -> List[RedundancyCluster]:
        """Find redundancies across all analyzed files and methods."""

        clusters = []

        # Find file redundancies
        if self.model:
            clusters.extend(self._find_embedding_redundancies())

        clusters.extend(self._find_lexical_redundancies())

        return clusters

    def _find_embedding_redundancies(self) -> List[RedundancyCluster]:
        """Find redundancies using semantic embeddings."""

        clusters = []

        # File-level redundancies
        file_embeddings = []
        file_paths = []

        for file_path, justification in self.file_justifications.items():
            if justification.purpose_embedding:
                file_embeddings.append(justification.purpose_embedding)
                file_paths.append(file_path)

        if len(file_embeddings) > 1:
            file_clusters = self._cluster_by_similarity(file_embeddings, file_paths, "file")
            clusters.extend(file_clusters)

        # Method-level redundancies
        method_embeddings = []
        method_signatures = []

        for method in self.method_justifications:
            if method.purpose_embedding:
                method_embeddings.append(method.purpose_embedding)
                method_signatures.append(
                    f"{method.file_path}:{method.line_number}:{method.method_name}"
                )

        if len(method_embeddings) > 1:
            method_clusters = self._cluster_by_similarity(
                method_embeddings, method_signatures, "method"
            )
            clusters.extend(method_clusters)

        return clusters

    def _cluster_by_similarity(
        self, embeddings: List[List[float]], identifiers: List[str], cluster_type: str
    ) -> List[RedundancyCluster]:
        """Cluster items by embedding similarity."""

        clusters = []
        embeddings_array = np.array(embeddings)

        # Calculate cosine similarity matrix
        norms = np.linalg.norm(embeddings_array, axis=1)
        similarity_matrix = np.dot(embeddings_array, embeddings_array.T) / np.outer(norms, norms)

        used_indices = set()

        for i in range(len(embeddings)):
            if i in used_indices:
                continue

            # Find similar items
            similar_indices = []
            for j in range(i + 1, len(embeddings)):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    similar_indices.append(j)

            if similar_indices:
                cluster_items = [identifiers[i]] + [identifiers[j] for j in similar_indices]
                avg_similarity = np.mean([similarity_matrix[i, j] for j in similar_indices])

                # Extract common purpose
                common_purpose = self._extract_common_purpose_from_cluster(
                    cluster_items, cluster_type
                )

                # Generate recommendation
                recommendation = self._generate_redundancy_recommendation(
                    cluster_items, cluster_type, avg_similarity
                )

                cluster = RedundancyCluster(
                    cluster_type=cluster_type,
                    similarity_score=avg_similarity,
                    items=cluster_items,
                    common_purpose=common_purpose,
                    recommendation=recommendation,
                )

                clusters.append(cluster)

                used_indices.add(i)
                used_indices.update(similar_indices)

        return clusters

    def _find_lexical_redundancies(self) -> List[RedundancyCluster]:
        """Find redundancies using lexical analysis (fallback when no embeddings)."""

        clusters = []

        # Group files by primary purpose
        purpose_groups = defaultdict(list)
        for file_path, justification in self.file_justifications.items():
            if not justification.primary_purpose.startswith("unclear"):
                purpose_groups[justification.primary_purpose].append(file_path)

        # Report groups with multiple files
        for purpose, file_paths in purpose_groups.items():
            if len(file_paths) > 1:
                cluster = RedundancyCluster(
                    cluster_type="file",
                    similarity_score=1.0,  # Exact lexical match
                    items=file_paths,
                    common_purpose=purpose,
                    recommendation=f"Consider consolidating files with purpose '{purpose}'",
                )
                clusters.append(cluster)

        # Group methods by functionality description
        functionality_groups = defaultdict(list)
        for method in self.method_justifications:
            functionality_groups[method.unique_functionality].append(
                f"{method.file_path}:{method.line_number}:{method.method_name}"
            )

        # Report groups with multiple methods across different files
        for functionality, method_sigs in functionality_groups.items():
            if len(method_sigs) > 1:
                # Check if methods are in different files
                files = set(sig.split(":")[0] for sig in method_sigs)
                if len(files) > 1:
                    cluster = RedundancyCluster(
                        cluster_type="method",
                        similarity_score=1.0,  # Exact lexical match
                        items=method_sigs,
                        common_purpose=functionality,
                        recommendation=f"Consider extracting common functionality: {functionality}",
                    )
                    clusters.append(cluster)

        return clusters

    def _extract_common_purpose_from_cluster(self, items: List[str], cluster_type: str) -> str:
        """Extract common purpose from a cluster of similar items."""

        if cluster_type == "file":
            # For files, use the primary purpose of the first file
            first_file = items[0]
            if first_file in self.file_justifications:
                return self.file_justifications[first_file].primary_purpose

        elif cluster_type == "method":
            # For methods, find common functionality
            method_functionalities = []
            for item in items:
                file_path, line_num, method_name = item.split(":", 2)
                for method in self.method_justifications:
                    if method.file_path == file_path and method.method_name == method_name:
                        method_functionalities.append(method.unique_functionality)
                        break

            if method_functionalities:
                # Find common words
                all_words = []
                for func in method_functionalities:
                    all_words.extend(func.split("_"))

                word_counts = defaultdict(int)
                for word in all_words:
                    word_counts[word] += 1

                common_words = [
                    word
                    for word, count in word_counts.items()
                    if count >= len(method_functionalities) // 2
                ]

                return "_".join(common_words[:3]) if common_words else "similar_functionality"

        return "unknown_common_purpose"

    def _generate_redundancy_recommendation(
        self, items: List[str], cluster_type: str, similarity_score: float
    ) -> str:
        """Generate recommendation for addressing redundancy."""

        if similarity_score > 0.95:
            action = "merge or eliminate duplicate"
        elif similarity_score > 0.85:
            action = "consolidate similar"
        else:
            action = "review related"

        if cluster_type == "file":
            return f"{action.capitalize()} files: consider merging into single file"
        else:
            return f"{action.capitalize()} methods: consider extracting common functionality"

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive justification analysis report."""

        redundancies = self.find_redundancies()

        # Calculate statistics
        total_files = len(self.file_justifications)
        total_methods = len(self.method_justifications)

        files_with_unclear_purpose = sum(
            1 for f in self.file_justifications.values() if f.primary_purpose.startswith("unclear")
        )

        methods_without_docs = sum(
            1 for m in self.method_justifications if not m.docstring and m.complexity_score > 3
        )

        # Redundancy statistics
        file_redundancies = [r for r in redundancies if r.cluster_type == "file"]
        method_redundancies = [r for r in redundancies if r.cluster_type == "method"]

        return {
            "summary": {
                "total_files_analyzed": total_files,
                "total_methods_analyzed": total_methods,
                "files_with_unclear_purpose": files_with_unclear_purpose,
                "methods_without_documentation": methods_without_docs,
                "embedding_analysis_enabled": self.model is not None,
            },
            "justification_quality": {
                "files_needing_attention": files_with_unclear_purpose,
                "methods_needing_documentation": methods_without_docs,
                "documentation_coverage": (
                    (total_methods - methods_without_docs) / total_methods
                    if total_methods > 0
                    else 0
                ),
            },
            "redundancy_analysis": {
                "file_redundancy_clusters": len(file_redundancies),
                "method_redundancy_clusters": len(method_redundancies),
                "total_redundant_files": sum(len(r.items) for r in file_redundancies),
                "total_redundant_methods": sum(len(r.items) for r in method_redundancies),
            },
            "recommendations": self._generate_recommendations(redundancies),
            "detailed_redundancies": [asdict(r) for r in redundancies],
        }

    def _generate_recommendations(
        self, redundancies: List[RedundancyCluster]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""

        recommendations = []

        # High-priority redundancies
        high_priority = [r for r in redundancies if r.similarity_score > 0.9]
        if high_priority:
            recommendations.append(
                {
                    "priority": "high",
                    "action": "immediate_consolidation",
                    "description": f"Found {len(high_priority)} clusters with >90% similarity requiring immediate attention",
                    "affected_items": sum(len(r.items) for r in high_priority),
                }
            )

        # Files with unclear purpose
        unclear_files = [
            f for f in self.file_justifications.values() if f.primary_purpose.startswith("unclear")
        ]
        if unclear_files:
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "clarify_purpose",
                    "description": f"Add clear documentation to {len(unclear_files)} files with unclear purpose",
                    "affected_items": len(unclear_files),
                }
            )

        # Complex methods without documentation
        complex_undocumented = [
            m for m in self.method_justifications if not m.docstring and m.complexity_score > 5
        ]
        if complex_undocumented:
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "document_complex_methods",
                    "description": f"Add documentation to {len(complex_undocumented)} complex methods",
                    "affected_items": len(complex_undocumented),
                }
            )

        return recommendations

    def export_data(self, output_dir: Path) -> None:
        """Export analysis data for external processing."""

        output_dir.mkdir(parents=True, exist_ok=True)

        # Export file justifications
        with open(output_dir / "file_justifications.json", "w") as f:
            json.dump([asdict(fj) for fj in self.file_justifications.values()], f, indent=2)

        # Export method justifications
        with open(output_dir / "method_justifications.json", "w") as f:
            json.dump([asdict(mj) for mj in self.method_justifications], f, indent=2)

        # Export redundancy analysis
        redundancies = self.find_redundancies()
        with open(output_dir / "redundancy_clusters.json", "w") as f:
            json.dump([asdict(r) for r in redundancies], f, indent=2)

        # Export comprehensive report
        report = self.generate_report()
        with open(output_dir / "justification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported justification analysis data to {output_dir}")


# New simple wrapper for backward compatibility
def run_justification_workflow(directory_path: Path, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Run the comprehensive justification workflow using the new modular engine.

    This is a simple wrapper for backward compatibility. For new code, use JustificationEngine directly.
    """
    try:
        from .justification.engine import JustificationEngine

        engine = JustificationEngine(config)
        result = engine.run_justification_workflow(directory_path)

        # Return the comprehensive report
        return result

    except ImportError as e:
        logger.error(f"Failed to import JustificationEngineV2: {e}")
        logger.info("Falling back to legacy JustificationAnalysisWorkflow")

        # Fallback to the old workflow (keeping existing functionality)
        workflow = JustificationAnalysisWorkflow(config)

        # Process all Python files in the directory
        for file_path in directory_path.rglob("*.py"):
            try:
                content = file_path.read_text(encoding='utf-8')
                workflow.analyze_file(file_path, content)
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")

        # Generate and return report
        report = workflow.generate_report()
        return json.dumps(report, indent=2)
