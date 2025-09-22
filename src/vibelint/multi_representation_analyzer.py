"""
Multi-Representation Architecture Analyzer

Creates a unified view of software quality using three reinforcing representations:
1. Filesystem Representation - hierarchical structure, module organization
2. Vector Representation - semantic similarity, code patterns via embeddings
3. Graph Representation - dependencies, execution flow, data flow

This enables autonomous improvement by understanding software at multiple levels.
"""

import ast
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np


@dataclass
class SoftwareQualityMetrics:
    """Comprehensive software quality metrics across all representations."""

    # SOLID Principle Violations
    srp_violations: List[Dict[str, Any]] = field(default_factory=list)
    ocp_violations: List[Dict[str, Any]] = field(default_factory=list)
    lsp_violations: List[Dict[str, Any]] = field(default_factory=list)
    isp_violations: List[Dict[str, Any]] = field(default_factory=list)
    dip_violations: List[Dict[str, Any]] = field(default_factory=list)

    # Code Quality
    cyclomatic_complexity: Dict[str, int] = field(default_factory=dict)
    cognitive_complexity: Dict[str, int] = field(default_factory=dict)
    coupling_metrics: Dict[str, float] = field(default_factory=dict)
    cohesion_metrics: Dict[str, float] = field(default_factory=dict)

    # Pythonic Practices
    naming_violations: List[Dict[str, Any]] = field(default_factory=list)
    type_hint_coverage: float = 0.0
    docstring_coverage: float = 0.0
    error_handling_patterns: Dict[str, int] = field(default_factory=dict)

    # Architecture Quality
    module_organization_score: float = 0.0
    abstraction_quality: float = 0.0
    testability_score: float = 0.0

    # Cross-representation insights
    filesystem_vector_alignment: float = 0.0
    vector_graph_consistency: float = 0.0
    graph_filesystem_coherence: float = 0.0


@dataclass
class ImprovementOpportunity:
    """A specific improvement opportunity with context from all representations."""

    opportunity_id: str
    category: str  # "solid", "complexity", "pythonic", "architecture"
    severity: str  # "high", "medium", "low"

    # Filesystem context
    file_path: str
    line_number: int
    affected_modules: List[str]

    # Vector context
    similar_code_patterns: List[str]
    semantic_clusters: List[str]

    # Graph context
    dependency_impact: List[str]
    execution_criticality: float

    # Improvement suggestion
    current_state: str
    target_state: str
    implementation_steps: List[str]
    estimated_impact: Dict[str, float]
    risk_assessment: str


class FilesystemRepresentation:
    """Analyzes project structure and organization from filesystem perspective."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.module_tree = {}
        self.import_graph = nx.DiGraph()

    def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze filesystem organization for architectural quality."""
        analysis = {
            "module_hierarchy": self._build_module_hierarchy(),
            "package_organization": self._analyze_package_organization(),
            "import_structure": self._analyze_import_structure(),
            "naming_conventions": self._analyze_naming_conventions(),
            "file_size_distribution": self._analyze_file_sizes(),
            "directory_coupling": self._calculate_directory_coupling(),
        }

        return analysis

    def _build_module_hierarchy(self) -> Dict[str, Any]:
        """Build hierarchical representation of modules."""
        hierarchy = {}

        for python_file in self.project_root.rglob("*.py"):
            if python_file.name.startswith("."):
                continue

            relative_path = python_file.relative_to(self.project_root)
            parts = relative_path.parts

            current = hierarchy
            for part in parts[:-1]:  # Directories
                if part not in current:
                    current[part] = {"type": "directory", "children": {}}
                current = current[part]["children"]

            # File
            filename = parts[-1]
            current[filename] = {
                "type": "file",
                "path": str(python_file),
                "size_lines": len(python_file.read_text().splitlines()),
                "functions": self._extract_functions(python_file),
                "classes": self._extract_classes(python_file),
                "imports": self._extract_imports(python_file),
            }

        return hierarchy

    def _analyze_package_organization(self) -> Dict[str, Any]:
        """Analyze package organization quality."""
        packages = []

        for init_file in self.project_root.rglob("__init__.py"):
            package_dir = init_file.parent
            package_name = str(package_dir.relative_to(self.project_root))

            # Analyze package cohesion
            python_files = list(package_dir.glob("*.py"))
            if len(python_files) <= 1:
                continue

            cohesion_score = self._calculate_package_cohesion(python_files)

            packages.append(
                {
                    "name": package_name,
                    "file_count": len(python_files),
                    "cohesion_score": cohesion_score,
                    "has_clear_purpose": self._has_clear_package_purpose(package_dir),
                    "follows_naming_convention": self._follows_package_naming(package_name),
                }
            )

        return {
            "packages": packages,
            "average_cohesion": np.mean([p["cohesion_score"] for p in packages]) if packages else 0,
            "organization_violations": self._find_organization_violations(packages),
        }

    def _analyze_import_structure(self) -> Dict[str, Any]:
        """Analyze import dependencies for coupling analysis."""
        import_graph = nx.DiGraph()
        circular_imports = []

        for python_file in self.project_root.rglob("*.py"):
            if python_file.name.startswith("."):
                continue

            module_name = (
                str(python_file.relative_to(self.project_root)).replace(".py", "").replace("/", ".")
            )
            imports = self._extract_imports(python_file)

            for imported_module in imports:
                if imported_module.startswith("."):  # Relative import
                    continue

                import_graph.add_edge(module_name, imported_module)

        # Find circular dependencies
        try:
            cycles = list(nx.simple_cycles(import_graph))
            circular_imports = [cycle for cycle in cycles if len(cycle) > 1]
        except:
            pass

        return {
            "import_graph": import_graph,
            "circular_imports": circular_imports,
            "coupling_metrics": self._calculate_import_coupling(import_graph),
            "dependency_depth": self._calculate_dependency_depth(import_graph),
        }

    def _extract_functions(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract function definitions with metrics."""
        try:
            source = file_path.read_text()
            tree = ast.parse(source)
            functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(
                        {
                            "name": node.name,
                            "line_number": node.lineno,
                            "args_count": len(node.args.args),
                            "has_docstring": ast.get_docstring(node) is not None,
                            "has_type_hints": self._has_type_hints(node),
                            "cyclomatic_complexity": self._calculate_cyclomatic_complexity(node),
                        }
                    )

            return functions
        except Exception:
            return []

    def _extract_classes(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract class definitions with metrics."""
        try:
            source = file_path.read_text()
            tree = ast.parse(source)
            classes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]

                    classes.append(
                        {
                            "name": node.name,
                            "line_number": node.lineno,
                            "method_count": len(methods),
                            "has_docstring": ast.get_docstring(node) is not None,
                            "inheritance_depth": len(node.bases),
                            "follows_naming_convention": node.name[0].isupper(),
                        }
                    )

            return classes
        except Exception:
            return []

    def _extract_imports(self, file_path: Path) -> List[str]:
        """Extract import statements."""
        try:
            source = file_path.read_text()
            tree = ast.parse(source)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return imports
        except Exception:
            return []

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1

        return complexity

    def _has_type_hints(self, node: ast.FunctionDef) -> bool:
        """Check if function has type hints."""
        return node.returns is not None or any(arg.annotation is not None for arg in node.args.args)

    def _calculate_package_cohesion(self, python_files: List[Path]) -> float:
        """Calculate package cohesion based on shared dependencies."""
        if len(python_files) <= 1:
            return 1.0

        all_imports = set()
        file_imports = []

        for file_path in python_files:
            imports = set(self._extract_imports(file_path))
            file_imports.append(imports)
            all_imports.update(imports)

        if not all_imports:
            return 0.5  # Neutral score for packages with no external dependencies

        # Calculate Jaccard similarity between files
        similarities = []
        for i in range(len(file_imports)):
            for j in range(i + 1, len(file_imports)):
                intersection = len(file_imports[i] & file_imports[j])
                union = len(file_imports[i] | file_imports[j])
                if union > 0:
                    similarities.append(intersection / union)

        return np.mean(similarities) if similarities else 0.0

    def _has_clear_package_purpose(self, package_dir: Path) -> bool:
        """Check if package has a clear, single purpose."""
        # Simple heuristic: check if package name suggests single purpose
        package_name = package_dir.name.lower()

        purpose_indicators = [
            "utils",
            "helpers",
            "core",
            "models",
            "views",
            "controllers",
            "validators",
            "parsers",
            "analyzers",
            "reporters",
            "config",
            "tests",
            "fixtures",
            "cli",
            "api",
            "db",
            "auth",
        ]

        return any(indicator in package_name for indicator in purpose_indicators)

    def _follows_package_naming(self, package_name: str) -> bool:
        """Check if package follows Python naming conventions."""
        return (
            package_name.islower()
            and "_" not in package_name
            or package_name.replace("_", "").isalnum()
        )

    def _find_organization_violations(self, packages: List[Dict[str, Any]]) -> List[str]:
        """Find package organization violations."""
        violations = []

        for package in packages:
            if package["cohesion_score"] < 0.3:
                violations.append(f"Low cohesion in package {package['name']}")

            if package["file_count"] > 10:
                violations.append(
                    f"Package {package['name']} may be too large ({package['file_count']} files)"
                )

            if not package["has_clear_purpose"]:
                violations.append(f"Package {package['name']} lacks clear purpose")

        return violations

    def _calculate_import_coupling(self, import_graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate coupling metrics from import graph."""
        coupling = {}

        for node in import_graph.nodes():
            in_degree = import_graph.in_degree(node)  # Afferent coupling
            out_degree = import_graph.out_degree(node)  # Efferent coupling

            # Instability metric (Martin's I metric)
            total = in_degree + out_degree
            instability = out_degree / total if total > 0 else 0

            coupling[node] = {
                "afferent_coupling": in_degree,
                "efferent_coupling": out_degree,
                "instability": instability,
            }

        return coupling

    def _calculate_dependency_depth(self, import_graph: nx.DiGraph) -> Dict[str, int]:
        """Calculate maximum dependency depth for each module."""
        depths = {}

        # Find root nodes (no incoming dependencies)
        root_nodes = [n for n in import_graph.nodes() if import_graph.in_degree(n) == 0]

        for node in import_graph.nodes():
            max_depth = 0

            for root in root_nodes:
                try:
                    if nx.has_path(import_graph, root, node):
                        depth = nx.shortest_path_length(import_graph, root, node)
                        max_depth = max(max_depth, depth)
                except:
                    continue

            depths[node] = max_depth

        return depths

    def _analyze_naming_conventions(self) -> Dict[str, Any]:
        """Analyze naming convention adherence."""
        violations = []

        for python_file in self.project_root.rglob("*.py"):
            filename = python_file.name

            # Check file naming
            if not filename.islower() or not filename.replace("_", "").replace(".py", "").isalnum():
                violations.append(
                    {
                        "type": "file_naming",
                        "file": str(python_file),
                        "issue": "File name should be lowercase with underscores",
                    }
                )

            # Check function and class naming in AST
            try:
                source = python_file.read_text()
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.islower() or not node.name.replace("_", "").isalnum():
                            violations.append(
                                {
                                    "type": "function_naming",
                                    "file": str(python_file),
                                    "line": node.lineno,
                                    "name": node.name,
                                    "issue": "Function name should be lowercase with underscores",
                                }
                            )

                    elif isinstance(node, ast.ClassDef):
                        if not self._is_camel_case(node.name):
                            violations.append(
                                {
                                    "type": "class_naming",
                                    "file": str(python_file),
                                    "line": node.lineno,
                                    "name": node.name,
                                    "issue": "Class name should be CamelCase",
                                }
                            )

            except Exception:
                continue

        return {
            "violations": violations,
            "violation_count": len(violations),
            "files_checked": len(list(self.project_root.rglob("*.py"))),
        }

    def _is_camel_case(self, name: str) -> bool:
        """Check if name is CamelCase."""
        return name[0].isupper() and "_" not in name and name.isalnum()

    def _analyze_file_sizes(self) -> Dict[str, Any]:
        """Analyze file size distribution."""
        sizes = []
        large_files = []

        for python_file in self.project_root.rglob("*.py"):
            try:
                line_count = len(python_file.read_text().splitlines())
                sizes.append(line_count)

                if line_count > 500:  # Large file threshold
                    large_files.append({"file": str(python_file), "lines": line_count})
            except:
                continue

        return {
            "average_file_size": np.mean(sizes) if sizes else 0,
            "median_file_size": np.median(sizes) if sizes else 0,
            "large_files": large_files,
            "size_distribution": {
                "small (<100 lines)": len([s for s in sizes if s < 100]),
                "medium (100-300 lines)": len([s for s in sizes if 100 <= s <= 300]),
                "large (300-500 lines)": len([s for s in sizes if 300 < s <= 500]),
                "very_large (>500 lines)": len([s for s in sizes if s > 500]),
            },
        }

    def _calculate_directory_coupling(self) -> Dict[str, float]:
        """Calculate coupling between directories."""
        directory_imports = defaultdict(set)

        # Group imports by directory
        for python_file in self.project_root.rglob("*.py"):
            if python_file.name.startswith("."):
                continue

            directory = python_file.parent.relative_to(self.project_root)
            imports = self._extract_imports(python_file)

            for imp in imports:
                # Convert import to directory path
                if "." in imp:
                    import_dir = Path(imp.replace(".", "/"))
                    directory_imports[str(directory)].add(str(import_dir))

        # Calculate coupling between directories
        coupling = {}
        directories = list(directory_imports.keys())

        for dir1 in directories:
            dir1_imports = directory_imports[dir1]
            coupling_score = 0

            for dir2 in directories:
                if dir1 != dir2:
                    # Check if dir1 imports from dir2
                    if any(dir2 in imp for imp in dir1_imports):
                        coupling_score += 1

            coupling[dir1] = coupling_score / max(len(directories) - 1, 1)

        return coupling


class VectorRepresentation:
    """Analyzes code using vector embeddings for semantic understanding."""

    def __init__(self, embedding_integration):
        self.embedding_integration = embedding_integration
        self.code_vectors = {}
        self.semantic_vectors = {}
        self.similarity_clusters = {}

    async def analyze_semantic_structure(self, project_files: List[Path]) -> Dict[str, Any]:
        """Analyze semantic code structure using embeddings."""
        print("🔍 Analyzing semantic structure with embeddings...")

        # Generate embeddings for all functions and classes
        await self._generate_code_embeddings(project_files)

        analysis = {
            "semantic_clusters": await self._find_semantic_clusters(),
            "code_duplication": await self._detect_semantic_duplication(),
            "architectural_patterns": await self._identify_architectural_patterns(),
            "naming_consistency": await self._analyze_naming_consistency(),
            "functional_cohesion": await self._measure_functional_cohesion(),
        }

        return analysis

    async def _generate_code_embeddings(self, project_files: List[Path]):
        """Generate embeddings for all code elements."""
        for file_path in project_files:
            try:
                source = file_path.read_text()
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        # Extract source code for this element
                        element_source = self._extract_element_source(source, node)
                        element_id = f"{file_path.name}.{node.name}"

                        # Get code embedding
                        code_embedding = await self.embedding_integration._get_code_embedding(
                            element_source
                        )
                        if code_embedding:
                            self.code_vectors[element_id] = {
                                "embedding": code_embedding,
                                "source": element_source,
                                "type": (
                                    "function" if isinstance(node, ast.FunctionDef) else "class"
                                ),
                                "file": str(file_path),
                                "line": node.lineno,
                            }

                        # Get semantic embedding from docstring and name
                        semantic_text = self._create_semantic_text(node, element_source)
                        semantic_embedding = (
                            await self.embedding_integration._get_semantic_embedding(semantic_text)
                        )
                        if semantic_embedding:
                            self.semantic_vectors[element_id] = {
                                "embedding": semantic_embedding,
                                "text": semantic_text,
                                "file": str(file_path),
                                "line": node.lineno,
                            }

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

    def _extract_element_source(self, full_source: str, node: ast.AST) -> str:
        """Extract source code for a specific AST node."""
        lines = full_source.split("\n")
        start_line = node.lineno - 1

        # Find end line by looking for next def/class at same indentation level
        end_line = len(lines)
        if start_line < len(lines):
            base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())

            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                if line.strip() and len(line) - len(line.lstrip()) <= base_indent:
                    if any(line.strip().startswith(kw) for kw in ["def ", "class ", "async def "]):
                        end_line = i
                        break

        return "\n".join(lines[start_line:end_line])

    def _create_semantic_text(self, node: ast.AST, source: str) -> str:
        """Create semantic description for embedding."""
        docstring = ast.get_docstring(node) if hasattr(ast, "get_docstring") else ""

        return f"""
        Name: {node.name}
        Type: {"function" if isinstance(node, ast.FunctionDef) else "class"}
        Documentation: {docstring or "No documentation"}
        Code context: {source[:500]}
        """.strip()

    async def _find_semantic_clusters(self) -> Dict[str, List[str]]:
        """Find clusters of semantically similar code."""
        if not self.semantic_vectors:
            return {}

        # Simple clustering using cosine similarity
        clusters = defaultdict(list)
        cluster_id = 0
        processed = set()

        for element_id, vector_data in self.semantic_vectors.items():
            if element_id in processed:
                continue

            cluster_name = f"cluster_{cluster_id}"
            clusters[cluster_name].append(element_id)
            processed.add(element_id)

            # Find similar elements
            for other_id, other_data in self.semantic_vectors.items():
                if other_id in processed:
                    continue

                similarity = self._cosine_similarity(
                    vector_data["embedding"], other_data["embedding"]
                )

                if similarity > 0.8:  # High similarity threshold
                    clusters[cluster_name].append(other_id)
                    processed.add(other_id)

            cluster_id += 1

        # Filter out single-element clusters
        return {k: v for k, v in clusters.items() if len(v) > 1}

    async def _detect_semantic_duplication(self) -> List[Dict[str, Any]]:
        """Detect semantically duplicate code."""
        duplicates = []

        processed_pairs = set()

        for elem1_id, vec1_data in self.code_vectors.items():
            for elem2_id, vec2_data in self.code_vectors.items():
                if elem1_id >= elem2_id:  # Avoid duplicate comparisons
                    continue

                pair = tuple(sorted([elem1_id, elem2_id]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                similarity = self._cosine_similarity(vec1_data["embedding"], vec2_data["embedding"])

                if similarity > 0.9:  # Very high similarity = potential duplication
                    duplicates.append(
                        {
                            "element1": elem1_id,
                            "element2": elem2_id,
                            "similarity": similarity,
                            "file1": vec1_data["file"],
                            "file2": vec2_data["file"],
                            "line1": vec1_data["line"],
                            "line2": vec2_data["line"],
                            "type": vec1_data["type"],
                        }
                    )

        return sorted(duplicates, key=lambda x: x["similarity"], reverse=True)

    async def _identify_architectural_patterns(self) -> Dict[str, List[str]]:
        """Identify architectural patterns in code."""
        patterns = {
            "factories": [],
            "singletons": [],
            "observers": [],
            "strategies": [],
            "decorators": [],
            "adapters": [],
        }

        # Simple pattern detection using naming and structure
        for element_id, vector_data in self.code_vectors.items():
            element_name = element_id.split(".")[-1].lower()
            source = vector_data["source"].lower()

            if "factory" in element_name or "create" in element_name:
                patterns["factories"].append(element_id)
            elif "singleton" in source or "_instance" in source:
                patterns["singletons"].append(element_id)
            elif "notify" in source or "observer" in source:
                patterns["observers"].append(element_id)
            elif "strategy" in element_name or "algorithm" in source:
                patterns["strategies"].append(element_id)
            elif "decorator" in element_name or "@" in source:
                patterns["decorators"].append(element_id)
            elif "adapter" in element_name or "convert" in element_name:
                patterns["adapters"].append(element_id)

        return {k: v for k, v in patterns.items() if v}

    async def _analyze_naming_consistency(self) -> Dict[str, Any]:
        """Analyze naming consistency using semantic similarity."""
        naming_issues = []

        # Group by semantic similarity
        semantic_groups = await self._find_semantic_clusters()

        for cluster_name, elements in semantic_groups.items():
            if len(elements) < 2:
                continue

            # Check if similar functions have consistent naming patterns
            names = [elem.split(".")[-1] for elem in elements]

            # Simple heuristic: similar functions should have similar naming patterns
            prefixes = set()
            suffixes = set()

            for name in names:
                if "_" in name:
                    parts = name.split("_")
                    prefixes.add(parts[0])
                    suffixes.add(parts[-1])

            if len(prefixes) > 1 or len(suffixes) > 1:
                naming_issues.append(
                    {
                        "cluster": cluster_name,
                        "elements": elements,
                        "issue": "Inconsistent naming in semantically similar functions",
                        "different_prefixes": list(prefixes),
                        "different_suffixes": list(suffixes),
                    }
                )

        return {
            "naming_issues": naming_issues,
            "consistency_score": 1.0 - (len(naming_issues) / max(len(semantic_groups), 1)),
        }

    async def _measure_functional_cohesion(self) -> Dict[str, float]:
        """Measure functional cohesion within modules."""
        module_cohesion = {}

        # Group by file/module
        modules = defaultdict(list)
        for element_id, vector_data in self.semantic_vectors.items():
            module_name = Path(vector_data["file"]).stem
            modules[module_name].append(element_id)

        # Calculate cohesion for each module
        for module_name, elements in modules.items():
            if len(elements) < 2:
                module_cohesion[module_name] = 1.0
                continue

            # Calculate pairwise semantic similarities within module
            similarities = []
            for i in range(len(elements)):
                for j in range(i + 1, len(elements)):
                    elem1 = elements[i]
                    elem2 = elements[j]

                    vec1 = self.semantic_vectors[elem1]["embedding"]
                    vec2 = self.semantic_vectors[elem2]["embedding"]

                    similarity = self._cosine_similarity(vec1, vec2)
                    similarities.append(similarity)

            # Average similarity = cohesion measure
            module_cohesion[module_name] = np.mean(similarities) if similarities else 0.0

        return module_cohesion

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class GraphRepresentation:
    """Analyzes code using graph structures for dependency and flow analysis."""

    def __init__(self, dependency_graph_manager):
        self.dependency_graph_manager = dependency_graph_manager
        self.call_graph = nx.DiGraph()
        self.data_flow_graph = nx.DiGraph()
        self.control_flow_graphs = {}

    def analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze graph-based code structure."""
        analysis = {
            "dependency_metrics": self._analyze_dependency_structure(),
            "control_flow_complexity": self._analyze_control_flow(),
            "data_flow_patterns": self._analyze_data_flow(),
            "graph_quality_metrics": self._calculate_graph_quality(),
            "architectural_violations": self._detect_architectural_violations(),
        }

        return analysis

    def _analyze_dependency_structure(self) -> Dict[str, Any]:
        """Analyze dependency graph structure."""
        graph = self.dependency_graph_manager.dependency_graph

        return {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "strongly_connected_components": len(list(nx.strongly_connected_components(graph))),
            "weakly_connected_components": len(list(nx.weakly_connected_components(graph))),
            "average_clustering": nx.average_clustering(graph.to_undirected()),
            "longest_path": self._find_longest_dependency_path(graph),
            "circular_dependencies": self._find_circular_dependencies(graph),
        }

    def _analyze_control_flow(self) -> Dict[str, Any]:
        """Analyze control flow complexity."""
        # This would build control flow graphs for each function
        # For now, simplified version using cyclomatic complexity
        return {
            "average_complexity": 5.2,  # Placeholder
            "high_complexity_functions": [],
            "control_flow_violations": [],
        }

    def _analyze_data_flow(self) -> Dict[str, Any]:
        """Analyze data flow patterns."""
        # This would track variable usage and data dependencies
        return {"data_coupling": {}, "unused_variables": [], "data_flow_violations": []}

    def _calculate_graph_quality(self) -> Dict[str, float]:
        """Calculate overall graph quality metrics."""
        graph = self.dependency_graph_manager.dependency_graph

        if graph.number_of_nodes() == 0:
            return {"error": "empty_graph"}

        try:
            return {
                "modularity": self._calculate_modularity(graph),
                "efficiency": nx.global_efficiency(graph),
                "assortativity": nx.degree_assortativity_coefficient(graph),
                "transitivity": nx.transitivity(graph),
            }
        except Exception:
            return {"error": "calculation_failed"}

    def _calculate_modularity(self, graph: nx.DiGraph) -> float:
        """Calculate graph modularity."""
        try:
            # Convert to undirected for modularity calculation
            undirected = graph.to_undirected()
            communities = nx.community.greedy_modularity_communities(undirected)
            return nx.community.modularity(undirected, communities)
        except:
            return 0.0

    def _find_longest_dependency_path(self, graph: nx.DiGraph) -> List[str]:
        """Find the longest dependency path."""
        try:
            if nx.is_directed_acyclic_graph(graph):
                return list(nx.dag_longest_path(graph))
            else:
                # For graphs with cycles, find longest simple path
                longest = []
                for node in graph.nodes():
                    for target in graph.nodes():
                        if node != target:
                            try:
                                path = nx.shortest_path(graph, node, target)
                                if len(path) > len(longest):
                                    longest = path
                            except nx.NetworkXNoPath:
                                continue
                return longest
        except:
            return []

    def _find_circular_dependencies(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find circular dependencies."""
        try:
            return list(nx.simple_cycles(graph))
        except:
            return []

    def _detect_architectural_violations(self) -> List[Dict[str, Any]]:
        """Detect architectural violations in graph structure."""
        violations = []
        graph = self.dependency_graph_manager.dependency_graph

        # Detect violations of common architectural principles

        # 1. Circular dependencies
        cycles = self._find_circular_dependencies(graph)
        for cycle in cycles:
            violations.append(
                {
                    "type": "circular_dependency",
                    "severity": "high",
                    "description": f"Circular dependency detected: {' -> '.join(cycle)}",
                    "affected_modules": cycle,
                }
            )

        # 2. High fan-out (single module depending on too many others)
        for node in graph.nodes():
            out_degree = graph.out_degree(node)
            if out_degree > 10:  # Threshold for high coupling
                violations.append(
                    {
                        "type": "high_fan_out",
                        "severity": "medium",
                        "description": f"Module {node} has high fan-out ({out_degree} dependencies)",
                        "affected_modules": [node],
                    }
                )

        # 3. High fan-in (single module being depended on by too many others)
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            if in_degree > 15:  # Threshold for high coupling
                violations.append(
                    {
                        "type": "high_fan_in",
                        "severity": "medium",
                        "description": f"Module {node} has high fan-in ({in_degree} dependents)",
                        "affected_modules": [node],
                    }
                )

        return violations


class MultiRepresentationAnalyzer:
    """
    Unified analyzer that combines filesystem, vector, and graph representations
    to provide comprehensive software quality analysis and improvement suggestions.
    """

    def __init__(self, project_root: Path, embedding_integration, dependency_graph_manager):
        self.project_root = project_root
        self.filesystem_analyzer = FilesystemRepresentation(project_root)
        self.vector_analyzer = VectorRepresentation(embedding_integration)
        self.graph_analyzer = GraphRepresentation(dependency_graph_manager)

        self.quality_metrics = SoftwareQualityMetrics()
        self.improvement_opportunities = []

    async def comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis across all representations."""
        print("🔍 Running comprehensive multi-representation analysis...")

        # Get all Python files
        python_files = list(self.project_root.rglob("*.py"))

        # Analyze each representation
        print("📁 Analyzing filesystem representation...")
        filesystem_analysis = self.filesystem_analyzer.analyze_project_structure()

        print("🧠 Analyzing vector representation...")
        vector_analysis = await self.vector_analyzer.analyze_semantic_structure(python_files)

        print("🕸️ Analyzing graph representation...")
        graph_analysis = self.graph_analyzer.analyze_graph_structure()

        # Cross-representation analysis
        print("🔗 Performing cross-representation analysis...")
        cross_analysis = await self._cross_representation_analysis(
            filesystem_analysis, vector_analysis, graph_analysis
        )

        # Generate improvement opportunities
        print("💡 Generating improvement opportunities...")
        self.improvement_opportunities = await self._generate_improvement_opportunities(
            filesystem_analysis, vector_analysis, graph_analysis, cross_analysis
        )

        return {
            "filesystem_analysis": filesystem_analysis,
            "vector_analysis": vector_analysis,
            "graph_analysis": graph_analysis,
            "cross_representation_analysis": cross_analysis,
            "improvement_opportunities": self.improvement_opportunities,
            "quality_metrics": self.quality_metrics,
            "summary": self._generate_analysis_summary(),
        }

    async def _cross_representation_analysis(
        self, fs_analysis, vec_analysis, graph_analysis
    ) -> Dict[str, Any]:
        """Analyze relationships between different representations."""

        # Calculate alignment between representations
        fs_vector_alignment = await self._calculate_fs_vector_alignment(fs_analysis, vec_analysis)
        vector_graph_consistency = await self._calculate_vector_graph_consistency(
            vec_analysis, graph_analysis
        )
        graph_fs_coherence = await self._calculate_graph_fs_coherence(graph_analysis, fs_analysis)

        # Store in quality metrics
        self.quality_metrics.filesystem_vector_alignment = fs_vector_alignment
        self.quality_metrics.vector_graph_consistency = vector_graph_consistency
        self.quality_metrics.graph_filesystem_coherence = graph_fs_coherence

        return {
            "representation_alignment": {
                "filesystem_vector": fs_vector_alignment,
                "vector_graph": vector_graph_consistency,
                "graph_filesystem": graph_fs_coherence,
            },
            "consistency_issues": await self._find_consistency_issues(
                fs_analysis, vec_analysis, graph_analysis
            ),
            "reinforcing_patterns": await self._find_reinforcing_patterns(
                fs_analysis, vec_analysis, graph_analysis
            ),
        }

    async def _calculate_fs_vector_alignment(self, fs_analysis, vec_analysis) -> float:
        """Calculate alignment between filesystem organization and semantic similarity."""
        # Check if files in same directory have high semantic similarity
        if not vec_analysis.get("semantic_clusters"):
            return 0.5

        alignment_scores = []

        # For each semantic cluster, check if elements are in related directories
        for cluster_name, elements in vec_analysis["semantic_clusters"].items():
            directories = set()
            for element in elements:
                if element in self.vector_analyzer.semantic_vectors:
                    file_path = self.vector_analyzer.semantic_vectors[element]["file"]
                    directories.add(str(Path(file_path).parent))

            # High alignment = semantically similar code is in same/related directories
            if len(directories) == 1:
                alignment_scores.append(1.0)  # Perfect alignment
            elif len(directories) <= 2:
                alignment_scores.append(0.7)  # Good alignment
            else:
                alignment_scores.append(0.3)  # Poor alignment

        return np.mean(alignment_scores) if alignment_scores else 0.5

    async def _calculate_vector_graph_consistency(self, vec_analysis, graph_analysis) -> float:
        """Calculate consistency between semantic similarity and dependency structure."""
        # Check if semantically similar code has similar dependency patterns
        if not vec_analysis.get("semantic_clusters"):
            return 0.5

        consistency_scores = []

        for cluster_name, elements in vec_analysis["semantic_clusters"].items():
            if len(elements) < 2:
                continue

            # Check if elements in cluster have similar dependency patterns
            dependency_patterns = []
            for element in elements:
                # Get dependencies for this element from graph
                if element in self.graph_analyzer.dependency_graph_manager.dependency_graph:
                    deps = list(
                        self.graph_analyzer.dependency_graph_manager.dependency_graph.successors(
                            element
                        )
                    )
                    dependency_patterns.append(set(deps))

            if len(dependency_patterns) >= 2:
                # Calculate similarity of dependency patterns
                avg_similarity = 0
                comparisons = 0

                for i in range(len(dependency_patterns)):
                    for j in range(i + 1, len(dependency_patterns)):
                        intersection = len(dependency_patterns[i] & dependency_patterns[j])
                        union = len(dependency_patterns[i] | dependency_patterns[j])
                        similarity = intersection / union if union > 0 else 1.0
                        avg_similarity += similarity
                        comparisons += 1

                if comparisons > 0:
                    consistency_scores.append(avg_similarity / comparisons)

        return np.mean(consistency_scores) if consistency_scores else 0.5

    async def _calculate_graph_fs_coherence(self, graph_analysis, fs_analysis) -> float:
        """Calculate coherence between graph structure and filesystem organization."""
        # Check if dependency relationships match filesystem hierarchy
        coherence_score = 0.5  # Default neutral score

        # This would check if modules that depend on each other are appropriately organized
        # in the filesystem hierarchy (e.g., core modules in core directory, utilities in utils)

        return coherence_score

    async def _find_consistency_issues(
        self, fs_analysis, vec_analysis, graph_analysis
    ) -> List[Dict[str, Any]]:
        """Find inconsistencies between representations."""
        issues = []

        # Issue: Semantically similar code scattered across filesystem
        for cluster_name, elements in vec_analysis.get("semantic_clusters", {}).items():
            directories = set()
            for element in elements:
                if element in self.vector_analyzer.semantic_vectors:
                    file_path = self.vector_analyzer.semantic_vectors[element]["file"]
                    directories.add(str(Path(file_path).parent))

            if len(directories) > 2:
                issues.append(
                    {
                        "type": "semantic_filesystem_mismatch",
                        "description": f"Semantically similar code scattered across {len(directories)} directories",
                        "affected_elements": elements,
                        "directories": list(directories),
                        "severity": "medium",
                    }
                )

        # Issue: High filesystem coupling but low semantic similarity
        # (would require more detailed analysis)

        return issues

    async def _find_reinforcing_patterns(
        self, fs_analysis, vec_analysis, graph_analysis
    ) -> List[Dict[str, Any]]:
        """Find patterns that are reinforced across multiple representations."""
        patterns = []

        # Pattern: Good package organization reinforced by semantic coherence
        for package in fs_analysis.get("package_organization", {}).get("packages", []):
            if package["cohesion_score"] > 0.7:
                patterns.append(
                    {
                        "type": "well_organized_package",
                        "description": f"Package {package['name']} shows good organization across representations",
                        "package": package["name"],
                        "evidence": {
                            "filesystem_cohesion": package["cohesion_score"],
                            "clear_purpose": package["has_clear_purpose"],
                            "naming_convention": package["follows_naming_convention"],
                        },
                        "strength": "high",
                    }
                )

        return patterns

    async def _generate_improvement_opportunities(
        self, fs_analysis, vec_analysis, graph_analysis, cross_analysis
    ) -> List[ImprovementOpportunity]:
        """Generate specific improvement opportunities based on analysis."""
        opportunities = []

        # SOLID principle violations
        opportunities.extend(
            await self._generate_solid_improvements(fs_analysis, vec_analysis, graph_analysis)
        )

        # Complexity reduction opportunities
        opportunities.extend(
            await self._generate_complexity_improvements(fs_analysis, graph_analysis)
        )

        # Pythonic improvements
        opportunities.extend(await self._generate_pythonic_improvements(fs_analysis, vec_analysis))

        # Architecture improvements
        opportunities.extend(
            await self._generate_architecture_improvements(graph_analysis, cross_analysis)
        )

        return sorted(
            opportunities, key=lambda x: self._calculate_opportunity_priority(x), reverse=True
        )

    async def _generate_solid_improvements(
        self, fs_analysis, vec_analysis, graph_analysis
    ) -> List[ImprovementOpportunity]:
        """Generate SOLID principle improvement opportunities."""
        opportunities = []

        # Single Responsibility Principle violations
        for file_info in fs_analysis.get("large_files", []):
            if file_info["lines"] > 500:
                opportunities.append(
                    ImprovementOpportunity(
                        opportunity_id=f"srp_large_file_{hash(file_info['file'])}",
                        category="solid",
                        severity="medium",
                        file_path=file_info["file"],
                        line_number=1,
                        affected_modules=[file_info["file"]],
                        similar_code_patterns=[],
                        semantic_clusters=[],
                        dependency_impact=[],
                        execution_criticality=0.5,
                        current_state=f"Large file with {file_info['lines']} lines",
                        target_state="Break into focused modules with single responsibilities",
                        implementation_steps=[
                            "Identify distinct responsibilities in the file",
                            "Extract classes/functions by responsibility",
                            "Create separate modules for each responsibility",
                            "Update imports and dependencies",
                        ],
                        estimated_impact={
                            "maintainability": 0.8,
                            "testability": 0.7,
                            "readability": 0.9,
                        },
                        risk_assessment="medium",
                    )
                )

        # Dependency Inversion Principle violations (high coupling to concrete classes)
        for violation in graph_analysis.get("architectural_violations", []):
            if violation["type"] == "high_fan_out":
                opportunities.append(
                    ImprovementOpportunity(
                        opportunity_id=f"dip_high_coupling_{hash(violation['affected_modules'][0])}",
                        category="solid",
                        severity="high",
                        file_path=violation["affected_modules"][0],
                        line_number=1,
                        affected_modules=violation["affected_modules"],
                        similar_code_patterns=[],
                        semantic_clusters=[],
                        dependency_impact=violation["affected_modules"],
                        execution_criticality=0.8,
                        current_state=violation["description"],
                        target_state="Introduce abstractions to reduce coupling",
                        implementation_steps=[
                            "Identify common interfaces in dependencies",
                            "Create abstract base classes or protocols",
                            "Refactor to depend on abstractions",
                            "Use dependency injection where appropriate",
                        ],
                        estimated_impact={
                            "maintainability": 0.9,
                            "testability": 0.8,
                            "flexibility": 0.9,
                        },
                        risk_assessment="medium",
                    )
                )

        return opportunities

    async def _generate_complexity_improvements(
        self, fs_analysis, graph_analysis
    ) -> List[ImprovementOpportunity]:
        """Generate complexity reduction opportunities."""
        opportunities = []

        # High cyclomatic complexity functions
        for file_path, file_data in fs_analysis.get("module_hierarchy", {}).items():
            if isinstance(file_data, dict) and file_data.get("type") == "file":
                for func in file_data.get("functions", []):
                    if func["cyclomatic_complexity"] > 10:
                        func_identifier = f"{file_path}_{func['name']}"
                        opportunities.append(
                            ImprovementOpportunity(
                                opportunity_id=f"complexity_function_{hash(func_identifier)}",
                                category="complexity",
                                severity="high" if func["cyclomatic_complexity"] > 15 else "medium",
                                file_path=file_data["path"],
                                line_number=func["line_number"],
                                affected_modules=[file_data["path"]],
                                similar_code_patterns=[],
                                semantic_clusters=[],
                                dependency_impact=[],
                                execution_criticality=0.7,
                                current_state=f"Function {func['name']} has complexity {func['cyclomatic_complexity']}",
                                target_state="Reduce complexity to below 10",
                                implementation_steps=[
                                    "Extract complex conditional logic into separate functions",
                                    "Use early returns to reduce nesting",
                                    "Consider using strategy pattern for complex branching",
                                    "Split function into smaller, focused functions",
                                ],
                                estimated_impact={
                                    "maintainability": 0.8,
                                    "testability": 0.9,
                                    "readability": 0.8,
                                },
                                risk_assessment="low",
                            )
                        )

        return opportunities

    async def _generate_pythonic_improvements(
        self, fs_analysis, vec_analysis
    ) -> List[ImprovementOpportunity]:
        """Generate Pythonic best practices improvements."""
        opportunities = []

        # Missing type hints
        total_functions = 0
        functions_with_hints = 0

        for file_path, file_data in fs_analysis.get("module_hierarchy", {}).items():
            if isinstance(file_data, dict) and file_data.get("type") == "file":
                for func in file_data.get("functions", []):
                    total_functions += 1
                    if func["has_type_hints"]:
                        functions_with_hints += 1

        type_hint_coverage = functions_with_hints / total_functions if total_functions > 0 else 1.0

        if type_hint_coverage < 0.8:
            opportunities.append(
                ImprovementOpportunity(
                    opportunity_id="pythonic_type_hints",
                    category="pythonic",
                    severity="medium",
                    file_path=str(self.project_root),
                    line_number=1,
                    affected_modules=["project_wide"],
                    similar_code_patterns=[],
                    semantic_clusters=[],
                    dependency_impact=[],
                    execution_criticality=0.3,
                    current_state=f"Type hint coverage: {type_hint_coverage:.1%}",
                    target_state="Achieve >80% type hint coverage",
                    implementation_steps=[
                        "Add type hints to function parameters and return values",
                        "Use typing module for complex types",
                        "Add mypy to CI/CD pipeline",
                        "Gradually improve coverage over time",
                    ],
                    estimated_impact={
                        "maintainability": 0.7,
                        "ide_support": 0.9,
                        "documentation": 0.6,
                    },
                    risk_assessment="low",
                )
            )

        # Naming convention violations
        naming_violations = fs_analysis.get("naming_conventions", {}).get("violations", [])
        if len(naming_violations) > 10:
            opportunities.append(
                ImprovementOpportunity(
                    opportunity_id="pythonic_naming",
                    category="pythonic",
                    severity="low",
                    file_path=str(self.project_root),
                    line_number=1,
                    affected_modules=["project_wide"],
                    similar_code_patterns=[],
                    semantic_clusters=[],
                    dependency_impact=[],
                    execution_criticality=0.2,
                    current_state=f"{len(naming_violations)} naming convention violations",
                    target_state="Follow PEP 8 naming conventions consistently",
                    implementation_steps=[
                        "Rename functions to use snake_case",
                        "Rename classes to use PascalCase",
                        "Rename files to use lowercase with underscores",
                        "Update all references and imports",
                    ],
                    estimated_impact={
                        "readability": 0.6,
                        "consistency": 0.8,
                        "professionalism": 0.7,
                    },
                    risk_assessment="low",
                )
            )

        return opportunities

    async def _generate_architecture_improvements(
        self, graph_analysis, cross_analysis
    ) -> List[ImprovementOpportunity]:
        """Generate architecture improvement opportunities."""
        opportunities = []

        # Circular dependencies
        circular_deps = graph_analysis.get("dependency_metrics", {}).get(
            "circular_dependencies", []
        )
        for cycle in circular_deps:
            opportunities.append(
                ImprovementOpportunity(
                    opportunity_id=f"arch_circular_dep_{hash(str(cycle))}",
                    category="architecture",
                    severity="high",
                    file_path=cycle[0] if cycle else "",
                    line_number=1,
                    affected_modules=cycle,
                    similar_code_patterns=[],
                    semantic_clusters=[],
                    dependency_impact=cycle,
                    execution_criticality=0.9,
                    current_state=f"Circular dependency: {' -> '.join(cycle)}",
                    target_state="Break circular dependency",
                    implementation_steps=[
                        "Identify the reason for the circular dependency",
                        "Extract common functionality to a shared module",
                        "Use dependency inversion to break the cycle",
                        "Consider using events or callbacks instead of direct dependencies",
                    ],
                    estimated_impact={
                        "maintainability": 0.9,
                        "testability": 0.8,
                        "modularity": 0.9,
                    },
                    risk_assessment="high",
                )
            )

        # Poor representation alignment
        if cross_analysis["representation_alignment"]["filesystem_vector"] < 0.5:
            opportunities.append(
                ImprovementOpportunity(
                    opportunity_id="arch_filesystem_semantic_mismatch",
                    category="architecture",
                    severity="medium",
                    file_path=str(self.project_root),
                    line_number=1,
                    affected_modules=["project_wide"],
                    similar_code_patterns=[],
                    semantic_clusters=[],
                    dependency_impact=[],
                    execution_criticality=0.6,
                    current_state="Filesystem organization doesn't match semantic relationships",
                    target_state="Reorganize files to group semantically related code",
                    implementation_steps=[
                        "Identify semantically related code that's scattered",
                        "Create logical package structure based on functionality",
                        "Move related modules to appropriate packages",
                        "Update imports and documentation",
                    ],
                    estimated_impact={
                        "discoverability": 0.8,
                        "maintainability": 0.7,
                        "onboarding": 0.9,
                    },
                    risk_assessment="medium",
                )
            )

        return opportunities

    def _calculate_opportunity_priority(self, opportunity: ImprovementOpportunity) -> float:
        """Calculate priority score for an improvement opportunity."""
        severity_weights = {"high": 1.0, "medium": 0.6, "low": 0.3}
        category_weights = {"solid": 1.0, "architecture": 0.9, "complexity": 0.8, "pythonic": 0.5}

        severity_score = severity_weights.get(opportunity.severity, 0.5)
        category_score = category_weights.get(opportunity.category, 0.5)
        impact_score = np.mean(list(opportunity.estimated_impact.values()))
        criticality_score = opportunity.execution_criticality

        return (
            severity_score * 0.3
            + category_score * 0.3
            + impact_score * 0.2
            + criticality_score * 0.2
        )

    def _generate_analysis_summary(self) -> Dict[str, Any]:
        """Generate summary of analysis results."""
        return {
            "total_improvement_opportunities": len(self.improvement_opportunities),
            "high_priority_opportunities": len(
                [op for op in self.improvement_opportunities if op.severity == "high"]
            ),
            "representation_alignment_score": np.mean(
                [
                    self.quality_metrics.filesystem_vector_alignment,
                    self.quality_metrics.vector_graph_consistency,
                    self.quality_metrics.graph_filesystem_coherence,
                ]
            ),
            "top_improvement_categories": self._get_top_categories(),
            "overall_quality_score": self._calculate_overall_quality_score(),
        }

    def _get_top_categories(self) -> List[str]:
        """Get top improvement categories by count."""
        category_counts = {}
        for op in self.improvement_opportunities:
            category_counts[op.category] = category_counts.get(op.category, 0) + 1

        return sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True)

    def _calculate_overall_quality_score(self) -> float:
        """Calculate overall software quality score."""
        # Combine multiple factors into overall score
        high_priority_penalty = (
            len([op for op in self.improvement_opportunities if op.severity == "high"]) * 0.1
        )
        alignment_bonus = (
            np.mean(
                [
                    self.quality_metrics.filesystem_vector_alignment,
                    self.quality_metrics.vector_graph_consistency,
                    self.quality_metrics.graph_filesystem_coherence,
                ]
            )
            * 0.3
        )

        base_score = 0.7  # Baseline quality
        quality_score = base_score + alignment_bonus - high_priority_penalty

        return max(0.0, min(1.0, quality_score))


# Main integration function
async def analyze_vibelint_quality(
    project_root: Path, embedding_integration, dependency_graph_manager
) -> Dict[str, Any]:
    """
    Run complete multi-representation quality analysis on vibelint.
    This is the main entry point for autonomous software improvement.
    """
    print("🔍 Starting comprehensive vibelint quality analysis...")

    analyzer = MultiRepresentationAnalyzer(
        project_root, embedding_integration, dependency_graph_manager
    )

    analysis_results = await analyzer.comprehensive_analysis()

    # Export results for LLM consumption
    export_path = project_root / ".vibelint-self-improvement" / "quality_analysis.json"
    export_path.parent.mkdir(exist_ok=True)

    with open(export_path, "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)

    print("📊 Quality analysis complete:")
    print(
        f"  - {analysis_results['summary']['total_improvement_opportunities']} improvement opportunities found"
    )
    print(f"  - Overall quality score: {analysis_results['summary']['overall_quality_score']:.2f}")
    print(f"  - Report saved to: {export_path}")

    return analysis_results


if __name__ == "__main__":
    # Example usage
    async def demo():
        from pathlib import Path

        from .dependency_graph_manager import DependencyGraphManager
        from .runtime_tracer import VanguardEmbeddingIntegration

        project_root = Path(__file__).parent.parent.parent
        embedding_integration = VanguardEmbeddingIntegration()
        dependency_manager = DependencyGraphManager()

        results = await analyze_vibelint_quality(
            project_root, embedding_integration, dependency_manager
        )

        print(
            f"Analysis complete! Found {len(results['improvement_opportunities'])} opportunities."
        )

    import asyncio

    asyncio.run(demo())
