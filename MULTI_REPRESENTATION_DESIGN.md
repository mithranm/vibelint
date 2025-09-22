# Multi-Representation System Design for Vibelint

## Overview
The multi-representation system builds four complementary views of Python codebases for comprehensive analysis: Filesystem, Vector, Graph, and Runtime representations.

## 1. Filesystem Representation

### Structure
```python
class FilesystemRepresentation:
    root_path: Path
    files: Dict[Path, FileMetadata]
    directories: Dict[Path, DirectoryMetadata]
    ignore_patterns: List[str]

class FileMetadata:
    path: Path
    size: int
    last_modified: datetime
    encoding: str
    ast_tree: Optional[ast.AST]
    imports: List[str]
    exports: List[str]
    complexity_metrics: Dict[str, float]
```

### Building Process
1. **Discovery Phase**
   - Find Python project root (pyproject.toml, setup.py, .git)
   - Discover all .py files (respect .gitignore)
   - **HUMAN CHECKPOINT**: Show discovered files, let human exclude/include specific areas

2. **Metadata Extraction**
   - Parse each file to AST
   - Extract imports, exports, function/class definitions
   - Calculate basic complexity metrics
   - Detect encoding and file structure

3. **Relationship Mapping**
   - Map file dependencies through imports
   - Identify entry points and leaf modules
   - Build directory hierarchy with metadata

## 2. Vector Representation

### Architecture
```python
class VectorRepresentation:
    embeddings: Dict[str, np.ndarray]  # chunk_id -> embedding
    metadata: Dict[str, ChunkMetadata]
    similarity_index: VectorIndex
    clusters: List[Cluster]

class ChunkMetadata:
    chunk_id: str
    file_path: Path
    line_range: Tuple[int, int]
    chunk_type: str  # function, class, module, comment
    content: str
    ast_node: Optional[ast.AST]
```

### Embedding Strategy
1. **Code Chunking**
   - Function-level chunks for functions/methods
   - Class-level chunks for class definitions
   - Module-level chunks for top-level code
   - Comment chunks for significant documentation

2. **Dual Model Embedding**
   - **VanguardOne**: Primary embedding generation
   - **VanguardTwo**: Alternative embedding for cross-validation
   - Store both embeddings for consensus building

3. **Qdrant Integration**
   - Store embeddings in Qdrant vector database
   - Include timestamps for Evidence-Based Reasoning
   - Support similarity search and clustering

4. **Human Interaction Points**
   - **HUMAN CHECKPOINT**: Show embedding clusters, let human label/categorize patterns
   - Human validates semantic similarity groupings
   - Human identifies important code patterns for focus

## 3. Graph Representation

### Structure
```python
class GraphRepresentation:
    dependency_graph: nx.DiGraph
    call_graph: nx.DiGraph
    inheritance_graph: nx.DiGraph
    semantic_graph: nx.Graph  # Enhanced with embedding similarities

class GraphNode:
    node_id: str
    node_type: str  # module, class, function, variable
    file_path: Path
    line_number: int
    metadata: Dict[str, Any]
    embedding_vector: Optional[np.ndarray]

class GraphEdge:
    source: str
    target: str
    edge_type: str  # import, call, inherit, semantic_similarity
    weight: float
    metadata: Dict[str, Any]
```

### Graph Building Process
1. **Dependency Graph Construction**
   - Build from import statements
   - Include internal and external dependencies
   - Weight edges by import frequency/importance

2. **Call Graph Construction**
   - Parse function calls within codebase
   - Track method calls and inheritance relationships
   - Include dynamic dispatch where detectable

3. **Semantic Enhancement**
   - Add edges based on embedding similarities
   - **HUMAN CHECKPOINT**: Visualize graph, let human identify critical paths to focus on
   - Weight semantic edges based on human feedback

4. **Graph Analysis**
   - Identify strongly connected components
   - Find critical paths and bottlenecks
   - Detect architectural patterns and anti-patterns

## 4. Runtime Representation

### Mock Execution Model
```python
class RuntimeRepresentation:
    execution_paths: List[ExecutionPath]
    call_patterns: Dict[str, CallPattern]
    data_flows: List[DataFlow]
    performance_estimates: Dict[str, float]

class ExecutionPath:
    path_id: str
    entry_point: str
    call_sequence: List[str]
    estimated_frequency: float
    complexity_score: float

class CallPattern:
    function_name: str
    call_sites: List[CallSite]
    parameter_patterns: Dict[str, Any]
    return_patterns: Dict[str, Any]
```

### Analysis Approach
1. **Static Execution Tracing**
   - Trace possible execution paths through AST
   - Identify common execution patterns
   - Estimate call frequencies based on structure

2. **Pattern Recognition**
   - Detect common design patterns
   - Identify potential performance bottlenecks
   - Find error handling patterns

3. **Integration with Other Representations**
   - Correlate with dependency graph for call validation
   - Use embeddings to identify similar execution patterns
   - Validate against filesystem structure

## 5. Representation Coordination

### Unified Analysis Interface
```python
class MultiRepresentationAnalyzer:
    filesystem: FilesystemRepresentation
    vector: VectorRepresentation
    graph: GraphRepresentation
    runtime: RuntimeRepresentation

    async def build_all_representations(self, project_path: Path) -> None:
        """Build all four representations with human checkpoints"""

    async def cross_correlate(self) -> CorrelationResult:
        """Find patterns across all representations"""

    async def generate_insights(self, models: ModelConfig) -> AnalysisInsights:
        """Use four-model system to generate comprehensive insights"""
```

### Cross-Representation Validation
1. **Consistency Checks**
   - Validate that imports in filesystem match graph dependencies
   - Ensure vector clusters align with architectural boundaries
   - Verify runtime patterns match static structure

2. **Conflict Resolution**
   - Identify discrepancies between representations
   - Use human decision points to resolve conflicts
   - Store resolutions in memory system for future reference

## 6. Human Checkpoint Integration

### Strategic Decision Points
- **Scope Selection**: Human chooses analysis depth and focus areas
- **Pattern Validation**: Human validates AI-discovered patterns
- **Priority Setting**: Human ranks insights by importance
- **Context Review**: Human approves context for next development steps

### Checkpoint Implementation
```python
class HumanCheckpoint:
    async def review_file_discovery(self, discovered_files: List[Path]) -> List[Path]:
        """Let human exclude/include specific files"""

    async def validate_embeddings_clusters(self, clusters: List[Cluster]) -> List[Cluster]:
        """Human labels and validates semantic clusters"""

    async def review_graph_visualization(self, graph: nx.Graph) -> CriticalPaths:
        """Human identifies important paths in dependency graph"""

    async def approve_analysis_summary(self, insights: AnalysisInsights) -> AnalysisInsights:
        """Human reviews and refines final analysis"""
```

## 7. Integration with Four-Model System

### Model Coordination Pattern
1. **VanguardOne/VanguardTwo**: Generate embeddings for vector representation
2. **Chip/Claudia**: Process all representations for architectural insights
3. **Human Resolution**: Resolve conflicts between model outputs
4. **Memory Integration**: Store decisions for Evidence-Based Reasoning

### Analysis Workflow
```python
async def comprehensive_analysis(project: Project, models: ModelConfig) -> AnalysisResult:
    # Phase 1: Build representations (with human checkpoints)
    representations = await build_all_representations(project)

    # Phase 2: Model analysis (parallel execution)
    vanguard_analysis = await models.vanguard_one.analyze(representations)
    chip_analysis = await models.chip.analyze(representations)

    # Phase 3: Human coordination
    consensus = await human_resolve_conflicts(vanguard_analysis, chip_analysis)

    # Phase 4: Context preparation
    return await prepare_development_context(consensus, representations)
```

## 8. Performance and Scalability

### Optimization Strategies
1. **Incremental Building**
   - Only rebuild changed parts of representations
   - Cache expensive computations (embeddings, graph analysis)
   - Use file modification timestamps for invalidation

2. **Parallel Processing**
   - Build representations in parallel where possible
   - Parallelize embedding generation across chunks
   - Distribute graph analysis across subgraphs

3. **Memory Management**
   - Stream large files rather than loading entirely
   - Use lazy loading for embeddings and graph nodes
   - Implement LRU caching for frequently accessed data

### Resource Requirements
- **Memory**: ~100MB per 10K lines of code
- **Storage**: ~50MB per project for cached representations
- **Compute**: ~30 seconds for initial build, ~5 seconds for incremental updates

## 9. Quality Assurance

### Validation Framework
1. **Representation Completeness**
   - Verify all files are included in filesystem representation
   - Ensure all code chunks have embeddings
   - Validate graph connectivity

2. **Cross-Representation Consistency**
   - Import statements match graph dependencies
   - Embedding clusters align with module boundaries
   - Runtime patterns reflect static structure

3. **Human Validation**
   - Human approves all major insights
   - Human validates pattern discoveries
   - Human confirms context packages are accurate

This multi-representation system provides the foundation for comprehensive code analysis while maintaining human control over strategic decisions.