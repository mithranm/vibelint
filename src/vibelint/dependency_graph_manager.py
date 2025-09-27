"""
Dependency Graph Manager for Vibelint

Builds, stores, and maintains the execution dependency graph using:
- NetworkX for in-memory graph analysis and algorithms
- Qdrant for persistent vector storage with embeddings
- Incremental updates for efficiency

This creates a living knowledge graph that an orchestrating LLM can query.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from vibelint.runtime_tracer import (DependencyNode, TraceSession,
                             VanguardEmbeddingIntegration)


@dataclass
class GraphMetrics:
    """Metrics about the dependency graph."""

    total_nodes: int
    total_edges: int
    max_depth: int
    most_connected_node: str
    performance_critical_path: List[str]
    embedding_coverage: float
    last_updated: float


@dataclass
class GraphQuery:
    """Query for finding relevant dependencies."""

    semantic_query: str
    code_similarity_threshold: float = 0.8
    performance_threshold_ms: float = 100.0
    max_results: int = 10


class DependencyGraphManager:
    """
    Manages the execution dependency graph with hybrid storage:
    - NetworkX for fast in-memory graph operations
    - Qdrant for persistent embedding-based storage
    - Incremental updates to keep both in sync
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vibelint_embeddings",
        config: Dict[str, Any] = None,
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.config = config or {}

        # In-memory graph for fast analysis
        self.dependency_graph = nx.DiGraph()

        # Embedding integration
        self.embedding_integration = VanguardEmbeddingIntegration(config)

        # Cache for performance
        self.node_cache = {}
        self.embedding_cache = {}

        # Graph statistics
        self.metrics = GraphMetrics(
            total_nodes=0,
            total_edges=0,
            max_depth=0,
            most_connected_node="",
            performance_critical_path=[],
            embedding_coverage=0.0,
            last_updated=time.time(),
        )

    async def build_graph_from_trace(self, trace_session: TraceSession) -> Dict[str, Any]:
        """
        Build/update dependency graph from a trace session.
        This is the main entry point for graph construction.
        """
        print(f"🔗 Building dependency graph from {trace_session.total_calls} calls...")

        # Create embedding-enhanced dependency nodes
        source_cache = {}
        dependency_nodes = await self.embedding_integration.create_dependency_embeddings(
            trace_session.call_stack, source_cache
        )

        # Update in-memory NetworkX graph
        self._update_networkx_graph(dependency_nodes, trace_session.call_stack)

        # Store/update nodes in Qdrant with embeddings
        await self._store_nodes_in_qdrant(dependency_nodes)

        # Update graph metrics
        self._update_graph_metrics()

        # Save dependency knowledge graph to trace session
        trace_session.dependency_knowledge_graph = dependency_nodes

        return {
            "nodes_processed": len(dependency_nodes),
            "graph_metrics": asdict(self.metrics),
            "critical_path": self._find_critical_path(),
            "performance_hotspots": self._identify_performance_hotspots(),
        }

    def _update_networkx_graph(self, nodes: Dict[str, DependencyNode], call_stack: List):
        """Update the in-memory NetworkX graph."""
        # Add/update nodes
        for node_id, node in nodes.items():
            # Add node with rich attributes
            self.dependency_graph.add_node(
                node_id,
                signature=node.function_signature,
                module=node.module_path,
                avg_time_ms=node.performance_profile.get("avg_time_ms", 0),
                call_frequency=node.execution_context.get("call_frequency", 1),
                io_operations=node.performance_profile.get("io_operations", 0),
                has_code_embedding=node.code_embedding is not None,
                has_semantic_embedding=node.semantic_embedding is not None,
            )

            # Add dependency edges
            for dependency in node.dependencies:
                if dependency in nodes:  # Only add edges to nodes we have
                    self.dependency_graph.add_edge(
                        node_id,
                        dependency,
                        weight=1.0,  # Could be based on call frequency
                        relationship_type="calls",
                    )

        # Add temporal edges (what calls what in sequence)
        self._add_temporal_edges(call_stack)

    def _add_temporal_edges(self, call_stack: List):
        """Add edges based on temporal execution order."""
        for i in range(len(call_stack) - 1):
            current_call = call_stack[i]
            next_call = call_stack[i + 1]

            current_node = f"{current_call.module_name}.{current_call.function_name}"
            next_node = f"{next_call.module_name}.{next_call.function_name}"

            # Add temporal edge if the next call is deeper (called by current)
            if next_call.call_depth > current_call.call_depth:
                if self.dependency_graph.has_edge(current_node, next_node):
                    # Increase weight of existing edge
                    self.dependency_graph[current_node][next_node]["weight"] += 0.1
                else:
                    self.dependency_graph.add_edge(
                        current_node, next_node, weight=0.5, relationship_type="temporal_sequence"
                    )

    async def _store_nodes_in_qdrant(self, nodes: Dict[str, DependencyNode]):
        """Store dependency nodes in Qdrant with embeddings."""
        if not HTTPX_AVAILABLE:
            print("⚠️  httpx not available, skipping Qdrant storage")
            return

        try:
            async with httpx.AsyncClient() as client:
                points = []

                for node_id, node in nodes.items():
                    # Use code embedding as primary vector, fallback to semantic
                    vector = node.code_embedding or node.semantic_embedding

                    if not vector:
                        # Create simple hash-based vector if no embeddings
                        vector = self._create_hash_vector(node_id)

                    # Create point for Qdrant
                    point = {
                        "id": self._generate_point_id(node_id),
                        "vector": vector,
                        "payload": {
                            "memory_type": "dependency_node",
                            "node_id": node_id,
                            "function_signature": node.function_signature,
                            "module_path": node.module_path,
                            "dependencies": node.dependencies,
                            "execution_context": node.execution_context,
                            "performance_profile": node.performance_profile,
                            "has_code_embedding": node.code_embedding is not None,
                            "has_semantic_embedding": node.semantic_embedding is not None,
                            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "timestamp_unix": time.time(),
                            "graph_version": int(time.time()),  # For versioning
                        },
                    }
                    points.append(point)

                # Batch upsert to Qdrant
                if points:
                    response = await client.put(
                        f"{self.qdrant_url}/collections/{self.collection_name}/points",
                        json={"points": points},
                        timeout=30.0,
                    )

                    if response.status_code == 200:
                        print(f"✅ Stored {len(points)} dependency nodes in Qdrant")
                    else:
                        print(f"❌ Failed to store nodes: {response.status_code}")

        except Exception as e:
            print(f"⚠️  Failed to store in Qdrant: {e}")

    def _create_hash_vector(self, node_id: str, dimensions: int = 768) -> List[float]:
        """Create a hash-based vector when no embeddings are available."""
        # Create deterministic hash-based vector
        hash_obj = hashlib.md5(node_id.encode())
        hash_bytes = hash_obj.digest()

        # Convert to normalized vector
        vector = []
        for i in range(dimensions):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1  # Normalize to [-1, 1]
            vector.append(value)

        return vector

    def _generate_point_id(self, node_id: str) -> str:
        """Generate consistent point ID for Qdrant."""
        return f"dep_{hashlib.md5(node_id.encode()).hexdigest()[:16]}"

    async def query_similar_dependencies(self, query: GraphQuery) -> List[Dict[str, Any]]:
        """
        Query for similar dependencies using semantic/code embeddings.
        This is what an orchestrating LLM would use to find relevant code.
        """
        if not HTTPX_AVAILABLE:
            return self._fallback_graph_query(query)

        try:
            # Get embedding for query
            query_embedding = await self.embedding_integration._get_semantic_embedding(
                query.semantic_query
            )

            if not query_embedding:
                return self._fallback_graph_query(query)

            # Search Qdrant for similar nodes
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                    json={
                        "vector": query_embedding,
                        "limit": query.max_results,
                        "score_threshold": query.code_similarity_threshold,
                        "filter": {
                            "must": [{"key": "memory_type", "match": {"value": "dependency_node"}}]
                        },
                        "with_payload": True,
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    results = response.json().get("result", [])

                    # Enrich with NetworkX graph analysis
                    enriched_results = []
                    for result in results:
                        payload = result.get("payload", {})
                        node_id = payload.get("node_id")

                        if node_id and node_id in self.dependency_graph:
                            # Add graph-based metrics
                            graph_metrics = self._calculate_node_metrics(node_id)
                            payload.update(graph_metrics)

                            enriched_results.append(
                                {
                                    "similarity_score": result.get("score", 0),
                                    "node_info": payload,
                                    "dependencies": payload.get("dependencies", []),
                                    "graph_position": graph_metrics,
                                }
                            )

                    return enriched_results

        except Exception as e:
            print(f"Query failed: {e}")

        return self._fallback_graph_query(query)

    def _fallback_graph_query(self, query: GraphQuery) -> List[Dict[str, Any]]:
        """Fallback query using NetworkX when vector search isn't available."""
        results = []

        # Simple text matching on node names
        query_terms = query.semantic_query.lower().split()

        for node_id in self.dependency_graph.nodes():
            node_data = self.dependency_graph.nodes[node_id]

            # Calculate text similarity
            node_text = f"{node_id} {node_data.get('signature', '')}".lower()
            matches = sum(1 for term in query_terms if term in node_text)
            similarity = matches / len(query_terms) if query_terms else 0

            if similarity > 0.3:  # Basic threshold
                graph_metrics = self._calculate_node_metrics(node_id)

                results.append(
                    {
                        "similarity_score": similarity,
                        "node_info": {"node_id": node_id, **node_data, **graph_metrics},
                        "dependencies": list(self.dependency_graph.successors(node_id)),
                        "graph_position": graph_metrics,
                    }
                )

        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)[
            : query.max_results
        ]

    def _calculate_node_metrics(self, node_id: str) -> Dict[str, Any]:
        """Calculate NetworkX-based metrics for a node."""
        if node_id not in self.dependency_graph:
            return {}

        try:
            return {
                "in_degree": self.dependency_graph.in_degree(node_id),
                "out_degree": self.dependency_graph.out_degree(node_id),
                "betweenness_centrality": nx.betweenness_centrality(self.dependency_graph).get(
                    node_id, 0
                ),
                "pagerank": nx.pagerank(self.dependency_graph).get(node_id, 0),
                "clustering": nx.clustering(self.dependency_graph.to_undirected()).get(node_id, 0),
                "shortest_path_to_root": self._distance_to_root(node_id),
                "is_critical_path": node_id in self._find_critical_path(),
            }
        except Exception:
            return {"error": "failed_to_calculate_metrics"}

    def _distance_to_root(self, node_id: str) -> int:
        """Calculate distance to root nodes (nodes with no predecessors)."""
        try:
            root_nodes = [
                n for n in self.dependency_graph.nodes() if self.dependency_graph.in_degree(n) == 0
            ]

            if not root_nodes:
                return 0

            distances = []
            for root in root_nodes:
                try:
                    distance = nx.shortest_path_length(self.dependency_graph, root, node_id)
                    distances.append(distance)
                except nx.NetworkXNoPath:
                    continue

            return min(distances) if distances else float("inf")

        except Exception:
            return 0

    def _find_critical_path(self) -> List[str]:
        """Find the performance-critical path through the graph."""
        try:
            # Weight edges by execution time
            weighted_graph = self.dependency_graph.copy()

            for u, v, data in weighted_graph.edges(data=True):
                node_time = self.dependency_graph.nodes[v].get("avg_time_ms", 1)
                data["weight"] = -node_time  # Negative for longest path

            # Find longest path (most time-consuming)
            if weighted_graph.nodes():
                # Simple heuristic: path from highest out-degree to highest in-degree
                start_node = max(weighted_graph.nodes(), key=lambda n: weighted_graph.out_degree(n))
                end_node = max(weighted_graph.nodes(), key=lambda n: weighted_graph.in_degree(n))

                try:
                    path = nx.shortest_path(weighted_graph, start_node, end_node, weight="weight")
                    return path
                except nx.NetworkXNoPath:
                    pass

            return []

        except Exception:
            return []

    def _identify_performance_hotspots(self) -> List[Dict[str, Any]]:
        """Identify performance hotspots in the graph."""
        hotspots = []

        for node_id in self.dependency_graph.nodes():
            node_data = self.dependency_graph.nodes[node_id]
            avg_time = node_data.get("avg_time_ms", 0)
            call_freq = node_data.get("call_frequency", 1)

            # Score based on time * frequency
            hotspot_score = avg_time * call_freq

            if hotspot_score > 10:  # Threshold for significant impact
                hotspots.append(
                    {
                        "node_id": node_id,
                        "avg_time_ms": avg_time,
                        "call_frequency": call_freq,
                        "hotspot_score": hotspot_score,
                        "graph_centrality": nx.betweenness_centrality(self.dependency_graph).get(
                            node_id, 0
                        ),
                    }
                )

        return sorted(hotspots, key=lambda x: x["hotspot_score"], reverse=True)[:10]

    def _update_graph_metrics(self):
        """Update graph-level metrics."""
        self.metrics.total_nodes = self.dependency_graph.number_of_nodes()
        self.metrics.total_edges = self.dependency_graph.number_of_edges()
        self.metrics.last_updated = time.time()

        if self.dependency_graph.nodes():
            # Find most connected node
            degrees = dict(self.dependency_graph.degree())
            self.metrics.most_connected_node = max(degrees, key=degrees.get)

            # Calculate max depth
            try:
                self.metrics.max_depth = (
                    max(
                        self._distance_to_root(node)
                        for node in self.dependency_graph.nodes()
                        if self._distance_to_root(node) != float("inf")
                    )
                    if self.dependency_graph.nodes()
                    else 0
                )
            except:
                self.metrics.max_depth = 0

            # Calculate embedding coverage
            nodes_with_embeddings = sum(
                1
                for node_id in self.dependency_graph.nodes()
                if self.dependency_graph.nodes[node_id].get("has_code_embedding")
                or self.dependency_graph.nodes[node_id].get("has_semantic_embedding")
            )
            self.metrics.embedding_coverage = nodes_with_embeddings / self.metrics.total_nodes

        # Update critical path
        self.metrics.performance_critical_path = self._find_critical_path()

    def export_graph_for_llm(self, output_path: Path) -> Dict[str, Any]:
        """
        Export graph in a format optimized for LLM consumption.
        This creates a structured knowledge base the orchestrating LLM can use.
        """
        llm_graph = {
            "metadata": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": asdict(self.metrics),
                "description": "Vibelint execution dependency graph for LLM orchestration",
            },
            "nodes": {},
            "dependencies": {},
            "performance_insights": {
                "critical_path": self.metrics.performance_critical_path,
                "hotspots": self._identify_performance_hotspots(),
                "bottlenecks": self._find_bottlenecks(),
            },
            "query_examples": self._generate_query_examples(),
        }

        # Export node information
        for node_id in self.dependency_graph.nodes():
            node_data = self.dependency_graph.nodes[node_id]
            metrics = self._calculate_node_metrics(node_id)

            llm_graph["nodes"][node_id] = {
                "signature": node_data.get("signature", ""),
                "module": node_data.get("module", ""),
                "performance": {
                    "avg_time_ms": node_data.get("avg_time_ms", 0),
                    "call_frequency": node_data.get("call_frequency", 1),
                    "io_operations": node_data.get("io_operations", 0),
                },
                "graph_metrics": metrics,
                "dependencies": list(self.dependency_graph.successors(node_id)),
                "dependents": list(self.dependency_graph.predecessors(node_id)),
            }

        # Export dependency relationships
        for node_id in self.dependency_graph.nodes():
            deps = list(self.dependency_graph.successors(node_id))
            if deps:
                llm_graph["dependencies"][node_id] = deps

        # Save to file
        with open(output_path, "w") as f:
            json.dump(llm_graph, f, indent=2, default=str)

        print(f"📊 LLM-optimized graph exported to: {output_path}")
        return llm_graph

    def _find_bottlenecks(self) -> List[Dict[str, Any]]:
        """Find potential bottlenecks in the execution graph."""
        bottlenecks = []

        try:
            # Nodes with high betweenness centrality are potential bottlenecks
            centrality = nx.betweenness_centrality(self.dependency_graph)

            for node_id, centrality_score in centrality.items():
                if centrality_score > 0.1:  # Significant centrality
                    node_data = self.dependency_graph.nodes[node_id]

                    bottlenecks.append(
                        {
                            "node_id": node_id,
                            "centrality_score": centrality_score,
                            "avg_time_ms": node_data.get("avg_time_ms", 0),
                            "in_degree": self.dependency_graph.in_degree(node_id),
                            "out_degree": self.dependency_graph.out_degree(node_id),
                            "bottleneck_type": "high_centrality",
                        }
                    )

        except Exception:
            pass

        return sorted(bottlenecks, key=lambda x: x["centrality_score"], reverse=True)

    def _generate_query_examples(self) -> List[Dict[str, str]]:
        """Generate example queries an orchestrating LLM might use."""
        return [
            {
                "query": "Find functions related to configuration loading",
                "semantic_query": "configuration loading config file parsing settings",
                "use_case": "When user asks to modify configuration behavior",
            },
            {
                "query": "Find performance-critical validation functions",
                "semantic_query": "validation performance critical slow execution time",
                "use_case": "When optimizing validation performance",
            },
            {
                "query": "Find functions that handle file I/O operations",
                "semantic_query": "file input output read write operations filesystem",
                "use_case": "When debugging file-related issues",
            },
            {
                "query": "Find error handling and exception management",
                "semantic_query": "error handling exception management try catch error",
                "use_case": "When improving error handling robustness",
            },
        ]


# Integration function for vibelint self-improvement
async def build_vibelint_dependency_graph(
    trace_sessions: List[TraceSession], config: Dict[str, Any] = None
) -> DependencyGraphManager:
    """
    Build a comprehensive dependency graph from multiple trace sessions.
    This is the main integration point for vibelint's self-improvement system.
    """
    print("🔗 Building comprehensive vibelint dependency graph...")

    graph_manager = DependencyGraphManager(config=config)

    for session in trace_sessions:
        await graph_manager.build_graph_from_trace(session)

    # Export for LLM consumption
    export_path = (
        Path(__file__).parent.parent.parent / ".vibelint-self-improvement" / "dependency_graph.json"
    )
    export_path.parent.mkdir(exist_ok=True)
    graph_manager.export_graph_for_llm(export_path)

    print(
        f"📊 Graph built: {graph_manager.metrics.total_nodes} nodes, {graph_manager.metrics.total_edges} edges"
    )
    print(f"🎯 Embedding coverage: {graph_manager.metrics.embedding_coverage:.1%}")

    return graph_manager


if __name__ == "__main__":
    # Example usage
    async def demo():
        from vibelint.runtime_tracer import trace_vibelint_module

        # Trace a module
        session = trace_vibelint_module("config")

        # Build graph
        graph_manager = await build_vibelint_dependency_graph([session])

        # Query example
        query = GraphQuery(semantic_query="configuration loading and parsing", max_results=5)

        results = await graph_manager.query_similar_dependencies(query)
        print(f"Found {len(results)} similar dependencies")

    asyncio.run(demo())
