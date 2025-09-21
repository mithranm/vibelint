"""
Vector Database Adapter for vibelint.

Provides a unified interface for vector storage and similarity search
across multiple backends: in-memory, Qdrant, Pinecone.

Architecture:
- VectorStore: Abstract base class defining the interface
- InMemoryVectorStore: Fast local storage using numpy/faiss
- QdrantVectorStore: Production-ready local/cloud vector database
- PineconeVectorStore: Cloud vector database for enterprise deployments

Usage:
    from vibelint.vector_store import get_vector_store

    # Auto-configured based on environment/config
    store = get_vector_store(config)

    # Store embeddings
    store.upsert("file_path", embedding, metadata={"type": "function"})

    # Search similar
    results = store.search(query_embedding, top_k=5, filter={"type": "function"})

vibelint/src/vibelint/vector_store.py
"""

import os
import json
import logging
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    id: str
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class VectorStoreConfig:
    """Configuration for vector store backends."""
    backend: str = "memory"  # "memory", "qdrant", "pinecone"

    # In-memory options
    cache_size: int = 10000
    persist_path: Optional[str] = None

    # Qdrant options
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "vibelint_embeddings"

    # Pinecone options
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west1-gcp"
    pinecone_index: str = "vibelint-embeddings"

    # Cloudflare Vectorize options
    vectorize_api_token: Optional[str] = None
    vectorize_account_id: Optional[str] = None
    vectorize_index: str = "vibelint-embeddings"

    # Common options
    dimension: int = 768
    similarity_metric: str = "cosine"  # "cosine", "euclidean", "dot"

class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.dimension = config.dimension

    @abstractmethod
    def upsert(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None) -> bool:
        """Store or update a vector with metadata."""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5,
               filter: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        pass

    @abstractmethod
    def get(self, id: str) -> Optional[VectorSearchResult]:
        """Get a specific vector by ID."""
        pass

    @abstractmethod
    def list_ids(self, filter: Optional[Dict[str, Any]] = None) -> List[str]:
        """List all vector IDs, optionally filtered."""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all vectors."""
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass

class InMemoryVectorStore(VectorStore):
    """In-memory vector store using numpy for fast similarity search."""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.persist_path = config.persist_path

        # Try to load from disk
        if self.persist_path and Path(self.persist_path).exists():
            self._load_from_disk()

    def upsert(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None) -> bool:
        """Store or update a vector with metadata."""
        try:
            embedding_array = np.array(embedding, dtype=np.float32)
            if embedding_array.shape[0] != self.dimension:
                logger.warning(f"Embedding dimension mismatch: expected {self.dimension}, got {embedding_array.shape[0]}")
                return False

            self.vectors[id] = embedding_array
            self.metadata[id] = metadata or {}

            # Persist if configured
            if self.persist_path:
                self._save_to_disk()

            return True
        except Exception as e:
            logger.error(f"Failed to upsert vector {id}: {e}")
            return False

    def search(self, query_embedding: List[float], top_k: int = 5,
               filter: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar vectors using cosine similarity."""
        if not self.vectors:
            return []

        try:
            query_array = np.array(query_embedding, dtype=np.float32)

            # Filter vectors based on metadata
            candidate_ids = self._apply_filter(filter) if filter else list(self.vectors.keys())

            if not candidate_ids:
                return []

            # Compute similarities
            similarities = []
            for vec_id in candidate_ids:
                if self.config.similarity_metric == "cosine":
                    similarity = self._cosine_similarity(query_array, self.vectors[vec_id])
                elif self.config.similarity_metric == "euclidean":
                    similarity = -np.linalg.norm(query_array - self.vectors[vec_id])  # Negative distance
                else:  # dot product
                    similarity = np.dot(query_array, self.vectors[vec_id])

                similarities.append((vec_id, similarity))

            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)

            results = []
            for vec_id, score in similarities[:top_k]:
                results.append(VectorSearchResult(
                    id=vec_id,
                    score=float(score),
                    metadata=self.metadata.get(vec_id, {}),
                    embedding=self.vectors[vec_id].tolist()
                ))

            return results

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []

    def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        try:
            if id in self.vectors:
                del self.vectors[id]
                del self.metadata[id]

                if self.persist_path:
                    self._save_to_disk()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete vector {id}: {e}")
            return False

    def get(self, id: str) -> Optional[VectorSearchResult]:
        """Get a specific vector by ID."""
        if id not in self.vectors:
            return None

        return VectorSearchResult(
            id=id,
            score=1.0,  # Perfect match
            metadata=self.metadata.get(id, {}),
            embedding=self.vectors[id].tolist()
        )

    def list_ids(self, filter: Optional[Dict[str, Any]] = None) -> List[str]:
        """List all vector IDs, optionally filtered."""
        if not filter:
            return list(self.vectors.keys())
        return self._apply_filter(filter)

    def clear(self) -> bool:
        """Clear all vectors."""
        try:
            self.vectors.clear()
            self.metadata.clear()

            if self.persist_path:
                self._save_to_disk()
            return True
        except Exception as e:
            logger.error(f"Failed to clear vectors: {e}")
            return False

    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "backend": "memory",
            "total_vectors": len(self.vectors),
            "dimension": self.dimension,
            "memory_usage_mb": sum(v.nbytes for v in self.vectors.values()) / (1024 * 1024),
            "persist_path": self.persist_path,
            "similarity_metric": self.config.similarity_metric
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _apply_filter(self, filter: Dict[str, Any]) -> List[str]:
        """Apply metadata filter to get matching vector IDs."""
        matching_ids = []

        for vec_id, metadata in self.metadata.items():
            if self._matches_filter(metadata, filter):
                matching_ids.append(vec_id)

        return matching_ids

    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def _save_to_disk(self):
        """Persist vectors and metadata to disk."""
        if not self.persist_path:
            return

        try:
            persist_dir = Path(self.persist_path).parent
            persist_dir.mkdir(parents=True, exist_ok=True)

            data = {
                "vectors": {k: v.tolist() for k, v in self.vectors.items()},
                "metadata": self.metadata,
                "dimension": self.dimension
            }

            with open(self.persist_path, 'w') as f:
                json.dump(data, f)

        except Exception as e:
            logger.warning(f"Failed to persist vectors: {e}")

    def _load_from_disk(self):
        """Load vectors and metadata from disk."""
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)

            self.vectors = {k: np.array(v, dtype=np.float32) for k, v in data["vectors"].items()}
            self.metadata = data["metadata"]

            logger.info(f"Loaded {len(self.vectors)} vectors from {self.persist_path}")

        except Exception as e:
            logger.warning(f"Failed to load vectors from disk: {e}")

class QdrantVectorStore(VectorStore):
    """Qdrant vector store adapter."""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.client = None
        self.collection_name = config.qdrant_collection

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct

            self.client = QdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key
            )

            # Create collection if it doesn't exist
            try:
                self.client.get_collection(self.collection_name)
            except:
                distance_map = {
                    "cosine": Distance.COSINE,
                    "euclidean": Distance.EUCLID,
                    "dot": Distance.DOT
                }

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=distance_map.get(config.similarity_metric, Distance.COSINE)
                    )
                )

            logger.info(f"Connected to Qdrant at {config.qdrant_url}")

        except ImportError:
            logger.error("qdrant-client not installed. Run: pip install qdrant-client")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def upsert(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None) -> bool:
        """Store or update a vector with metadata."""
        try:
            from qdrant_client.models import PointStruct

            point = PointStruct(
                id=id,
                vector=embedding,
                payload=metadata or {}
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            return True

        except Exception as e:
            logger.error(f"Failed to upsert vector {id}: {e}")
            return False

    def search(self, query_embedding: List[float], top_k: int = 5,
               filter: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Convert filter to Qdrant format
            qdrant_filter = None
            if filter:
                conditions = []
                for key, value in filter.items():
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                qdrant_filter = Filter(must=conditions)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True
            )

            return [
                VectorSearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    metadata=hit.payload or {}
                )
                for hit in results
            ]

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []

    def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[id]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector {id}: {e}")
            return False

    def get(self, id: str) -> Optional[VectorSearchResult]:
        """Get a specific vector by ID."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[id],
                with_payload=True,
                with_vectors=True
            )

            if result:
                point = result[0]
                return VectorSearchResult(
                    id=str(point.id),
                    score=1.0,
                    metadata=point.payload or {},
                    embedding=point.vector
                )
            return None

        except Exception as e:
            logger.error(f"Failed to get vector {id}: {e}")
            return None

    def list_ids(self, filter: Optional[Dict[str, Any]] = None) -> List[str]:
        """List all vector IDs, optionally filtered."""
        try:
            # Qdrant doesn't have a direct "list IDs" method, so we do a scroll
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            qdrant_filter = None
            if filter:
                conditions = []
                for key, value in filter.items():
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                qdrant_filter = Filter(must=conditions)

            result, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                with_payload=False,
                with_vectors=False
            )

            return [str(point.id) for point in result]

        except Exception as e:
            logger.error(f"Failed to list IDs: {e}")
            return []

    def clear(self) -> bool:
        """Clear all vectors."""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)

            from qdrant_client.models import Distance, VectorParams
            distance_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot": Distance.DOT
            }

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=distance_map.get(self.config.similarity_metric, Distance.COSINE)
                )
            )
            return True

        except Exception as e:
            logger.error(f"Failed to clear vectors: {e}")
            return False

    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "backend": "qdrant",
                "total_vectors": info.points_count,
                "dimension": self.dimension,
                "collection": self.collection_name,
                "url": self.config.qdrant_url,
                "similarity_metric": self.config.similarity_metric
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"backend": "qdrant", "error": str(e)}

class PineconeVectorStore(VectorStore):
    """Pinecone vector store adapter."""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.index = None
        self.index_name = config.pinecone_index

        try:
            import pinecone

            pinecone.init(
                api_key=config.pinecone_api_key,
                environment=config.pinecone_environment
            )

            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                metric_map = {
                    "cosine": "cosine",
                    "euclidean": "euclidean",
                    "dot": "dotproduct"
                }

                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=metric_map.get(config.similarity_metric, "cosine")
                )

            self.index = pinecone.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")

        except ImportError:
            logger.error("pinecone-client not installed. Run: pip install pinecone-client")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise

    def upsert(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None) -> bool:
        """Store or update a vector with metadata."""
        try:
            self.index.upsert(vectors=[(id, embedding, metadata or {})])
            return True
        except Exception as e:
            logger.error(f"Failed to upsert vector {id}: {e}")
            return False

    def search(self, query_embedding: List[float], top_k: int = 5,
               filter: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter,
                include_metadata=True
            )

            return [
                VectorSearchResult(
                    id=match.id,
                    score=match.score,
                    metadata=match.metadata or {}
                )
                for match in results.matches
            ]

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []

    def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        try:
            self.index.delete(ids=[id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector {id}: {e}")
            return False

    def get(self, id: str) -> Optional[VectorSearchResult]:
        """Get a specific vector by ID."""
        try:
            result = self.index.fetch(ids=[id])
            if id in result.vectors:
                vector_data = result.vectors[id]
                return VectorSearchResult(
                    id=id,
                    score=1.0,
                    metadata=vector_data.metadata or {},
                    embedding=vector_data.values
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get vector {id}: {e}")
            return None

    def list_ids(self, filter: Optional[Dict[str, Any]] = None) -> List[str]:
        """List all vector IDs, optionally filtered."""
        # Pinecone doesn't provide a direct list IDs method
        # This would require implementing pagination through queries
        logger.warning("list_ids not efficiently supported by Pinecone")
        return []

    def clear(self) -> bool:
        """Clear all vectors."""
        try:
            self.index.delete(delete_all=True)
            return True
        except Exception as e:
            logger.error(f"Failed to clear vectors: {e}")
            return False

    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "backend": "pinecone",
                "total_vectors": stats.total_vector_count,
                "dimension": self.dimension,
                "index_name": self.index_name,
                "environment": self.config.pinecone_environment,
                "similarity_metric": self.config.similarity_metric
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"backend": "pinecone", "error": str(e)}

class CloudflareVectorizeStore(VectorStore):
    """Cloudflare Vectorize adapter - perfect for edge deployment."""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.api_token = config.vectorize_api_token or os.getenv("VECTORIZE_API_TOKEN")
        self.account_id = config.vectorize_account_id or os.getenv("VECTORIZE_ACCOUNT_ID")
        self.index_name = config.vectorize_index

        if not self.api_token or not self.account_id:
            raise ValueError("Vectorize API token and account ID required")

        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/vectorize/indexes/{self.index_name}"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        # Create index if needed
        self._ensure_index_exists()
        logger.info(f"Connected to Cloudflare Vectorize index: {self.index_name}")

    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        try:
            import requests

            # Check if index exists
            response = requests.get(self.base_url, headers=self.headers)
            if response.status_code == 404:
                # Create index
                create_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/vectorize/indexes"
                payload = {
                    "name": self.index_name,
                    "config": {
                        "dimensions": self.dimension,
                        "metric": "cosine"  # Vectorize uses cosine by default
                    }
                }
                response = requests.post(create_url, headers=self.headers, json=payload)
                response.raise_for_status()
                logger.info(f"Created Vectorize index: {self.index_name}")

        except Exception as e:
            logger.warning(f"Failed to ensure index exists: {e}")

    def upsert(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None) -> bool:
        """Store or update a vector with metadata."""
        try:
            import requests

            payload = {
                "vectors": [
                    {
                        "id": id,
                        "values": embedding,
                        "metadata": metadata or {}
                    }
                ]
            }

            response = requests.post(
                f"{self.base_url}/upsert",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return True

        except Exception as e:
            logger.error(f"Failed to upsert vector {id}: {e}")
            return False

    def search(self, query_embedding: List[float], top_k: int = 5,
               filter: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        try:
            import requests

            payload = {
                "vector": query_embedding,
                "topK": top_k,
                "returnValues": True,
                "returnMetadata": True
            }

            # Vectorize filter format (if supported)
            if filter:
                payload["filter"] = filter

            response = requests.post(
                f"{self.base_url}/query",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            if not data.get("success", False):
                logger.error(f"Vectorize query failed: {data}")
                return []

            results = []
            for match in data.get("result", {}).get("matches", []):
                results.append(VectorSearchResult(
                    id=match["id"],
                    score=match["score"],
                    metadata=match.get("metadata", {}),
                    embedding=match.get("values")
                ))

            return results

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []

    def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        try:
            import requests

            payload = {"ids": [id]}
            response = requests.post(
                f"{self.base_url}/delete",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return True

        except Exception as e:
            logger.error(f"Failed to delete vector {id}: {e}")
            return False

    def get(self, id: str) -> Optional[VectorSearchResult]:
        """Get a specific vector by ID."""
        try:
            import requests

            payload = {"ids": [id]}
            response = requests.post(
                f"{self.base_url}/getByIds",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            if data.get("success") and data.get("result"):
                vectors = data["result"]
                if vectors:
                    vector = vectors[0]
                    return VectorSearchResult(
                        id=vector["id"],
                        score=1.0,
                        metadata=vector.get("metadata", {}),
                        embedding=vector.get("values")
                    )
            return None

        except Exception as e:
            logger.error(f"Failed to get vector {id}: {e}")
            return None

    def list_ids(self, filter: Optional[Dict[str, Any]] = None) -> List[str]:
        """List all vector IDs, optionally filtered."""
        # Vectorize doesn't provide a direct list method
        # Would need to implement via query with high top_k
        logger.warning("list_ids not efficiently supported by Vectorize")
        return []

    def clear(self) -> bool:
        """Clear all vectors."""
        try:
            import requests

            # Delete index and recreate
            response = requests.delete(self.base_url, headers=self.headers)
            if response.status_code in [200, 404]:  # OK or already deleted
                self._ensure_index_exists()
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to clear vectors: {e}")
            return False

    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            import requests

            response = requests.get(self.base_url, headers=self.headers)
            response.raise_for_status()

            data = response.json()
            if data.get("success"):
                index_info = data["result"]
                return {
                    "backend": "vectorize",
                    "index_name": self.index_name,
                    "dimension": self.dimension,
                    "account_id": self.account_id,
                    "similarity_metric": "cosine",
                    "created_on": index_info.get("created_on"),
                    "modified_on": index_info.get("modified_on")
                }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")

        return {"backend": "vectorize", "error": "Failed to get stats"}

def get_vector_store(config: Union[Dict[str, Any], VectorStoreConfig]) -> VectorStore:
    """Factory function to get appropriate vector store based on configuration."""

    if isinstance(config, dict):
        # Convert from vibelint config format
        embeddings_config = config.get("embeddings", {})
        vector_config = config.get("vector_store", {})

        store_config = VectorStoreConfig(
            backend=vector_config.get("backend", "memory"),
            dimension=embeddings_config.get("code_dimensions", 768),
            cache_size=vector_config.get("cache_size", 10000),
            persist_path=vector_config.get("persist_path"),
            qdrant_url=vector_config.get("qdrant_url", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=vector_config.get("qdrant_collection", "vibelint_embeddings"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_environment=vector_config.get("pinecone_environment", "us-west1-gcp"),
            pinecone_index=vector_config.get("pinecone_index", "vibelint-embeddings"),
            similarity_metric=vector_config.get("similarity_metric", "cosine")
        )
    else:
        store_config = config

    backend = store_config.backend.lower()

    if backend == "qdrant":
        return QdrantVectorStore(store_config)
    elif backend == "pinecone":
        return PineconeVectorStore(store_config)
    elif backend == "vectorize" or backend == "cloudflare":
        return CloudflareVectorizeStore(store_config)
    else:  # Default to memory
        return InMemoryVectorStore(store_config)