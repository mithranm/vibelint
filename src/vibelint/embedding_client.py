"""
Embedding Client for Specialized Code and Natural Language Embeddings.

This module provides a unified interface for accessing both local and remote
embedding models, with specialized endpoints for code analysis and natural
language processing.

Usage:
    from vibelint.embedding_client import EmbeddingClient

    client = EmbeddingClient()

    # Code embeddings (optimized for code similarity, patterns, architecture)
    code_embeddings = client.get_code_embeddings(["def function():", "class MyClass:"])

    # Natural language embeddings (optimized for documentation, comments)
    natural_embeddings = client.get_natural_embeddings(["docstring text", "comment text"])

vibelint/src/vibelint/embedding_client.py
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Unified embedding client supporting both specialized remote endpoints
    and local fallback models.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedding client with configuration.

        Args:
            config: Configuration dictionary with embedding settings
        """
        self.config = config or {}
        self._load_configuration()
        self._initialize_clients()

    def _load_configuration(self):
        """Load configuration from various sources."""
        from .llm.llm_config import get_embedding_config

        # Use typed configuration with explicit validation
        embedding_config = get_embedding_config()

        self.code_api_url = embedding_config.code_api_url
        self.natural_api_url = embedding_config.natural_api_url
        self.code_model = embedding_config.code_model
        self.natural_model = embedding_config.natural_model
        self.use_specialized = embedding_config.use_specialized_embeddings
        self.similarity_threshold = embedding_config.get("similarity_threshold", 0.85)

        # Load API keys from environment
        self.code_api_key = os.getenv("CODE_EMBEDDING_API_KEY")
        self.natural_api_key = os.getenv("NATURAL_EMBEDDING_API_KEY")

        # Local model fallback
        self.local_model_name = embedding_config.get("local_model", "google/embeddinggemma-300m")

    def _initialize_clients(self):
        """Initialize embedding clients."""
        self._local_model = None

        # Check if we can use specialized endpoints
        # Allow endpoints without API keys for internal services
        self._can_use_code_api = bool(self.code_api_url and self.use_specialized)
        self._can_use_natural_api = bool(self.natural_api_url and self.use_specialized)

        # Initialize local model as fallback
        if not (self._can_use_code_api and self._can_use_natural_api):
            self._initialize_local_model()

        logger.info("Embedding client initialized:")
        logger.info(f"  Code API: {'✓' if self._can_use_code_api else '✗'}")
        logger.info(f"  Natural API: {'✓' if self._can_use_natural_api else '✗'}")
        logger.info(f"  Local fallback: {'✓' if self._local_model else '✗'}")

    def _initialize_local_model(self):
        """Initialize local embedding model as fallback."""
        try:
            try:
                from sentence_transformers import SentenceTransformer

                self._local_model = SentenceTransformer(self.local_model_name)
                logger.info(f"Local embedding model loaded: {self.local_model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, remote endpoints required")
        except Exception as e:
            logger.warning(f"Failed to load local model {self.local_model_name}: {e}")

    def _call_remote_api(
        self, api_url: str, api_key: str, model: str, texts: List[str]
    ) -> List[List[float]]:
        """
        Call remote embedding API.

        Args:
            api_url: API endpoint URL
            api_key: API authentication key
            model: Model name to use
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {"input": texts, "model": model}

        try:
            response = requests.post(
                f"{api_url}/v1/embeddings", headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]

            logger.debug(f"Generated {len(embeddings)} embeddings via {api_url}")
            return embeddings

        except Exception as e:
            logger.warning(f"Remote embedding API failed ({api_url}): {e}")
            raise

    def _get_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings using local model.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._local_model:
            raise RuntimeError("Local embedding model not available")

        embeddings = self._local_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    def get_code_embeddings(self, code_texts: List[str]) -> List[List[float]]:
        """
        Get embeddings optimized for code analysis.

        Args:
            code_texts: List of code snippets to embed

        Returns:
            List of embedding vectors optimized for code similarity
        """
        if self._can_use_code_api:
            try:
                return self._call_remote_api(
                    self.code_api_url, self.code_api_key, self.code_model, code_texts
                )
            except Exception as e:
                logger.warning(f"Code API failed, falling back to local: {e}")

        # Fallback to local model
        return self._get_local_embeddings(code_texts)

    def get_natural_embeddings(self, natural_texts: List[str]) -> List[List[float]]:
        """
        Get embeddings optimized for natural language analysis.

        Args:
            natural_texts: List of natural language texts to embed

        Returns:
            List of embedding vectors optimized for natural language understanding
        """
        if self._can_use_natural_api:
            try:
                return self._call_remote_api(
                    self.natural_api_url, self.natural_api_key, self.natural_model, natural_texts
                )
            except Exception as e:
                logger.warning(f"Natural API failed, falling back to local: {e}")

        # Fallback to local model
        return self._get_local_embeddings(natural_texts)

    def get_embeddings(self, texts: List[str], content_type: str = "mixed") -> List[List[float]]:
        """
        Get embeddings with automatic routing based on content type.

        Args:
            texts: List of texts to embed
            content_type: "code", "natural", or "mixed"

        Returns:
            List of embedding vectors
        """
        if content_type == "code":
            return self.get_code_embeddings(texts)
        elif content_type == "natural":
            return self.get_natural_embeddings(texts)
        else:
            # For mixed content, use natural language embeddings as default
            return self.get_natural_embeddings(texts)

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        import numpy as np

        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def find_similar_pairs(
        self, texts: List[str], content_type: str = "mixed", threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Find pairs of texts that exceed similarity threshold.

        Args:
            texts: List of texts to analyze
            content_type: Type of content for optimal embedding selection
            threshold: Similarity threshold (uses config default if None)

        Returns:
            List of similar pairs with metadata
        """
        if threshold is None:
            threshold = self.similarity_threshold

        embeddings = self.get_embeddings(texts, content_type)
        similar_pairs = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = self.compute_similarity(embeddings[i], embeddings[j])

                if similarity >= threshold:
                    similar_pairs.append(
                        {
                            "index1": i,
                            "index2": j,
                            "text1": texts[i],
                            "text2": texts[j],
                            "similarity": similarity,
                            "content_type": content_type,
                        }
                    )

        # Sort by similarity descending
        similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_pairs
