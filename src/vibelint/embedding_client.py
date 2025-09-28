"""
Stub embedding client for vibelint.

This is a placeholder to resolve import errors. In a full implementation,
this would provide embeddings for semantic similarity analysis.
"""

import logging

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Stub embedding client."""

    def __init__(self):
        logger.warning("Using stub embedding client - semantic analysis disabled")

    def get_embedding(self, text: str):
        """Return a dummy embedding."""
        return [0.0] * 384  # Common embedding dimension

    def similarity(self, text1: str, text2: str) -> float:
        """Return dummy similarity score."""
        return 0.5