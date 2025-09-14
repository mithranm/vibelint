"""
Semantic similarity validator using EmbeddingGemma for detecting redundant code patterns.

Uses embedding models to find semantically similar docstrings, functions, and classes
that may indicate architectural redundancy or code duplication.

vibelint/validators/semantic_similarity.py
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

    # Create dummy np module for type hints
    class DummyNumpy:
        class ndarray:
            pass

    np = DummyNumpy()

from ...config import Config
from ...plugin_system import BaseValidator, Finding, Severity

__all__ = ["SemanticSimilarityValidator"]
logger = logging.getLogger(__name__)


class SemanticSimilarityValidator(BaseValidator):
    """
    Validates code for semantic similarity patterns that indicate potential redundancy.

    Uses EmbeddingGemma to find:
    - Functions with similar docstrings but different implementations
    - Classes that serve similar purposes
    - Code patterns that are semantically duplicate
    """

    rule_id = "SEMANTIC-SIMILARITY"

    def __init__(
        self,
        severity: Optional[Severity] = None,
        config: Optional[Dict] = None,
        shared_model: Optional[SentenceTransformer] = None,
    ):
        super().__init__(severity, config)
        self._model: Optional[SentenceTransformer] = shared_model
        self._code_cache: Dict[str, List[Tuple[Path, str, str, np.ndarray]]] = {}
        self._setup_attempted = False

    def _setup_embedding_model(self, config: Config) -> bool:
        """
        Initialize the embedding model if available and configured.

        Returns:
            True if model is available and ready, False otherwise.
        """
        if self._setup_attempted:
            return self._model is not None

        self._setup_attempted = True

        # If we already have a shared model, use it
        if self._model is not None:
            embedding_config = config.get("embedding_analysis", {})
            self._similarity_threshold = embedding_config.get("similarity_threshold", 0.85)
            logger.info("Using pre-loaded embedding model")
            return True

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.debug(
                "Semantic similarity analysis disabled: sentence-transformers not available"
            )
            return False

        # Check configuration
        embedding_config = config.get("embedding_analysis", {})
        model_name = embedding_config.get("model", "google/embeddinggemma-300m")
        similarity_threshold = embedding_config.get("similarity_threshold", 0.85)

        # Check if embedding analysis is enabled
        if not embedding_config.get("enabled", False):
            logger.debug("Semantic similarity analysis disabled in configuration")
            return False

        try:
            # Handle HF token from config, .env file, or environment
            hf_token = embedding_config.get("hf_token")
            if not hf_token:
                import os

                # Try to load from .env file
                env_file = config.project_root / ".env" if config.project_root else None
                if env_file and env_file.exists():
                    for line in env_file.read_text().splitlines():
                        if line.startswith("HF_TOKEN="):
                            hf_token = line.split("=", 1)[1].strip().strip("\"'")
                            break
                # Fallback to environment variable
                if not hf_token:
                    hf_token = os.getenv("HF_TOKEN")

            if hf_token:
                # Set HF token for this session
                os.environ["HF_TOKEN"] = hf_token

            logger.info(f"Loading embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
            self._similarity_threshold = similarity_threshold
            logger.info(f"Semantic similarity analysis enabled (threshold: {similarity_threshold})")
            return True
        except Exception as e:
            logger.warning(f"Failed to load embedding model {model_name}: {e}")
            return False

    def _extract_code_elements(self, file_path: Path, content: str) -> List[Tuple[str, str, str]]:
        """
        Extract functions and classes with their docstrings for analysis.

        Returns:
            List of (element_type, name, content) tuples where content includes docstring
        """
        elements = []
        lines = content.splitlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for function or class definition
            if line.startswith("def ") or line.startswith("class "):
                element_type = "function" if line.startswith("def ") else "class"
                name = (
                    line.split("(")[0]
                    .split(":")[0]
                    .replace("def ", "")
                    .replace("class ", "")
                    .strip()
                )

                # Extract the full definition including docstring
                element_lines = [lines[i]]
                i += 1

                # Find the docstring (if any) and some implementation
                in_docstring = False
                docstring_delimiter = None
                implementation_lines = 0

                while i < len(lines) and implementation_lines < 10:  # Limit to avoid huge extracts
                    current_line = lines[i].strip()
                    element_lines.append(lines[i])

                    # Detect docstring start
                    if not in_docstring and (
                        current_line.startswith('"""') or current_line.startswith("'''")
                    ):
                        in_docstring = True
                        docstring_delimiter = '"""' if current_line.startswith('"""') else "'''"
                        if current_line.count(docstring_delimiter) >= 2:  # Single line docstring
                            in_docstring = False
                    elif in_docstring and docstring_delimiter in current_line:
                        in_docstring = False
                    elif not in_docstring and current_line and not current_line.startswith("#"):
                        implementation_lines += 1

                    i += 1

                    # Stop if we hit another function/class or unindented content
                    if (
                        i < len(lines)
                        and not lines[i].startswith(" ")
                        and not lines[i].startswith("\t")
                    ):
                        if lines[i].strip() and not lines[i].strip().startswith("#"):
                            break

                element_content = "\n".join(element_lines)
                elements.append((element_type, name, element_content))
            else:
                i += 1

        return elements

    def _get_embedding(self, text: str, task_type: str = "clustering") -> Optional[np.ndarray]:
        """Generate embedding for text using task-specific prompting."""
        if not self._model:
            return None

        try:
            # Use EmbeddingGemma task-specific prompts
            if task_type == "similarity":
                prompt = f"task: sentence similarity | query: {text}"
            elif task_type == "clustering":
                prompt = f"task: clustering | query: {text}"
            elif task_type == "code":
                prompt = f"task: code retrieval | query: {text}"
            else:
                prompt = text

            embedding = self._model.encode(prompt, normalize_embeddings=True)
            return embedding
        except Exception as e:
            logger.debug(f"Failed to generate embedding: {e}")
            return None

    def _get_embeddings_batch(
        self, texts: List[str], task_type: str = "clustering"
    ) -> Optional[np.ndarray]:
        """Generate embeddings for multiple texts using batch processing for efficiency."""
        if not self._model or not texts:
            return None

        try:
            # Use EmbeddingGemma task-specific prompts for all texts
            prompts = []
            for text in texts:
                if task_type == "similarity":
                    prompt = f"task: sentence similarity | query: {text}"
                elif task_type == "clustering":
                    prompt = f"task: clustering | query: {text}"
                elif task_type == "code":
                    prompt = f"task: code retrieval | query: {text}"
                else:
                    prompt = text
                prompts.append(prompt)

            # Batch encode for efficiency
            embeddings = self._model.encode(prompts, normalize_embeddings=True, batch_size=32)
            return embeddings
        except Exception as e:
            logger.debug(f"Failed to generate batch embeddings: {e}")
            return None

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2))

    def validate(self, file_path: Path, content: str, config: Config) -> Iterator[Finding]:
        """
        Perform semantic similarity analysis on the current file.

        Compares functions and classes in this file against previously seen ones
        to detect semantic redundancy.
        """
        if not self._setup_embedding_model(config):
            return

        # Extract code elements from current file
        elements = self._extract_code_elements(file_path, content)
        if not elements:
            return

        # Batch encode all elements at once for efficiency
        element_contents = [element_content for _, _, element_content in elements]
        embeddings = self._get_embeddings_batch(element_contents, "code")

        if embeddings is None or len(embeddings) != len(elements):
            return

        for (element_type, name, element_content), embedding in zip(elements, embeddings):
            # Check against cached elements of the same type
            cache_key = element_type
            if cache_key not in self._code_cache:
                self._code_cache[cache_key] = []

            # Compare against existing elements
            for cached_file, cached_name, cached_content, cached_embedding in self._code_cache[
                cache_key
            ]:
                # Skip if same file and same name
                if cached_file == file_path and cached_name == name:
                    continue

                similarity = self._compute_similarity(embedding, cached_embedding)

                if similarity >= self._similarity_threshold:
                    # Generate a unique rule ID based on similarity type
                    rule_suffix = f"{element_type.upper()}-SIMILARITY"

                    # Create descriptive message
                    if cached_file == file_path:
                        message = f"Similar {element_type}s '{name}' and '{cached_name}' found in same file (similarity: {similarity:.3f})"
                        recommendation = f"Consider consolidating similar {element_type}s or renaming if they serve different purposes"
                    else:
                        relative_cached = cached_file.name
                        message = f"{element_type.title()} '{name}' is very similar to '{cached_name}' in {relative_cached} (similarity: {similarity:.3f})"
                        recommendation = f"Consider consolidating duplicate {element_type}s across files or documenting the differences"

                    yield Finding(
                        rule_id=f"{self.rule_id}-{rule_suffix}",
                        message=f"{message}. Recommendation: {recommendation}",
                        file_path=file_path,
                        line=1,  # Could be enhanced to find actual line number
                        severity=self.severity,
                    )

            # Add current element to cache for future comparisons
            self._code_cache[cache_key].append((file_path, name, element_content, embedding))

    def _cleanup_cache(self):
        """Clean up cache to prevent memory issues."""
        # Keep only the most recent 100 elements per type
        for cache_key in self._code_cache:
            if len(self._code_cache[cache_key]) > 100:
                self._code_cache[cache_key] = self._code_cache[cache_key][-100:]
