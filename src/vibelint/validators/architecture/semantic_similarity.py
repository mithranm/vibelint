"""
Semantic similarity discovery heuristic using EmbeddingGemma.

This validator serves as a DISCOVERY TOOL rather than a definitive problem indicator.
It identifies potentially redundant code patterns by analyzing semantic similarity of
docstrings, functions, and classes using local embedding models.

Primary use cases:
1. Human-driven redundancy exploration (INFO level findings)
2. Prioritization heuristic for expensive LLM analysis
3. Architectural refactoring planning

The findings are informational - high similarity doesn't always indicate problems,
but helps focus attention on areas that may benefit from consolidation or refactoring.

vibelint/validators/semantic_similarity.py
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

try:
    import importlib.util

    SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("numpy") is not None
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

if TYPE_CHECKING or SENTENCE_TRANSFORMERS_AVAILABLE:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = None

from ...plugin_system import BaseValidator, Finding, Severity
from ...embedding_client import EmbeddingClient

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
    default_severity = Severity.INFO

    def __init__(
        self,
        severity: Optional[Severity] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(severity, config)
        # Get shared model from config if available
        self._model: Optional[Any] = config.get("_shared_model") if config else None
        self._code_cache: Dict[str, List[Tuple[Path, str, str, Any]]] = {}
        self._setup_attempted = False

    def _safe_config_get(self, config: Any, key: str, default=None):
        """Safely get a value from config object."""
        if hasattr(config, "get"):
            return config.get(key, default)
        elif hasattr(config, "__getitem__"):
            try:
                return config[key]
            except (KeyError, TypeError):
                return default
        else:
            return default

    def _setup_embedding_model(self, config: Any) -> bool:
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
            embedding_config = self._safe_config_get(config, "embedding_analysis", {})
            self._similarity_threshold = self._safe_config_get(
                embedding_config, "similarity_threshold", 0.85
            )
            logger.info("Using pre-loaded embedding model")
            return True

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.debug(
                "Semantic similarity analysis disabled: sentence-transformers not available"
            )
            return False

        # Check configuration
        embedding_config = self._safe_config_get(config, "embedding_analysis", {})
        embeddings_config = self._safe_config_get(config, "embeddings", {})

        # Use new embeddings config if available, otherwise fall back to legacy config
        if embeddings_config:
            similarity_threshold = self._safe_config_get(embeddings_config, "similarity_threshold", 0.85)
            enabled = self._safe_config_get(embeddings_config, "use_specialized_embeddings", True)
        else:
            model_name = self._safe_config_get(embedding_config, "model", "google/embeddinggemma-300m")
            similarity_threshold = self._safe_config_get(embedding_config, "similarity_threshold", 0.85)
            enabled = self._safe_config_get(embedding_config, "enabled", False)

        # Check if embedding analysis is enabled
        if not enabled:
            logger.debug("Semantic similarity analysis disabled in configuration")
            return False

        try:
            # Try to initialize the new EmbeddingClient for specialized endpoints
            try:
                self._embedding_client = EmbeddingClient(config)
                self._similarity_threshold = similarity_threshold
                logger.info(f"Initialized EmbeddingClient with specialized endpoints (threshold: {similarity_threshold})")

                # Keep legacy _model for backward compatibility
                if hasattr(self._embedding_client, '_local_model'):
                    self._model = self._embedding_client._local_model
                return True
            except Exception as client_error:
                logger.debug(f"EmbeddingClient initialization failed: {client_error}")

            # Fallback to legacy local model approach
            if embeddings_config:
                logger.debug("Falling back to legacy embedding model approach")
                return False

            # Handle HF token from config, .env file, or environment
            hf_token = self._safe_config_get(embedding_config, "hf_token")
            if not hf_token:
                import os

                # Try to load from .env file
                env_file = getattr(config, "project_root", None)
                if env_file:
                    env_file = env_file / ".env"
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
            if SentenceTransformer is not None:
                self._model = SentenceTransformer(model_name)
            self._similarity_threshold = similarity_threshold
            logger.info(f"Semantic similarity analysis enabled (threshold: {similarity_threshold})")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize embedding system: {e}")
            return False

    def _extract_code_elements(self, file_path: Path, content: str) -> List[Tuple[str, str, str]]:
        """
        Extract functions and classes with ONLY their docstrings for analysis.

        This focuses on semantic intent rather than implementation details,
        making it much more effective at finding truly redundant code.

        Returns:
            List of (element_type, name, docstring_content) tuples
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

                # Look for docstring only - skip implementation entirely
                i += 1
                docstring_content = ""

                # Skip until we find the first non-empty, non-comment line
                while i < len(lines):
                    current_line = lines[i].strip()
                    if current_line and not current_line.startswith("#"):
                        break
                    i += 1

                # Check if it's a docstring
                if i < len(lines):
                    current_line = lines[i].strip()
                    if current_line.startswith('"""') or current_line.startswith("'''"):
                        docstring_delimiter = '"""' if current_line.startswith('"""') else "'''"

                        # Handle single-line docstring
                        if current_line.count(docstring_delimiter) >= 2:
                            docstring_content = current_line.strip(docstring_delimiter).strip()
                        else:
                            # Multi-line docstring
                            docstring_lines = []
                            if len(current_line) > 3:  # Content on same line as opening quotes
                                docstring_lines.append(current_line[3:])

                            i += 1
                            while i < len(lines):
                                line_content = lines[i].rstrip()
                                if docstring_delimiter in line_content:
                                    # End of docstring
                                    final_content = line_content.split(docstring_delimiter)[0]
                                    if final_content.strip():
                                        docstring_lines.append(final_content)
                                    break
                                docstring_lines.append(line_content)
                                i += 1

                            docstring_content = "\n".join(docstring_lines).strip()

                # Only include elements that have substantial docstrings
                if (
                    docstring_content and len(docstring_content.strip()) > 20
                ):  # Meaningful docstring
                    elements.append((element_type, name, docstring_content))
                # Elements without docstrings will be caught by DOCSTRING-MISSING rule

            i += 1

        return elements

    def _get_embedding(self, text: str, task_type: str = "clustering") -> Optional[Any]:
        """Generate embedding for text using task-specific prompting."""
        try:
            # Try using the new EmbeddingClient first
            if hasattr(self, '_embedding_client'):
                # Determine content type for specialized routing
                if task_type == "code":
                    embeddings = self._embedding_client.get_code_embeddings([text])
                else:
                    embeddings = self._embedding_client.get_natural_embeddings([text])

                if embeddings:
                    return embeddings[0]

            # Fallback to legacy local model
            if not self._model:
                return None

            # Use EmbeddingGemma task-specific prompts for local model
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
    ) -> Optional[Any]:
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

    def _compute_similarity(self, embedding1: Any, embedding2: Any) -> float:
        """Compute cosine similarity between two embeddings."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            import numpy as np

            return float(np.dot(embedding1, embedding2))
        else:
            return 0.0

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """
        Perform semantic similarity analysis on the current file.

        Compares functions and classes in this file against previously seen ones
        to detect semantic redundancy.
        """
        if config is None or not self._setup_embedding_model(config):
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
            for cached_file, cached_name, _, cached_embedding in self._code_cache[cache_key]:
                # Skip if same file and same name
                if cached_file == file_path and cached_name == name:
                    continue

                similarity = self._compute_similarity(embedding, cached_embedding)

                threshold = getattr(self, "_similarity_threshold", 0.85)
                if similarity >= threshold:
                    # Create descriptive message
                    if cached_file == file_path:
                        message = f"Similar {element_type}s '{name}' and '{cached_name}' found in same file (similarity: {similarity:.3f})"
                        recommendation = f"Consider consolidating similar {element_type}s or renaming if they serve different purposes"
                    else:
                        relative_cached = cached_file.name
                        message = f"{element_type.title()} '{name}' is very similar to '{cached_name}' in {relative_cached} (similarity: {similarity:.3f})"
                        recommendation = f"Consider consolidating duplicate {element_type}s across files or documenting the differences"

                    yield self.create_finding(
                        message=f"{message}. Recommendation: {recommendation}",
                        file_path=file_path,
                        line=1,  # Could be enhanced to find actual line number
                    )

            # Add current element to cache for future comparisons
            self._code_cache[cache_key].append((file_path, name, element_content, embedding))

    def _cleanup_cache(self):
        """Clean up cache to prevent memory issues."""
        # Keep only the most recent 100 elements per type
        for cache_key in self._code_cache:
            if len(self._code_cache[cache_key]) > 100:
                self._code_cache[cache_key] = self._code_cache[cache_key][-100:]
