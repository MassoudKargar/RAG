"""HTTP-backed local embedding provider.

When ``EMBEDDING_PROVIDER=local`` this delegates embedding to a separate
microservice (see ``embedding_service/``) that runs the Persian GTE model
(``xmanii/maux-gte-persian``) in its own venv. This keeps the heavy ML
dependencies (torch / transformers) out of the main RAG API entirely.

The chat provider (e.g. OpenRouter) is unaffected and is still used for
``create_chat_completion`` / streaming.
"""
from typing import List, Union, Dict, Any
import requests
from app.config.settings import settings


class LocalEmbeddingProvider:
    """Embedding provider backed by the local embedding microservice."""

    def __init__(self):
        self.api_url = settings.LOCAL_EMBEDDING_API_URL

    def create_embedding(self, text: str) -> List[float]:
        """Return a single embedding vector via the local service."""
        response = requests.post(
            self.api_url,
            json={"texts": [text]},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]

    def create_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for multiple texts via the local service.
        
        Uses the local microservice's batch endpoint for efficiency.
        Falls back to individual embeddings if batch fails.
        """
        response = requests.post(
            self.api_url,
            json={"texts": texts},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]


class HTTPChromaEmbeddingFunction:
    """A ChromaDB-compatible embedding function that calls the local service.

    ChromaDB calls ``embedding_function(documents)`` (a list of strings) to
    compute embeddings at storage time. Delegating this to the local service
    guarantees the stored document embeddings live in the same space as the
    query embeddings returned by :class:`LocalEmbeddingProvider`.
    """

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.model_name = settings.LOCAL_EMBEDDING_MODEL
        self.is_legacy = False  # Mark as new-style embedding function

    def __call__(self, input: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            json={"texts": input},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    @staticmethod
    def name() -> str:
        """Return the name of this embedding function for ChromaDB."""
        return "http_local_embedding"

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        return {"api_url": self.api_url}
    
    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "HTTPChromaEmbeddingFunction":
        """Reconstruct from configuration."""
        instance = cls(api_url=config.get("api_url", settings.LOCAL_EMBEDDING_API_URL))
        return instance
    
    @staticmethod
    def default_metadata() -> Dict[str, Any]:
        """Return default metadata."""
        return {}
    
    @property
    def supported_spaces(self):
        """Return supported spaces."""
        return ["cosine", "l2", "ip"]
    
    def default_space(self):
        """Return default space."""
        return "cosine"


# Module-level singletons (mirrors the pattern of the other providers).
local_embedding_provider = LocalEmbeddingProvider()
