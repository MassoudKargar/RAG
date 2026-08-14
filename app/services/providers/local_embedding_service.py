"""HTTP-backed local embedding provider.

When ``EMBEDDING_PROVIDER=local`` this delegates embedding to a separate
microservice (see ``embedding_service/``) that runs the Persian GTE model
(``xmanii/maux-gte-persian``) in its own venv. This keeps the heavy ML
dependencies (torch / transformers) out of the main RAG API entirely.

The chat provider (e.g. OpenRouter) is unaffected and is still used for
``create_chat_completion`` / streaming.
"""
from typing import List
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

    def __call__(self, input: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            json={"texts": input},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embeddings"]


# Module-level singletons (mirrors the pattern of the other providers).
local_embedding_provider = LocalEmbeddingProvider()
