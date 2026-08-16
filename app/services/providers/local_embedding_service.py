"""HTTP-backed local embedding provider.

When ``EMBEDDING_PROVIDER=local`` this delegates embedding to a separate
microservice (see ``embedding_service/``) that runs the Persian GTE model
(``xmanii/maux-gte-persian``) in its own venv. This keeps the heavy ML
dependencies (torch / transformers) out of the main RAG API entirely.

The chat provider (e.g. OpenRouter) is unaffected and is still used for
``create_chat_completion`` / streaming.

ChromaDB compatibility:
------------------------
The ``HTTPChromaEmbeddingFunction`` follows the ChromaDB embedding-function
protocol:
    - ``name()``            -> classmethod, registered name
    - ``get_config()``      -> instance method, serialised config
    - ``build_from_config`` -> classmethod, reconstruct from persisted config
    - ``__call__``          -> embed a list of documents

It is registered with ``@register_embedding_function`` so ChromaDB can
deserialise it from the persisted collection configuration.
"""
from typing import Any, Dict, List

import requests

from chromadb.api.types import EmbeddingFunction, Space
from chromadb.utils.embedding_functions import register_embedding_function

from app.config.settings import settings

_DEFAULT_API_URL = "http://127.0.0.1:9010/embed"


class LocalEmbeddingProvider:
    """Embedding provider backed by the local embedding microservice."""

    def __init__(self):
        self.api_url = settings.LOCAL_EMBEDDING_API_URL or _DEFAULT_API_URL

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


@register_embedding_function
class HTTPChromaEmbeddingFunction(EmbeddingFunction):
    """A ChromaDB-compatible embedding function that calls the local service.

    ChromaDB calls ``embedding_function(documents)`` (a list of strings) to
    compute embeddings at storage time. Delegating this to the local service
    guarantees the stored document embeddings live in the same space as the
    query embeddings returned by :class:`LocalEmbeddingProvider`.
    """

    def __init__(self, api_url: str = _DEFAULT_API_URL):
        self.api_url = api_url

    @classmethod
    def name(cls) -> str:
        return "http_local_embedding"

    def get_config(self) -> Dict[str, Any]:
        return {"api_url": self.api_url}

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "HTTPChromaEmbeddingFunction":
        return cls(api_url=config.get("api_url", _DEFAULT_API_URL))

    def default_space(self) -> Space:
        # GTE-style Persian embeddings work best with cosine similarity
        return "cosine"

    def supported_spaces(self) -> List[Space]:
        return ["cosine"]

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