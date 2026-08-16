"""Local multilingual embedding model.

Uses ``intfloat/multilingual-e5-small`` (XLM-RoBERTa based, standard
architecture) so it works with any recent transformers release. Loaded
directly with ``transformers`` (no sentence-transformers), so this service
stays self-contained in its own venv.

Embeddings are L2-normalised mean-pooled vectors (384 dims), which is the
expected format for cosine-similarity retrieval in ChromaDB.

The model can be overridden through the ``LOCAL_EMBEDDING_MODEL`` env var.
"""
import os
from typing import List, Union

import torch
from transformers import AutoModel, AutoTokenizer

# Read the embedding model from the environment so this service stays
# independent of the main RAG API's settings module.
_DEFAULT_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
_MAX_LENGTH = int(os.environ.get("LOCAL_EMBEDDING_MAX_LENGTH", "512"))


class PersianEmbedder:
    def __init__(self, model_name: str = _DEFAULT_MODEL, max_length: int = _MAX_LENGTH):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Encode one or more texts into L2-normalised mean-pooled embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, S, D)

        # Mean pooling weighted by the attention mask
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        # L2 normalisation (standard for embedding/retrieval models)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.tolist()


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = PersianEmbedder()
        # warm up so the first request isn't slow / doesn't surface first-run errors
        _ = _embedder.embed("warmup")
    return _embedder