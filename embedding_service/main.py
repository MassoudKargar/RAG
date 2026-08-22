"""Local Persian embedding microservice.

Run with its own venv (see embedding_service/requirements.txt):
    .venv-emb/bin/uvicorn embedding_service.main:app --host 127.0.0.1 --port 8010

Endpoints:
    GET  /health      -> {"status": "ok", "model": "..."}
    POST /embed       -> {"texts": ["..."]} -> {"embeddings": [[...], ...]}
    POST /embed_batch -> {"texts": ["..."]} -> {"embeddings": [[...], ...]}
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from embedding_service.embedder import get_embedder

app = FastAPI(title="Maux Local Embedding Service", version="0.1.0")


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedBatchRequest(BaseModel):
    texts: List[str]
    batch_size: int = 32


@app.get("/health")
async def health():
    model_name = "xmanii/maux-gte-persian"
    return {"status": "ok", "model": model_name}


@app.post("/embed")
async def embed(req: EmbedRequest):
    """Embed a single text (backward compatible)."""
    embedder = get_embedder()
    embeddings = embedder.embed([req.texts[0]] if req.texts else [""])
    return {"embeddings": embeddings, "model": embedder.model_name}


@app.post("/embed_batch")
async def embed_batch(req: EmbedBatchRequest):
    """Embed multiple texts in batches for efficiency."""
    embedder = get_embedder()
    all_embeddings = embedder.embed(req.texts)
    return {
        "embeddings": all_embeddings,
        "model": embedder.model_name,
        "count": len(all_embeddings)
    }