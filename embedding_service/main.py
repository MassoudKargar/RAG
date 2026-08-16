"""Local Persian embedding microservice.

Run with its own venv (see embedding_service/requirements.txt):
    .venv-emb/bin/uvicorn embedding_service.main:app --host 127.0.0.1 --port 8010

Endpoints:
    GET  /health      -> {"status": "ok", "model": "..."}
    POST /embed       -> {"texts": ["..."]} -> {"embeddings": [[...], ...]}
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from embedding_service.embedder import get_embedder, _DEFAULT_MODEL

app = FastAPI(title="Maux Local Embedding Service", version="0.1.0")


class EmbedRequest(BaseModel):
    texts: List[str]


@app.get("/health")
async def health():
    return {"status": "ok", "model": _DEFAULT_MODEL}


@app.post("/embed")
async def embed(req: EmbedRequest):
    embedder = get_embedder()
    embeddings = embedder.embed(req.texts)
    return {"embeddings": embeddings, "model": embedder.model_name}
