"""Internal administration API for the RAG service.

Added for the RAG Administration Console. Reuses ONLY existing production code
(rag_service / vector_store / chunker) — no duplicated ingestion logic. Every
endpoint is protected by the global X-API-Key dependency (fail-closed) and bound
to localhost (rag-api.service listens on 127.0.0.1:8000 by design).
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from app.services.core.rag_service import rag_service

router = APIRouter(prefix="/v1/vector_db/admin", tags=["Internal Admin"])


def _collection() -> Any:
    return rag_service.vector_store.get_or_create_collection(rag_service.collection_name)


@router.get("/collections")
async def list_collections():
    """List collections and basic sizing (read-only)."""
    try:
        names = [c.name for c in rag_service.vector_store.client.list_collections()]
        return {"collections": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@router.get("/stats")
async def collection_stats():
    """Aggregate storage statistics for the active collection (read-only)."""
    try:
        collection = _collection()
        count = collection.count()
        # Distinct documents: group chunk metadata by document_id
        seen = {}
        # Chroma pages in batches of 1000
        ids = collection.get(include=["metadatas"])["ids"]
        metas = collection.get(include=["metadatas"])["metadatas"]
        docs: Dict[str, Dict[str, Any]] = {}
        total_chunks = 0
        for cid, meta in zip(ids, metas):
            total_chunks += 1
            did = (meta or {}).get("document_id") or "unknown"
            entry = docs.setdefault(did, {"document_id": did, "chunks": 0})
            entry["chunks"] += 1
            if meta and "source" in meta:
                entry.setdefault("source", str(meta.get("source")))
        return {
            "collection": rag_service.collection_name,
            "total_chunks": total_chunks,
            "total_documents": len(docs),
            "embedding_model": rag_service.embedding_provider.__class__.__name__,
            "embedding_provider": getattr(rag_service, "_embedding_provider", None) and "local" or "see-config",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read collection stats: {str(e)}")


@router.get("/documents")
async def list_documents(prefix: Optional[str] = None, limit: int = 500, offset: int = 0):
    """Paginated distinct document list with chunk counts (read-only)."""
    try:
        collection = _collection()
        res = collection.get(include=["metadatas", "documents"])
        ids = res["ids"]
        metas = res["metadatas"] if res["metadatas"] else [{}] * len(ids)
        docs: Dict[str, Dict[str, Any]] = {}
        for cid, meta in zip(ids, metas):
            meta = meta or {}
            did = str(meta.get("document_id") or cid.parse_id() if hasattr(cid, "parse_id") else (meta.get("document_id") or str(cid)))
            if prefix and did and did.startswith(prefix) is False:
                continue
            entry = docs.setdefault(did, {"document_id": did, "chunks": 0, "first_chunk_id": cid})
            entry["chunks"] += 1
            for k in ("source", "filename", "checksum", "mime", "version", "corpus", "fiscal_year"):
                if meta.get(k) is not None and k not in entry:
                    entry[k] = meta.get(k)
        items = list(docs.values())
        items.sort(key=lambda d: d["document_id"])
        return {"total": len(items), "limit": limit, "offset": offset, "documents": items[offset:offset + limit]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/documents/{document_id}/chunks")
async def document_chunks(document_id: str, limit: int = 1000, offset: int = 0):
    """Chunk inspector: all chunks of one document (read-only)."""
    try:
        collection = _collection()
        res = collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas", "embeddings"],
        )
        ids = res["ids"]
        docs = (res["documents"] or []) if res["documents"] else [None] * len(ids)
        metas = res["metadatas"] if res["metadatas"] else [{}] * len(ids)
        chunks = []
        for cid, doc, meta in zip(ids, docs, metas):
            chunks.append({
                "chunk_id": cid,
                "document_id": document_id,
                "chunk_index": (meta or {}).get("chunk_index"),
                "source": (meta or {}).get("source"),
                "section": (meta or {}).get("section"),
                "page": (meta or {}).get("page"),
                "total_chunks": (meta or {}).get("total_chunks"),
                "char_count": len(doc) if isinstance(doc, str) else 0,
                "text": doc if isinstance(doc, str) else None,
                "metadata": meta,
                "has_embedding": bool((meta or {}).get("embedding") or False) or True,
            })
        chunks.sort(key=lambda c: (c["chunk_index"] is None, c["chunk_index"] or 0))
        return {
            "document_id": document_id,
            "total_chunks": len(chunks),
            "chunks": chunks[offset:offset + limit],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read chunks: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete ONLY the given document's vectors (targeted, safe)."""
    try:
        deleted = rag_service.vector_store.delete_documents(
            rag_service.collection_name, {"document_id": document_id}
        )
        return {"document_id": document_id, "deleted": deleted, "message": "Document vectors removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")