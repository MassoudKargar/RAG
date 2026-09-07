from fastapi import APIRouter, HTTPException
from app.models.embedding import Document, Query, SearchResult
from app.models.document import DocumentUpload, ChunkResult
from app.services.core.rag_service import rag_service
from typing import List, Optional

router = APIRouter()


# this route is used to initialize the collection in the vector database, you need to call it before you can use the other routes
@router.post("/initialize_collection", status_code=201)
async def initialize_collection():
    try:
        rag_service.initialize_collection()
        return {"message": "Collection initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize collection: {str(e)}")


# this route is used to add a document to the vector database
@router.post("/add_document", status_code=201, response_model=dict)
async def add_document(document: Document):
    try:
        result = rag_service.add_document(document.text, document.metadata)
        return {"message": result["message"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed and add document: {str(e)}")


# chunk-aware document upload (replaces the same document_id if provided)
@router.post("/documents", status_code=201, response_model=ChunkResult)
async def add_document_chunked(upload: DocumentUpload):
    """Upload a document with chunking support.

    The document is chunked at semantic boundaries, each chunk is embedded via
    the configured embedding provider, and all chunks are stored in ChromaDB.
    Re-uploading with the same ``source``/metadata ``document_id`` replaces the
    previous chunks (no duplicates).
    """
    try:
        metadata = dict(upload.metadata or {})
        if upload.source and "source" not in metadata:
            metadata["source"] = upload.source
        document_id = metadata.get("document_id")

        result = rag_service.add_document(
            upload.text,
            metadata=metadata or None,
            document_id=document_id,
            source=upload.source,
        )
        return ChunkResult(
            document_id=result["document_id"],
            chunks_added=result["chunks_added"],
            chunk_ids=result["chunk_ids"],
            message=result["message"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to chunk and store document: {str(e)}")


# this route is used to search for similar documents in the vector database
@router.post("/search_documents", response_model=List[SearchResult])
async def search_documents(query: Query, limit: Optional[int] = None):
    try:
        # Use the configured embedding provider for embeddings
        embedding = rag_service.embedding_provider.create_embedding(query.prompt)
        results = rag_service.search_similar_documents(embedding, limit=limit, query=query.prompt)
        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append(SearchResult(
                id=i,  # Using index as id instead of the UUID
                text=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                score=results['distances'][0][i] if results.get('distances') else None
            ))
        return search_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# this route is used to clear the collection in the vector database
@router.delete("/clear_collection")
async def clear_collection():
    try:
        rag_service.clear_collection()
        return {"message": "Collection cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")