from fastapi import APIRouter, HTTPException
from app.models.embedding import Document, Query, SearchResult
from app.models.document import DocumentUpload, ChunkResult, SearchResultWithScore
from app.services.core.rag_service import rag_service
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# this route is used to initialize the collection in the vector database, you need to call it before you can use the other routes
@router.post("/initialize_collection", status_code=201)
async def initialize_collection():
    try:
        rag_service.initialize_collection()
        return {"message": "Collection initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize collection: {str(e)}")


@router.post("/add_document", status_code=201, response_model=dict)
async def add_document(document: Document):
    """
    Add a document to the vector database.
    
    This endpoint now supports chunking for large documents. The document
    will be split into semantic chunks, each embedded separately.
    
    For backward compatibility, if the document text is small (< RAG_CHUNK_SIZE*3),
    it will be treated as a single chunk.
    """
    try:
        result = rag_service.add_document(
            text=document.text,
            metadata=document.metadata,
        )
        
        if result.get("total_chunks", 0) == 0:
            return {"message": "Document was empty or produced no chunks"}
        
        return {
            "message": result["message"],
            "document_id": result["document_id"],
            "total_chunks": result["total_chunks"],
            "chunk_ids": result["chunk_ids"],
        }
    except Exception as e:
        logger.error(f"Failed to add document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to embed and add document: {str(e)}")


@router.post("/add_document_chunked", status_code=201)
async def add_document_chunked(document: DocumentUpload):
    """
    Add a document with intelligent chunking for large document support.
    
    Supports:
    - Large documents (1,000+ lines)
    - Semantic chunking (paragraph/section boundaries)
    - Custom chunk_size and chunk_overlap
    
    Args:
        document: DocumentUpload with text, optional metadata, and optional source
    """
    try:
        result = rag_service.add_document(
            text=document.text,
            metadata=document.metadata,
            source=document.source,
        )
        
        if result.get("total_chunks", 0) == 0:
            return {"message": "Document was empty or produced no chunks", "document_id": result["document_id"]}
        
        return {
            "message": result["message"],
            "document_id": result["document_id"],
            "total_chunks": result["total_chunks"],
            "chunk_ids": result["chunk_ids"],
        }
    except Exception as e:
        logger.error(f"Failed to add chunked document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to embed and add document: {str(e)}")


# this route is used to search for similar documents in the vector database
@router.post("/search_documents", response_model=List[SearchResult])
async def search_documents(query: Query):
    try:
        # Use the configured embedding provider for embeddings
        embedding = rag_service.embedding_provider.create_embedding(query.prompt)
        search_results = rag_service.search_similar_documents(embedding)
        
        # Convert to list of SearchResult objects
        results = []
        n = len(search_results.get("documents", [[]])[0])
        
        for i in range(n):
            results.append(SearchResult(
                id=i,  # Using index as id for backward compatibility
                text=search_results["documents"][0][i],
                metadata=search_results["metadatas"][0][i] if search_results.get("metadatas") else None,
                score=search_results.get("distances", [[]])[0][i] if search_results.get("distances") else None,
            ))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/search_documents_enriched")
async def search_documents_enriched(query: Query, n_results: Optional[int] = None):
    """
    Search for similar documents with full metadata and scores.
    
    Returns results with:
    - Original id from database
    - Text content
    - Full metadata (document_id, chunk_index, etc.)
    - Similarity score
    """
    try:
        from app.config.settings import settings
        k = n_results or settings.RAG_RETRIEVAL_K
        
        results = rag_service.search_with_scores(
            query=query.prompt,
            n_results=k,
        )
        
        return {
            "results": results,
            "count": len(results),
            "query": query.prompt,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enriched search failed: {str(e)}")


# this route is used to clear the collection in the vector database
@router.delete("/clear_collection")
async def clear_collection():
    try:
        rag_service.clear_collection()
        return {"message": "Collection cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")


@router.get("/collection_info")
async def collection_info():
    """Get information about the collection."""
    try:
        from app.config.settings import settings
        count = rag_service.vector_store.get_collection_count(rag_service.collection_name)
        return {
            "collection_name": rag_service.collection_name,
            "document_count": count,
            "chunk_size": settings.RAG_CHUNK_SIZE,
            "chunk_overlap": settings.RAG_CHUNK_OVERLAP,
            "retrieval_k": settings.RAG_RETRIEVAL_K,
            "final_context_k": settings.RAG_FINAL_K,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")


def get_collection_info() -> Dict[str, Any]:
    """Get collection info as dict (helper)."""
    from app.config.settings import settings
    return {
        "collection_name": rag_service.collection_name,
        "document_count": rag_service.vector_store.get_collection_count(rag_service.collection_name),
        "chunk_size": settings.RAG_CHUNK_SIZE,
        "chunk_overlap": settings.RAG_CHUNK_OVERLAP,
        "retrieval_k": settings.RAG_RETRIEVAL_K,
        "final_context_k": settings.RAG_FINAL_K,
    }