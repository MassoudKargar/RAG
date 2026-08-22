"""Analysis routes for SmartAnalysisService.

Provides /v1/analysis/query and /v1/analysis/status endpoints.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.chat import ChatCompletionRequest
from app.services.core.smart_analysis import smart_analysis_service
from app.config.settings import settings
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analysis/query")
async def analysis_query(request: ChatCompletionRequest):
    """
    Analyze a query using RAG when retrieval is available.
    
    This endpoint:
    1. Embeds the user's query
    2. Searches ChromaDB for relevant context
    3. Builds context from retrieved chunks with proper metadata
    4. Sends to LLM for response generation
    
    If no relevant context is found, it falls back to LLM-only response.
    """
    try:
        # Get the last user message
        last_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
        if not last_message:
            raise HTTPException(status_code=400, detail="No user message found in the conversation")
        
        # Rewrite query for conversational context
        retrieval_query = smart_analysis_service.rag_service.rewrite_query(request.messages)
        
        # Convert messages to dict format for the service
        messages_dict = []
        for msg in request.messages:
            if hasattr(msg, 'model_dump'):
                messages_dict.append(msg.model_dump())
            elif isinstance(msg, dict):
                messages_dict.append(msg)
            else:
                messages_dict.append({"role": msg.role, "content": msg.content})
        
        result = smart_analysis_service.analyze_with_rag(
            query=retrieval_query,
            messages=messages_dict,
            retrieval_k=settings.RAG_RETRIEVAL_K,
        )
        
        return {
            "response": result["response"],
            "retrieval_used": result["retrieval_used"],
            "retrieval_count": result["retrieval_count"],
            "context_length": result["context_length"],
            "retrieved_chunks": result["retrieved_chunks"],
        }
        
    except Exception as e:
        logger.error(f"Analysis query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analysis/stream")
async def analysis_query_stream(request: ChatCompletionRequest):
    """Stream an RAG-enabled analysis response."""
    try:
        last_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
        if not last_message:
            raise HTTPException(status_code=400, detail="No user message found in the conversation")
        
        retrieval_query = smart_analysis_service.rag_service.rewrite_query(request.messages)
        
        messages_dict = []
        for msg in request.messages:
            if hasattr(msg, 'model_dump'):
                messages_dict.append(msg.model_dump())
            elif isinstance(msg, dict):
                messages_dict.append(msg)
            else:
                messages_dict.append({"role": msg.role, "content": msg.content})
        
        async def event_stream():
            async for chunk in smart_analysis_service.analyze_with_rag_stream(
                query=retrieval_query,
                messages=messages_dict,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Analysis stream failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis stream failed: {str(e)}")


@router.get("/analysis/status")
async def analysis_status():
    """Get the status of the analysis system."""
    try:
        collection_count = smart_analysis_service.rag_service.vector_store.get_collection_count(
            smart_analysis_service.rag_service.collection_name
        )
        
        return {
            "status": "operational",
            "rag_enabled": True,
            "vector_db_available": True,
            "documents_in_collection": collection_count,
            "embedding_provider": settings.effective_embedding_provider,
            "chat_provider": settings.PROVIDER,
            "retrieval_k": settings.RAG_RETRIEVAL_K,
            "final_context_k": settings.RAG_FINAL_K,
            "chunk_size": settings.RAG_CHUNK_SIZE,
            "chunk_overlap": settings.RAG_CHUNK_OVERLAP,
            "system_prompt_set": bool(settings.SYSTEM_PROMPT),
        }
    except Exception as e:
        logger.error(f"Failed to get analysis status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
