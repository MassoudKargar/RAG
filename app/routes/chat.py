from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.chat import ChatCompletionRequest
from app.services.core.rag_service import rag_service
from app.config.settings import settings
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Get the last user message for context search
        last_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
        if not last_message:
            raise HTTPException(status_code=400, detail="No user message found in the conversation")
        
        # Rewrite query for conversational context
        original_query = last_message.content
        retrieval_query = rag_service.rewrite_query(request.messages)
        
        logger.info(f"Retrieval query (original: '{original_query}' -> rewritten: '{retrieval_query}')")
        
        # Get embedding for the query
        embedding = rag_service.embedding_provider.create_embedding(retrieval_query)
        
        # Search for relevant documents
        search_results = rag_service.search_similar_documents(embedding)
        
        # Build context from search results (retrieve more, send fewer)
        context, used_results = rag_service.build_context(search_results)
        
        retrieved_context_used = len(used_results) > 0
        logger.info(f"Retrieved {len(used_results)} relevant chunks for query")
        
        if request.stream:
            async def stream_response():
                async for chunk in rag_service.generate_stream_response(
                    messages=request.messages,
                    context=context,
                    model=request.model
                ):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream"
            )
        # Otherwise return the complete response
        response = rag_service.generate_response(
            messages=request.messages,
            context=context,
            model=request.model
        )
        return response
    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")