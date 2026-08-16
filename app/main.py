from fastapi import FastAPI, Depends, Header, HTTPException
from app.routes import vector_db, chat
from app.services.analysis_service import get_analysis_service, SmartAnalysisService
from app.services.providers.mcp_service import get_mcp_provider, MCPProvider
import logging
from app.config.settings import settings
from app.services.core.rag_service import rag_service


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Fail-closed API key check: every request must carry a valid X-API-Key header."""
    if not settings.RAG_API_KEY or x_api_key != settings.RAG_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide it via the 'X-API-Key' header.",
        )


# Initialize analysis service with fallback capabilities
analysis_service = SmartAnalysisService(rag_service=rag_service)

# Try to integrate MCP provider if available
try:
    mcp_provider = get_mcp_provider()
    analysis_service.set_mcp_provider(mcp_provider)
    logger.info("MCP provider integrated with analysis service")
except Exception as e:
    logger.warning(f"MCP provider integration skipped (non-critical): {str(e)}")

app = FastAPI(
    title="RAG API with Smart Analysis",
    description="API for RAG operations with intelligent analysis and resilient fallback",
    dependencies=[Depends(require_api_key)],
)
logging.basicConfig(level=logging.INFO)


@app.on_event("startup")
async def startup_event():
    try:
        # Check if collection exists first
        collection = rag_service.vector_store.get_or_create_collection(rag_service.collection_name)
        if not collection:
            rag_service.initialize_collection()
            logging.info("Vector database collection initialized successfully on startup")
        else:
            logging.info("Vector database collection already exists")
        
        # Check analysis service health
        status = analysis_service.get_status()
        logger.info(f"Analysis service status: {status}")
    except Exception as e:
        logging.error(f"Failed to initialize: {str(e)}")


# Include routers with tags
app.include_router(vector_db.router, prefix="/v1/vector_db", tags=["Vector Database"])
app.include_router(chat.router, prefix="/v1", tags=["Chat"])


@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to the Maux RAG API"}


# Smart analysis endpoint
@app.get("/v1/analysis/status", tags=["Analysis"])
async def analysis_status():
    """Get analysis service status including RAG and MCP availability."""
    status = analysis_service.get_status()
    return {"status": status}


@app.post("/v1/analysis/query", tags=["Analysis"])
async def analyze_query(
    query: str,
    use_rag: bool = True
):
    """Analyze a user query with intelligent fallback.
    
    If RAG is available and successful, uses context from the knowledge base.
    If RAG is unavailable, gracefully falls back to pure LLM response.
    Always returns a result, never fails entirely.
    """
    try:
        result = await analysis_service.analyze_query(query, use_rag=use_rag)
        return {
            "success": True,
            "response": result.response,
            "used_rag_context": result.used_rag_context,
            "confidence": result.confidence,
            "fallback_reason": result.fallback_reason,
            "metadata": result.metadata
        }
    except Exception as e:
        logger.error(f"Analysis endpoint error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "response": "I'm sorry, I'm having trouble processing your request. Please try again later.",
            "used_rag_context": False,
            "confidence": 0.1,
            "fallback_reason": str(e),
            "metadata": {"analysis_method": "error_handling"}
        }


# Health check endpoint
@app.get("/v1/health", tags=["Health"])
async def health_check():
    """Comprehensive health check for all services."""
    rag_healthy = False
    try:
        # Quick RAG health check
        if rag_service and rag_service.vector_store:
            rag_healthy = True
    except Exception:
        pass
    
    analysis_status = analysis_service.get_status()
    
    return {
        "status": "healthy" if rag_healthy else "degraded",
        "rag_available": rag_healthy,
        "analysis_service": analysis_status,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)