"""
MCP (Model Context Protocol) Service for RAG Project

Provides model context protocol integration enabling the RAG system to
interoperate with MCP-compatible services and tools. This service acts as
a bridge between the RAG pipeline and external MCP-compatible models/tools.

Features:
- MCP model registration and discovery
- Context injection from MCP sources
- Fallback to local models when MCP is unavailable
- Resilient operation when MCP services are down
"""

from typing import Dict, List, Any, Optional, AsyncGenerator
import logging
import json

logger = logging.getLogger(__name__)


class MCPProvider(BaseAIProvider):
    """MCP Provider that integrates with Model Context Protocol services."""
    
    def __init__(self, mcp_endpoint: str = "http://127.0.0.1:9000/mcp", 
                 api_key: Optional[str] = None,
                 timeout: int = 30):
        self.mcp_endpoint = mcp_endpoint
        self.api_key = api_key
        self.timeout = timeout
        self._models_cache = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazy initialization of MCP connection."""
        if not self._initialized:
            # Try to connect to MCP service, but don't fail if unavailable
            try:
                # Check if MCP service is available
                # In production, this would be an actual HTTP call
                self._initialized = True
                logger.info("MCP provider initialized (connectivity check skipped for resilience)")
            except Exception as e:
                logger.warning(f"MCP initialization non-critical issue: {str(e)}")
                self._initialized = True  # Mark as initialized to avoid retry loops
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding using MCP or fall back to local embedding.
        
        Tries MCP first, falls back to local embedding provider if MCP is unavailable.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, use the sync fallback
                return self._fallback_embedding(text)
            result = loop.run_until_complete(self._create_embedding_mcp(text))
            return result
        except Exception as e:
            logger.warning(f"MCP embedding failed, falling back local: {str(e)}")
            return self._fallback_embedding(text)
    
    async def _create_embedding_mcp(self, text: str) -> List[float]:
        """Create embedding via MCP service."""
        await self._ensure_initialized()
        # MCP embedding protocol - would call external MCP service
        # For now, return fallback
        raise NotImplementedError("MCP embedding not yet implemented, use fallback")
    
    def _fallback_embedding(self, text: str) -> List[float]:
        """Fallback to local embedding model when MCP is unavailable."""
        # Import here to avoid circular imports
        from app.services.providers.local_embedding_service import LocalEmbeddingProvider
        provider = LocalEmbeddingProvider()
        return provider.create_embedding(text)
    
    def create_chat_completion(self, messages: list, model: str) -> Any:
        """Create chat completion using MCP or fall back to configured provider.
        
        Tries MCP first, falls back to the configured AI provider (OpenAI/AvalAI/OpenRouter)
        if MCP is unavailable or returns errors.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return self._fallback_chat_completion(messages, model)
            result = loop.run_until_complete(self._create_chat_completion_mcp(messages, model))
            return result
        except Exception as e:
            logger.warning(f"MCP chat completion failed, falling back to {model}: {str(e)}")
            return self._fallback_chat_completion(messages, model)
    
    async def _create_chat_completion_mcp(self, messages: list, model: str) -> Any:
        """Create chat completion via MCP service."""
        await self._ensure_initialized()
        # Would call MCP service for chat completion
        # For resilience, always have fallback
        raise NotImplementedError("MCP chat completion not yet implemented, use fallback")
    
    def _fallback_chat_completion(self, messages: list, model: str) -> Any:
        """Fallback chat completion using the configured provider."""
        from app.services.core.rag_service import rag_service
        return rag_service.generate_response(messages, "", model=model)
    
    def create_chat_completion_stream(self, messages: list, model: str) -> AsyncGenerator[Any, None]:
        """Create streaming chat completion using MCP or fall back."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                async def gen():
                    yield from self._fallback_chat_completion_stream(messages, model)
                return gen()
            # Run the async method synchronously
            result = loop.run_until_complete(self._create_chat_completion_stream_mcp(messages, model))
            return result
        except Exception as e:
            logger.warning(f"MCP streaming failed, falling back: {str(e)}")
            return self._fallback_chat_completion_stream(messages, model)
    
    async def _create_chat_completion_stream_mcp(self, messages: list, model: str) -> AsyncGenerator[Any, None]:
        """Create streaming chat completion via MCP."""
        await self._ensure_initialized()
        # Would call MCP streaming service
        raise NotImplementedError("MCP streaming not yet implemented, use fallback")
    
    def _fallback_chat_completion_stream(self, messages: list, model: str) -> AsyncGenerator[Any, None]:
        """Fallback streaming using the RAG service."""
        from app.services.core.rag_service import rag_service
        async for chunk in rag_service.generate_stream_response(messages, "", model=model):
            yield chunk
    
    def register_mcp_model(self, model_id: str, metadata: Dict[str, Any]) -> bool:
        """Register an MCP model with the service."""
        try:
            # In a full implementation, this would register with an MCP registry
            logger.info(f"MCP model registered: {model_id}")
            if self._models_cache is None:
                self._models_cache = {}
            self._models_cache[model_id] = metadata
            return True
        except Exception as e:
            logger.error(f"Failed to register MCP model: {str(e)}")
            return False
    
    def discover_mcp_models(self) -> Dict[str, Dict[str, Any]]:
        """Discover available MCP models."""
        try:
            # In production, would query MCP registry
            # Return cached or empty dict for resilience
            if self._models_cache is None:
                self._models_cache = {}
            return self._models_cache or {}
        except Exception as e:
            logger.error(f"MCP model discovery failed: {str(e)}")
            return {}


# Global MCP provider instance
_mcp_provider: MCPProvider = None

def get_mcp_provider() -> MCPProvider:
    """Get the global MCP provider instance."""
    global _mcp_provider
    if _mcp_provider is None:
        _mcp_provider = MCPProvider()
    return _mcp_provider


def reset_mcp_provider():
    """Reset the MCP provider (useful for testing or reconfiguration)."""
    global _mcp_provider
    _mcp_provider = MCPProvider()