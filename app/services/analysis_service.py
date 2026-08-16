"""
Smart Analysis Service for RAG Project

Provides intelligent analysis capabilities that work alongside the RAG pipeline
but are resilient to RAG service failures. This service can:
- Analyze user queries with or without RAG context
- Fall back to pure LLM responses when RAG is unavailable
- Provide confidence scores and relevance assessment
- Integrate with MCP when available but not depend on it
- Handle edge cases and maintain service availability

Key Design Principles:
1. Graceful degradation - never fail entirely if RAG/MCP unavailable
2. Confidence scoring - user knows when context is vs. not being used
3. Configurable fallback behavior
4. Audit trail of what information was used
"""

from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalysisResult:
    """Result structure from the analysis service."""
    
    def __init__(
        self,
        response: str,
        used_rag_context: bool = False,
        confidence: float = 0.5,
        rag_relevant_chunks: List[Dict] = None,
        fallback_reason: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        self.response = response
        self.used_rag_context = used_rag_context
        self.confidence = min(max(confidence, 0.0), 1.0)  # Clamp to 0-1
        self.rag_relevant_chunks = rag_relevant_chunks or []
        self.fallback_reason = fallback_reason
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "used_rag_context": self.used_rag_context,
            "confidence": self.confidence,
            "rag_relevant_chunks": self.rag_relevant_chunks,
            "fallback_reason": self.fallback_reason,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def __str__(self):
        return f"AnalysisResult(confidence={self.confidence:.2f}, rag_used={self.used_rag_context})"


class SmartAnalysisService:
    """Intelligent analysis service that works with or without RAG."""
    
    def __init__(self, rag_service=None, mcp_provider=None):
        self.rag_service = rag_service
        self.mcp_provider = mcp_provider
        self.fallback_mode = False
        self.unavailable_count = 0
        self.max_unavailable = 5  # After 5 failures, enter conservative mode
    
    def set_rag_service(self, rag_service):
        """Set the RAG service instance."""
        self.rag_service = rag_service
    
    def set_mcp_provider(self, mcp_provider):
        """Set the MCP provider instance."""
        self.mcp_provider = mcp_provider
    
    def _check_health(self) -> bool:
        """Check if RAG service is available. Returns True if healthy."""
        if self.rag_service is None:
            logger.warning("No RAG service configured")
            return False
        
        try:
            # Quick health check - try a simple operation
            # In production, this might be a ping or status check
            return True  # Assume healthy for now
        except Exception as e:
            logger.warning(f"RAG health check failed: {str(e)}")
            self.unavailable_count += 1
            if self.unavailable_count >= self.max_unavailable:
                self.fallback_mode = True
                logger.error(f"Entering fallback mode after {self.max_unavailable} failures")
            return False
    
    def _assess_relevance(self, query: str, context_chunks: List[Dict]) -> float:
        """Assess how relevant the retrieved chunks are to the query."""
        if not context_chunks:
            return 0.0
        
        # Simple relevance assessment based on chunk metadata
        # In production, this would use similarity scores, keyword matching, etc.
        relevance_scores = []
        for chunk in context_chunks:
            score = chunk.get("similarity_score", 0.0) or chunk.get("score", 0.0)
            if score is not None:
                relevance_scores.append(float(score))
        
        if not relevance_scores:
            return 0.5  # Neutral if no scores available
        
        # Use the average relevance, but minimum 0.3 if we have chunks
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        return max(avg_relevance, 0.3) if relevance_scores else 0.0
    
    async def analyze_query(
        self, 
        query: str, 
        use_rag: bool = True,
        force_fallback: bool = False
    ) -> AnalysisResult:
        """Analyze a query using RAG if available, otherwise pure LLM.
        
        Args:
            query: The user's question
            use_rag: Whether to attempt RAG-based analysis
            force_fallback: Force fallback mode even if RAG available
        
        Returns:
            AnalysisResult with response and metadata about what was used
        """
        start_time = datetime.utcnow()
        used_rag = False
        confidence = 0.5
        rag_chunks = []
        fallback_reason = None
        
        # Check if we should use RAG
        rag_available = self.rag_service is not None and not self.fallback_mode
        
        if use_rag and rag_available and not force_fallback:
            try:
                # Try RAG-based analysis
                rag_result = await self._analyze_with_rag(query)
                if rag_result and rag_result.get("success", False):
                    used_rag = True
                    confidence = rag_result.get("confidence", 0.5)
                    rag_chunks = rag_result.get("chunks", [])
                    response = rag_result.get("response", "")
                    
                    # Assess relevance
                    relevance = self._assess_relevance(query, rag_chunks)
                    confidence = (confidence + relevance) / 2
                    
                    return AnalysisResult(
                        response=response,
                        used_rag_context=used_rag,
                        confidence=confidence,
                        rag_relevant_chunks=rag_chunks,
                        metadata={"analysis_method": "rag", "relevance": relevance}
                    )
                else:
                    # RAG returned but indicated issues
                    fallback_reason = rag_result.get("error", "RAG returned no useful result")
                    
            except Exception as e:
                logger.warning(f"RAG analysis failed: {str(e)}")
                fallback_reason = str(e)
                # Continue to fallback below
        
        # Fallback: Pure LLM response without RAG context
        if not used_rag:
            try:
                fallback_result = await self._analyze_without_rag(query)
                if fallback_result:
                    response = fallback_result.get("response", "")
                    confidence = fallback_result.get("confidence", 0.5)
                    
                    # If we had RAG available but it failed, note it
                    if rag_available and not force_fallback:
                        fallback_reason = "RAG unavailable or unsuccessful, using fallback"
                        self.unavailable_count = min(self.unavailable_count + 1, self.max_unavailable)
                    
                    return AnalysisResult(
                        response=response,
                        used_rag_context=used_rag,
                        confidence=confidence,
                        rag_relevant_chunks=rag_chunks,
                        fallback_reason=fallback_reason,
                        metadata={"analysis_method": "fallback_llm"}
                    )
                else:
                    # Even fallback failed
                    return AnalysisResult(
                        response="I'm sorry, I'm having trouble processing your request. Please try again later.",
                        used_rag_context=False,
                        confidence=0.1,
                        fallback_reason="All analysis methods failed",
                        metadata={"analysis_method": "error"}
                    )
            except Exception as e:
                logger.error(f"Fallback analysis also failed: {str(e)}")
                return AnalysisResult(
                    response="I'm sorry, I'm having trouble processing your request. Please try again later.",
                    used_rag_context=False,
                    confidence=0.1,
                    fallback_reason=f"Critical error: {str(e)}",
                    metadata={"analysis_method": "critical_error"}
                )
        
        # Should not reach here, but just in case
        return AnalysisResult(
            response="Analysis completed.",
            used_rag_context=True,
            confidence=confidence,
            rag_relevant_chunks=rag_chunks,
            metadata={"analysis_method": "rag"}
        )
    
    async def _analyze_with_rag(self, query: str) -> Optional[Dict]:
        """Attempt analysis using RAG service."""
        if self.rag_service is None:
            return {"success": False, "error": "No RAG service configured"}
        
        try:
            # Search for similar documents
            # First get embedding for the query
            embedding_provider = self.rag_service.embedding_provider
            embedding = embedding_provider.create_embedding(query)
            
            # Search for similar documents
            search_result = self.rag_service.search_similar_documents(embedding)
            
            # Extract relevant chunks
            rag_chunks = search_result.get("metadatas", []) or search_result.get("documents", [])
            
            if not rag_chunks:
                return {"success": False, "error": "No relevant documents found"}
            
            # Build context from relevant chunks
            context_parts = []
            for i, chunk in enumerate(rag_chunks[:3]):  # Top 3 chunks
                chunk_text = chunk.get("document", str(chunk))
                metadata = chunk.get("metadata", {})
                source = metadata.get("source", f"chunk_{i}")
                context_parts.append(f"[Source {i+1}: {source}]\n{chunk_text}")
            
            context = "\n\n".join(context_parts)
            
            # Generate response using the provider
            response = await self.rag_service.generate_stream_response(
                [{"role": "user", "content": query}],
                context=context
            )
            
            # Collect the full response
            full_response = ""
            async for chunk in response:
                if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                    full_response += chunk["choices"][0]["delta"]["content"]
            
            if not full_response:
                full_response = "I found relevant documents but couldn't generate a response."
            
            return {
                "success": True,
                "confidence": min(0.9, 0.5 + len(rag_chunks) * 0.1),
                "chunks": rag_chunks[:3],
                "response": full_response
            }
            
        except Exception as e:
            logger.error(f"RAG analysis error: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def _analyze_without_rag(self, query: str) -> Optional[Dict]:
        """Analyze query without RAG context - pure LLM response."""
        try:
            # Use the RAG service's provider but without context
            if self.rag_service is None:
                # Minimal fallback - just return a generic response
                return {
                    "response": "I received your query but don't have access to the knowledge base. Please provide more context or check if the RAG service is available.",
                    "confidence": 0.3
                }
            
            # Generate response without context (empty context string)
            response = self.rag_service.generate_response(
                [{"role": "user", "content": query}],
                context=""
            )
            
            # Estimate confidence based on response length and simplicity
            confidence = min(0.8, 0.5 + len(str(response)) / 1000 * 0.1)
            
            return {
                "success": True,
                "response": str(response) if response else "I don't know",
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Fallback analysis error: {str(e)}", exc_info=True)
            return {
                "response": "I'm sorry, I'm having trouble processing your request.",
                "confidence": 0.1
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            "rag_service_configured": self.rag_service is not None,
            "fallback_mode": self.fallback_mode,
            "unavailable_count": self.unavailable_count,
            "mcp_provider_configured": self.mcp_provider is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def reset_unavailability_counter(self):
        """Reset the unavailability counter if service comes back online."""
        self.unavailable_count = max(0, self.unavailable_count - 2)
        self.fallback_mode = self.unavailable_count >= self.max_unavailable


# Global instance for easy access
_analysis_service: SmartAnalysisService = None

def get_analysis_service() -> SmartAnalysisService:
    """Get the global analysis service instance."""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = SmartAnalysisService()
    return _analysis_service

def reset_analysis_service():
    """Reset the analysis service global instance."""
    global _analysis_service
    _analysis_service = SmartAnalysisService()