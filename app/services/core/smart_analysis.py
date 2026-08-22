"""Smart Analysis Service.

This service provides intelligent analysis with RAG fallback.
It inspects ChromaDB search results and properly handles the result structure.
"""

import logging
from typing import Dict, List, Any, Optional, Literal
from app.config.settings import settings
from app.services.core.rag_service import RAGService

logger = logging.getLogger(__name__)


class SmartAnalysisService:
    """Service for intelligent analysis with RAG capabilities."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.collection_name = rag_service.collection_name
    
    def analyze_with_rag(
        self,
        query: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        retrieval_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a query using RAG (Retrieval-Augmented Generation).
        
        This method properly handles the ChromaDB result structure and ensures
        RAG is actually used when available.
        
        Returns:
            Dict with:
            - 'response': the generated text
            - 'retrieval_used': boolean indicating RAG was used
            - 'retrieved_chunks': list of chunks used
            - 'context_length': character length of context
        """
        retrieval_k = retrieval_k or settings.RAG_RETRIEVAL_K
        
        try:
            # Step 1: Create embedding for the query
            embedding = self.rag_service.embedding_provider.create_embedding(query)
            
            # Step 2: Search for similar documents with proper error handling
            search_results = self.rag_service.search_similar_documents(embedding, n_results=retrieval_k)
            
            # Step 3: Extract documents and metadata from results
            # ChromaDB returns results in a dict with keys:
            # - 'documents': List[List[str]]
            # - 'metadatas': List[List[Dict]]
            # - 'distances': List[List[float]]
            # - 'ids': List[List[str]]
            
            documents = search_results.get("documents", [[]])[0]
            metadatas = search_results.get("metadatas", [[]])[0]
            distances = search_results.get("distances", [[]])[0] if search_results.get("distances") else []
            ids = search_results.get("ids", [[]])[0]
            
            if not documents or len(documents) == 0:
                logger.warning("No documents retrieved from ChromaDB")
                return self._fallback_to_llm(query, messages)
            
            # Step 4: Build context from retrieved documents
            context_parts = []
            retrieved_chunks = []
            
            final_k = min(len(documents), settings.RAG_FINAL_K)
            
            for i in range(final_k):
                doc_text = documents[i] if i < len(documents) else ""
                meta = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else None
                doc_id = ids[i] if i < len(ids) else f"result_{i}"
                
                # Build clean metadata header
                header_parts = []
                if meta:
                    if meta.get("source"):
                        header_parts.append(f"Source: {meta['source']}")
                    if meta.get("section"):
                        header_parts.append(f"Section: {meta['section']}")
                    if meta.get("document_id"):
                        header_parts.append(f"Document: {meta['document_id']}")
                    if meta.get("chunk_index") is not None:
                        total = meta.get("total_chunks", "?")
                        header_parts.append(f"Chunk: {meta['chunk_index']}/{total}")
                
                header = " | ".join(header_parts) if header_parts else f"Result {i+1}"
                
                context_parts.append(f"[{header}]\n{doc_text}")
                retrieved_chunks.append({
                    "id": doc_id,
                    "text": doc_text,
                    "metadata": meta,
                    "distance": distance,
                })
            
            context = "\n\n".join(context_parts)
            
            # Step 5: Verify we actually have useful context
            if not context.strip():
                logger.warning("Retrieved context is empty (whitespace only), falling back to LLM")
                return self._fallback_to_llm(query, messages)
            
            # Step 6: Build messages for LLM
            if messages:
                messages_with_context = messages.copy()
            else:
                messages_with_context = [{"role": "user", "content": query}]
            
            system_prompt = settings.SYSTEM_PROMPT
            system_msg = None
            
            # Find or create system message
            for msg in messages_with_context:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    system_msg = msg
                    break
                if hasattr(msg, 'role') and msg.role == "system":
                    system_msg = msg
                    break
            
            context_string = f"Use this context to answer the user's question:\n\n{context}"
            
            if system_msg:
                if hasattr(system_msg, 'content'):
                    system_msg.content = f"{system_msg.content}\n\n{context_string}"
                else:
                    system_msg["content"] = f"{system_msg['content']}\n\n{context_string}"
            else:
                messages_with_context.insert(0, {
                    "role": "system",
                    "content": context_string,
                })
            
            # Convert messages to dict format
            messages_dict = []
            for msg in messages_with_context:
                if isinstance(msg, dict):
                    messages_dict.append(msg)
                elif hasattr(msg, 'model_dump'):
                    messages_dict.append(msg.model_dump())
                else:
                    messages_dict.append({"role": msg.role, "content": msg.content})
            
            # Step 7: Generate LLM response
            # messages_with_context already has the system prompt with context
            # Pass empty context to generate_response to avoid double-injecting
            response = self.rag_service.generate_response(
                messages=messages_with_context,
                context="",
            )
            
            # Step 8: Extract response text (handle both dict and object responses)
            response_text = ""
            if isinstance(response, dict):
                response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
            elif hasattr(response, 'message'):
                response_text = response.message.content or ""
            
            result = {
                "response": response_text,
                "retrieval_used": True,
                "retrieved_chunks": retrieved_chunks,
                "context_length": len(context),
                "retrieval_count": len(retrieved_chunks),
            }
            
            logger.info(f"RAG analysis completed: retrieval_used=True, chunks={len(retrieved_chunks)}")
            return result
            
        except Exception as e:
            logger.error(f"RAG analysis failed: {str(e)}")
            return self._fallback_to_llm(query, messages)
    
    def _fallback_to_llm(
        self,
        query: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Fallback to LLM without RAG context."""
        logger.info("Falling back to LLM-only response")
        
        try:
            if messages:
                messages_with_context = messages.copy()
            else:
                messages_with_context = [
                    {"role": "system", "content": settings.SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ]
            
            # Ensure we have a system message
            has_system = any(
                (isinstance(m, dict) and m.get("role") == "system") or
                (hasattr(m, 'role') and m.role == "system")
                for m in messages_with_context
            )
            
            if not has_system and not messages:
                messages_with_context.insert(0, {"role": "system", "content": settings.SYSTEM_PROMPT})
            
            messages_dict = []
            for msg in messages_with_context:
                if isinstance(msg, dict):
                    messages_dict.append(msg)
                elif hasattr(msg, 'model_dump'):
                    messages_dict.append(msg.model_dump())
                else:
                    messages_dict.append({"role": msg.role, "content": msg.content})
            
            response = self.rag_service.generate_response(
                messages=messages_dict,
                context="",
            )
            
            response_text = ""
            if isinstance(response, dict):
                response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
            elif hasattr(response, 'message'):
                response_text = response.message.content or ""
            
            return {
                "response": response_text,
                "retrieval_used": False,
                "retrieved_chunks": [],
                "context_length": 0,
                "retrieval_count": 0,
            }
            
        except Exception as e:
            logger.error(f"LLM fallback failed: {str(e)}")
            return {
                "response": "I'm unable to process your query at the moment. Please try again later.",
                "retrieval_used": False,
                "retrieved_chunks": [],
                "context_length": 0,
                "retrieval_count": 0,
                "error": str(e),
            }
    
    async def analyze_with_rag_stream(
        self,
        query: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """
        Stream an RAG-enabled analysis response.
        
        Properly handles the streaming response from the provider.
        """
        try:
            # Get embedding and search
            embedding = self.rag_service.embedding_provider.create_embedding(query)
            search_results = self.rag_service.search_similar_documents(embedding)
            
            documents = search_results.get("documents", [[]])[0]
            metadatas = search_results.get("metadatas", [[]])[0]
            
            if not documents or len(documents) == 0:
                # Fallback to LLM streaming
                yield json.dumps({
                    "choices": [{
                        "delta": {"content": "No relevant context found in the knowledge base."}
                    }]
                })
                return
            
            # Build context
            context_parts = []
            final_k = min(len(documents), settings.RAG_FINAL_K)
            
            for i in range(final_k):
                doc_text = documents[i]
                meta = metadatas[i] if i < len(metadatas) else {}
                
                header_parts = []
                if meta:
                    if meta.get("source"):
                        header_parts.append(f"Source: {meta['source']}")
                    if meta.get("document_id"):
                        header_parts.append(f"Document: {meta['document_id']}")
                    if meta.get("chunk_index") is not None:
                        total = meta.get("total_chunks", "?")
                        header_parts.append(f"Chunk: {meta['chunk_index']}/{total}")
                
                header = " | ".join(header_parts) if header_parts else f"Result {i+1}"
                context_parts.append(f"[{header}]\n{doc_text}")
            
            context = "\n\n".join(context_parts)
            
            # Stream response
            messages_dict = [
                {"role": "system", "content": f"{settings.SYSTEM_PROMPT}\n\nUse context:\n{context}"},
                {"role": "user", "content": query},
            ]
            
            async for chunk in self.rag_service.generate_stream_response(
                messages=messages_dict,
                context=context,
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming RAG analysis failed: {str(e)}")
            yield json.dumps({
                "choices": [{
                    "delta": {"content": f"Error: {str(e)}"}
                }]
            })


# Module-level singleton
from app.services.core.rag_service import rag_service
smart_analysis_service = SmartAnalysisService(rag_service)
