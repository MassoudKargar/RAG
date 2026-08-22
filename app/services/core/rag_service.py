"""RAG Service - Orchestration layer for document ingestion and retrieval.

Responsibilities:
- Document ingestion (with chunking for large documents)
- Batch embedding for efficiency
- Context building from retrieved chunks
- Query rewriting for conversational queries
- Fallback to LLM when no relevant context is found

Architecture:
- ChunkingService handles document → chunks
- EmbeddingProvider handles text → vector
- VectorStoreService handles vectors + metadata → ChromaDB
- RAGService orchestrates the flow
"""

from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
import uuid
import json
import logging
import time
from app.config.settings import settings
from app.services.core.vector_store import VectorStoreService
from app.services.core.chunker import ChunkerService, Chunk
from app.services.providers.openai_service import OpenAIProvider
from app.services.providers.avalai_service import AvalaiProvider
from app.services.providers.openrouter_service import OpenRouterProvider
from app.services.providers.local_embedding_service import LocalEmbeddingProvider
from app.services.base import BaseAIProvider

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.collection_name = "RAG_COLLECTION"
        self.vector_store = VectorStoreService()
        self.chunker = ChunkerService(
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        )
        self._provider = None
        self._embedding_provider = None
        
    @property
    def provider(self) -> BaseAIProvider:
        """Lazy load the chat provider based on settings"""
        if self._provider is None:
            if settings.PROVIDER == "openai":
                self._provider = OpenAIProvider()
            elif settings.PROVIDER == "avalai":
                self._provider = AvalaiProvider()
            elif settings.PROVIDER == "openrouter":
                self._provider = OpenRouterProvider()
            else:
                raise ValueError("Invalid provider selected")
        return self._provider

    @property
    def embedding_provider(self):
        """Lazy load the embedding provider.

        When EMBEDDING_PROVIDER=local, a local HuggingFace model is used for
        embeddings (decoupled from the chat provider, e.g. OpenRouter for chat).
        Otherwise it falls back to the chat provider so existing behaviour is
        preserved when EMBEDDING_PROVIDER is unset.
        """
        if self._embedding_provider is None:
            eff = settings.effective_embedding_provider
            if eff == "local":
                self._embedding_provider = LocalEmbeddingProvider()
            else:
                # Reuse the chat provider for embeddings (backward compatible)
                self._embedding_provider = self.provider
        return self._embedding_provider

    def initialize_collection(self) -> None:
        """Initialize the vector store collection"""
        self.vector_store.create_collection(self.collection_name)

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        source: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Add a document to the vector store with intelligent chunking.
        
        For large documents, this:
        1. Chunks the text into semantically meaningful pieces
        2. Generates deterministic IDs for each chunk
        3. Embeds chunks in batches for efficiency
        4. Stores all chunks with rich metadata
        
        Args:
            text: The full document text
            metadata: Optional metadata dict (merged into chunk metadata)
            document_id: Optional unique document ID (auto-generated if not provided)
            source: Optional source identifier (e.g., filename)
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            
        Returns:
            Dict with document_id, total_chunks, chunk_ids
        """
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        if source is None:
            source = metadata.get("source", document_id) if metadata else document_id
        
        # Use provided override or default chunker
        if chunk_size is not None or chunk_overlap is not None:
            chunker = ChunkerService(
                chunk_size=chunk_size or settings.RAG_CHUNK_SIZE,
                chunk_overlap=chunk_overlap or settings.RAG_CHUNK_OVERLAP,
            )
        else:
            chunker = self.chunker
        
        logger.info(f"Document ID: {document_id}")
        logger.info(f"Starting chunk processing for document {document_id}")
        
        # Chunk the document
        chunks = chunker.chunk_document(
            text=text,
            document_id=document_id,
            source=source,
            metadata=metadata,
        )
        
        total_chunks = len(chunks)
        logger.info(f"Total chunks for document {document_id}: {total_chunks}")
        
        if total_chunks == 0:
            logger.warning(f"No chunks generated for document {document_id}")
            return {
                "document_id": document_id,
                "total_chunks": 0,
                "chunk_ids": [],
                "message": "Document produced no chunks",
            }
        
        # Generate deterministic chunk IDs: document_id + chunk_index
        chunk_ids = [f"{document_id}_{i}" for i in range(total_chunks)]
        
        # Convert chunks to format for vector store
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunks_data.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
                "id": chunk_ids[i],
            })
        
        # Store chunks in batches for memory efficiency
        batch_size = min(32, total_chunks)
        logger.info(f"Batch size: {batch_size}")
        
        embedded_count = 0
        for batch_start in range(0, total_chunks, batch_size):
            batch = chunks_data[batch_start:batch_start + batch_size]
            batch_texts = [c["text"] for c in batch]
            
            # The vector store's embedding function handles embedding
            # ChromaDB will call the embedding function automatically
            start_time = time.time()
            self.vector_store.add_chunks_with_metadata(
                collection_name=self.collection_name,
                chunks=batch,
                batch_size=len(batch),
            )
            elapsed = time.time() - start_time
            
            embedded_count += len(batch)
            logger.info(f"Embedding progress: {embedded_count}/{total_chunks} ({elapsed:.2f}s)")
        
        logger.info(f"Document {document_id} stored successfully with {total_chunks} chunks")
        
        return {
            "document_id": document_id,
            "total_chunks": total_chunks,
            "chunk_ids": chunk_ids,
            "message": f"Document embedded and added successfully ({total_chunks} chunks)",
        }
    
    def add_document_simple(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a document WITHOUT chunking (backward compatible method).
        
        This is the original behavior - entire document as one embedding.
        Kept for backward compatibility with existing clients.
        """
        self.vector_store.add_documents(
            collection_name=self.collection_name,
            documents=[text],
            ids=[f"doc_{uuid.uuid4()}"],
            metadatas=[metadata] if metadata else None
        )

    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        self.vector_store.clear_collection(self.collection_name)

    def search_similar_documents(
        self,
        embedding: List[float],
        n_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search for similar documents using the provided embedding"""
        return self.vector_store.search(
            collection_name=self.collection_name,
            query_embeddings=embedding,
            n_results=n_results,
        )
    
    def search_with_scores(
        self,
        query: str,
        n_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents by query text.
        
        Returns enriched results with scores and metadata.
        """
        # Get embedding for the query
        embedding = self.embedding_provider.create_embedding(query)
        
        # Search with scores
        return self.vector_store.search_with_scores(
            collection_name=self.collection_name,
            query_embeddings=embedding,
            n_results=n_results,
        )
    
    def build_context(
        self,
        search_results: Dict[str, Any],
        final_k: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Build context string from search results.
        
        This method:
        1. Takes the top retrieval_k results
        2. Ranks/filters to final_k results
        3. Builds a clean context string with metadata
        
        Args:
            search_results: Results from ChromaDB search
            final_k: Number of results to include in final context
            
        Returns:
            Tuple of (context_string, list_of_used_results)
        """
        final_k = final_k or settings.RAG_FINAL_K
        
        documents = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        distances = search_results.get("distances", [[]])[0] if search_results.get("distances") else [None] * len(documents)
        ids = search_results.get("ids", [[]])[0]
        
        # Filter to final_k results
        n = min(len(documents), final_k)
        
        context_parts = []
        used_results = []
        
        for i in range(n):
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
            used_results.append({
                "id": doc_id,
                "text": doc_text,
                "metadata": meta,
                "distance": distance,
            })
        
        context = "\n\n".join(context_parts)
        return context, used_results
    
    def rewrite_query(self, messages: List, max_history: int = 2) -> str:
        """
        Rewrite the last user query to be self-contained using recent conversation history.
        
        This addresses the issue where follow-up questions like "برای سال قبل چطور؟"
        don't contain enough context for semantic search.
        
        Args:
            messages: Chat messages
            max_history: Number of recent messages to consider
            
        Returns:
            Self-contained query string
        """
        user_messages = [m for m in messages if m.role == "user"]
        
        if len(user_messages) < 2:
            return user_messages[-1].content if user_messages else ""
        
        last_user_msg = user_messages[-1].content
        second_last_user_msg = user_messages[-2].content if len(user_messages) >= 2 else ""
        
        # Check if the last query references a previous topic
        # Simple heuristic: if the last query is short and contains pronouns/references
        last_words = last_user_msg.split()
        
        # If the query is self-contained (long enough), use as-is
        if len(last_words) > 7:
            return last_user_msg
        
        # Check for pronouns/references that suggest context dependency
        reference_patterns = [
            "چطور", "چه", "که", "این", "آن", "اون", "قبلی", "قبلی", 
            "سال قبل", "ماه قبل", "وقتی", "بعدا", "بعداً", "همین", "همینطور",
            "the", "how", "what", "it", "that", "year", "last", "previous",
        ]
        
        needs_rewrite = any(ref in last_user_msg.lower() for ref in reference_patterns)
        
        if needs_rewrite and second_last_user_msg:
            # Combine the last two queries for a more complete retrieval query
            rewritten = f"{second_last_user_msg} {last_user_msg}"
            logger.info(f"Query rewritten from '{last_user_msg}' to combined context")
            return rewritten
        
        return last_user_msg

    def generate_response(self, messages: list, context: str, model: str = settings.CHAT_MODEL) -> Any:
        """Generate a response using the AI provider"""
        messages_with_context = messages.copy()
        
        # Handle both dataclass and dict messages
        def to_dict(msg):
            if hasattr(msg, 'model_dump'):
                return msg.model_dump()
            elif isinstance(msg, dict):
                return msg
            else:
                return {"role": msg.role, "content": msg.content}
        
        system_msg = next((msg for msg in messages_with_context if to_dict(msg).get("role") == "system"), None)
        
        if system_msg:
            if hasattr(system_msg, 'content'):
                system_msg.content = f"{system_msg.content}\n\nUse this context to answer the question:\n{context}"
            else:
                system_msg["content"] = f"{system_msg['content']}\n\nUse this context to answer the question:\n{context}"

        messages_dict = [to_dict(msg) for msg in messages_with_context]
        return self.provider.create_chat_completion(messages_dict, model)

    async def generate_stream_response(self, messages: list, context: str, model: str = settings.CHAT_MODEL) -> AsyncGenerator[str, None]:
        """Generate a streaming response using the AI provider"""
        messages_with_context = messages.copy()
        
        def to_dict(msg):
            if hasattr(msg, 'model_dump'):
                return msg.model_dump()
            elif isinstance(msg, dict):
                return msg
            else:
                return {"role": msg.role, "content": msg.content}
        
        system_msg = next((msg for msg in messages_with_context if to_dict(msg).get("role") == "system"), None)
        
        if system_msg:
            if hasattr(system_msg, 'content'):
                system_msg.content = f"{system_msg.content}\n\nUse this context to answer the question:\n{context}"
            else:
                system_msg["content"] = f"{system_msg['content']}\n\nUse this context to answer the question:\n{context}"
        
        messages_dict = [to_dict(msg) for msg in messages_with_context]
        stream = self.provider.create_chat_completion_stream(messages_dict, model)
        
        for chunk in stream:
            yield json.dumps({
                "choices": [
                    {
                        "delta": {
                            "content": chunk.choices[0].delta.content if chunk.choices[0].delta.content else "",
                            "role": chunk.choices[0].delta.role if hasattr(chunk.choices[0].delta, 'role') and chunk.choices[0].delta.role else None,
                            "function_call": chunk.choices[0].delta.function_call if hasattr(chunk.choices[0].delta, 'function_call') else None
                        }
                    }
                ],
                "model": chunk.model,
                "object": chunk.object,
                "created": chunk.created,
                "id": chunk.id
            })


rag_service = RAGService()
