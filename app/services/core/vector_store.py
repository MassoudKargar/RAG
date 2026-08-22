import chromadb
from chromadb.config import Settings as ChromaSettings
import chromadb.utils.embedding_functions as embedding_functions
from typing import List, Dict, Any, Optional, Tuple
import uuid
import logging
import time
from app.config.settings import settings
from app.services.providers.openai_service import OpenAIProvider
from app.services.providers.avalai_service import AvalaiProvider
from app.services.providers.openrouter_service import OpenRouterProvider
from app.services.providers.local_embedding_service import HTTPChromaEmbeddingFunction

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        """Initialize the vector store with the appropriate embedding function."""
        self._embedding_function = None
        self._client = None
        self._collection = None
        
    @property
    def embedding_function(self):
        """Lazy initialization of embedding function."""
        if self._embedding_function is None:
            eff = settings.effective_embedding_provider
            if eff == "local":
                self._embedding_function = HTTPChromaEmbeddingFunction(settings.LOCAL_EMBEDDING_API_URL)
            elif eff == "openai":
                self._embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=settings.OPENAI_API_KEY,
                    model_name=settings.OPENAI_EMBEDDING_MODEL
                )
            elif eff == "avalai":
                self._embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=settings.AVALAI_API_KEY,
                    api_base=settings.AVALAI_BASE_URL,
                    model_name=settings.OPENAI_EMBEDDING_MODEL
                )
            elif eff == "openrouter":
                self._embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=settings.OPENROUTER_API_KEY,
                    api_base=settings.OPENROUTER_BASE_URL,
                    model_name=settings.OPENROUTER_EMBEDDING_MODEL
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {eff}")
        return self._embedding_function
            
    @property
    def client(self) -> chromadb.PersistentClient:
        """Get or create the ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
        return self._client

    def create_collection(self, collection_name: str):
        """Create a new collection or get existing one"""
        try:
            return self.client.create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
        except Exception as e:
            # ChromaDB may raise ValueError, InternalError, or other exceptions
            # when the collection already exists
            if "already exists" in str(e).lower() or "Collection" in str(e):
                return self.get_collection(collection_name)
            raise e

    def get_collection(self, collection_name: str):
        """Get an existing collection"""
        return self.client.get_collection(
            name=collection_name, 
            embedding_function=self.embedding_function
        )

    def get_or_create_collection(self, collection_name: str):
        """Get an existing collection or create a new one"""
        return self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embedding_function
        )

    def add_documents(
        self, 
        collection_name: str, 
        documents: List[str], 
        ids: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Add documents to a collection.
        
        This method now properly stores chunks with:
        - Metadata preserved and augmented
        - Unique IDs generated if not provided
        - Batch insertion for efficiency
        """
        if not documents:
            logger.warning("Attempted to add empty document list")
            return {"status": "skipped", "reason": "no documents provided"}
        
        collection = self.get_or_create_collection(collection_name)
        
        # Generate IDs if not provided
        if not ids:
            ids = [f"chunk_{uuid.uuid4()}" for _ in documents]
        
        # Ensure we have the right number of items
        n_docs = len(documents)
        if len(ids) != n_docs:
            ids = ids[:n_docs] if len(ids) > n_docs else ids + [f"chunk_{uuid.uuid4()}" for _ in range(n_docs - len(ids))]
        
        # Create default metadata if not provided
        if metadatas is None:
            metadatas = [{"source": "unnamed"} for _ in range(n_docs)]
        elif len(metadatas) != n_docs:
            # Extend or truncate metadata to match documents
            if len(metadatas) < n_docs:
                metadatas = metadatas + [{"source": "unnamed"} for _ in range(n_docs - len(metadatas))]
            else:
                metadatas = metadatas[:n_docs]
        
        # ChromaDB rejects empty metadata dicts, ensure each has at least one key
        for i in range(len(metadatas)):
            if not metadatas[i]:
                metadatas[i] = {"source": "unnamed"}
        
        start_time = time.time()
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Added {n_docs} documents to {collection_name} in {elapsed:.2f}s")
        
        return {
            "status": "success",
            "documents_added": n_docs,
            "collection": collection_name,
            "processing_time_seconds": elapsed
        }

    def add_chunks_with_metadata(
        self, 
        collection_name: str,
        chunks: List[Dict[str, Any]],
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Add pre-chunked documents with full metadata in batches.
        
        Args:
            collection_name: Name of the collection
            chunks: List of dicts with 'text', 'metadata' keys
            batch_size: Number of chunks to insert per batch
            
        Returns:
            Summary of the operation
        """
        if not chunks:
            logger.warning("Attempted to add empty chunks list")
            return {"status": "skipped", "reason": "no chunks provided"}
        
        collection = self.get_or_create_collection(collection_name)
        
        total_chunks = len(chunks)
        total_inserted = 0
        
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            
            texts = [chunk["text"] for chunk in batch]
            ids = [chunk.get("id", f"chunk_{uuid.uuid4()}") for chunk in batch]
            metadatas = [chunk.get("metadata", {}) for chunk in batch]
            
            start_time = time.time()
            collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            elapsed = time.time() - start_time
            
            total_inserted += len(batch)
            logger.info(f"Batch {i//batch_size + 1}: inserted {len(batch)} chunks ({elapsed:.2f}s)")
        
        return {
            "status": "success",
            "total_chunks": total_chunks,
            "inserted": total_inserted,
            "collection": collection_name
        }
        
    def search(
        self, 
        collection_name: str, 
        query_embeddings: List[float],
        n_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents in a collection.
        
        Args:
            collection_name: Name of the collection
            query_embeddings: Query embedding vector
            n_results: Number of results to retrieve (defaults to RAG_SEARCH_LIMIT or RAG_RETRIEVAL_K)
            
        Returns:
            Dict with 'documents', 'ids', 'metadatas', 'distances' keys
        """
        collection = self.get_or_create_collection(collection_name)
        
        # Use retrieval K if not specified, preferring higher value for more context
        k = n_results or settings.RAG_RETRIEVAL_K or settings.RAG_SEARCH_LIMIT or 3
        
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=k
        )
    
    def search_with_scores(
        self, 
        collection_name: str, 
        query_embeddings: List[float],
        n_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents and return enriched results with scores.
        
        Returns:
            List of dicts with 'id', 'text', 'metadata', 'score' keys
        """
        results = self.search(collection_name, query_embeddings, n_results)
        
        enriched_results = []
        n = len(results.get("documents", [[]])[0])
        
        for i in range(n):
            result = {
                "id": results.get("ids", [[""]])[0][i],
                "text": results.get("documents", [[]])[0][i],
                "metadata": results.get("metadatas", [[None]])[0][i] or {},
                "score": results.get("distances", [[]])[0][i] if results.get("distances") else None,
            }
            enriched_results.append(result)
        
        return enriched_results
    
    def clear_collection(self, collection_name: str):
        """Delete a collection"""
        collection = self.get_or_create_collection(collection_name)
        return collection.delete()
    
    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of items in a collection."""
        collection = self.get_or_create_collection(collection_name)
        try:
            return collection.count()
        except Exception:
            return 0
    
    def reset_collection(self, collection_name: str):
        """Delete and recreate a collection (useful for testing)."""
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        try:
            self.client._pop_collection_from_cache(collection_name)
        except Exception:
            pass
        return self.get_or_create_collection(collection_name)


# Global singleton instance
rag_vector_store = None

def get_vector_store() -> VectorStoreService:
    """Get the singleton vector store instance."""
    global rag_vector_store
    if rag_vector_store is None:
        rag_vector_store = VectorStoreService()
    return rag_vector_store