from typing import Dict, List, Any, AsyncGenerator
import uuid
import json
from app.config.settings import settings
from app.services.core.vector_store import VectorStoreService
from app.services.providers.openai_service import OpenAIProvider
from app.services.providers.avalai_service import AvalaiProvider
from app.services.providers.openrouter_service import OpenRouterProvider
from app.services.providers.local_embedding_service import LocalEmbeddingProvider
from app.services.base import BaseAIProvider

class RAGService:
    def __init__(self):
        self.collection_name = "RAG_COLLECTION"
        self.vector_store = VectorStoreService()
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

    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> None:
        """Add a document to the vector store"""
        self.vector_store.add_documents(
            collection_name=self.collection_name,
            documents=[text],
            ids=[f"doc_{uuid.uuid4()}"],
            metadatas=[metadata] if metadata else None
        )

    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        self.vector_store.clear_collection(self.collection_name)

    def search_similar_documents(self, embedding: List[float]) -> Dict[str, Any]:
        """Search for similar documents using the provided embedding"""
        return self.vector_store.search(
            collection_name=self.collection_name,
            query_embeddings=embedding,
        )
    
    def generate_response(self, messages: list, context: str, model: str = settings.CHAT_MODEL) -> Any:
        """Generate a response using the AI provider"""
        messages_with_context = messages.copy()
        system_msg = next((msg for msg in messages_with_context if msg.role == "system"), None)
        
        if system_msg:
            system_msg.content = f"{system_msg.content}\n\nUse this context to answer the question:\n{context}"
        else:
            messages_with_context.insert(0, {
                "role": "system",
                "content": f"Use this context to enhance your responses:\n{context}"
            })

        messages_dict = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg
            for msg in messages_with_context
        ]
        return self.provider.create_chat_completion(messages_dict, model)

    async def generate_stream_response(self, messages: list, context: str, model: str = settings.CHAT_MODEL) -> AsyncGenerator[str, None]:
        """Generate a streaming response using the AI provider"""
        messages_with_context = messages.copy()
        system_msg = next((msg for msg in messages_with_context if msg.role == "system"), None)
        
        if system_msg:
            system_msg.content = f"{system_msg.content}\n\nUse this context to answer the question:\n{context}"
        else:
            messages_with_context.insert(0, {
                "role": "system",
                "content": f"Use this context to enhance your responses:\n{context}"
            })
        
        messages_dict = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg
            for msg in messages_with_context
        ]
        stream = self.provider.create_chat_completion_stream(messages_dict, model)
        
        for chunk in stream:
            yield json.dumps({
                "choices": [
                    {
                        "delta": {
                            "content": chunk.choices[0].delta.content,
                            "role": chunk.choices[0].delta.role if hasattr(chunk.choices[0].delta, 'role') else None,
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