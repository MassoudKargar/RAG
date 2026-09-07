from typing import Dict, List, Any, AsyncGenerator, Optional
import uuid
import json
import re
import logging
from app.config.settings import settings
from app.services.core.vector_store import VectorStoreService
from app.services.core.chunker import ChunkerService, Chunk
from app.services.providers.openai_service import OpenAIProvider
from app.services.providers.avalai_service import AvalaiProvider
from app.services.providers.openrouter_service import OpenRouterProvider
from app.services.providers.local_embedding_service import LocalEmbeddingProvider
from app.services.base import BaseAIProvider

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_BATCH_SIZE = 32


def _as_message_dict(message: Any) -> Dict[str, Any]:
    """Normalize a chat message (dict or pydantic model) to a plain dict."""
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return {"role": getattr(message, "role", "user"), "content": getattr(message, "content", "")}


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

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------
    def add_document(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
        document_id: Optional[str] = None,
        source: Optional[str] = None,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> Dict[str, Any]:
        """Chunk + embed + store a (possibly very large) document.

        Returns a summary dict: {document_id, chunks_added, chunk_ids, message}.
        The document is replaced (same document_id -> chunks are upserted), so
        re-adding the same document does not create duplicates.
        """
        if document_id is None and metadata and isinstance(metadata, dict):
            document_id = metadata.get("document_id")
        if document_id is None:
            document_id = f"doc_{uuid.uuid4().hex}"
        batch_size = batch_size or getattr(settings, "RAG_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)
        batch_size = max(1, int(batch_size))

        # Canonical metadata propagated to every chunk (standardized fields for
        # SEC/10-K style corpora: company / fiscal_year / document_type).
        normalized_meta = dict(metadata or {})
        if "document_id" not in normalized_meta:
            normalized_meta["document_id"] = document_id
        if "chunk_id" in normalized_meta:
            normalized_meta.pop("chunk_id")  # chunker owns chunk_id
        if source is None and metadata:
            source = metadata.get("source") if isinstance(metadata, dict) else None

        chunks = self.chunker.chunk_document(
            text, document_id=document_id, source=source, metadata=normalized_meta
        )
        if not chunks:
            raise ValueError("Document is empty or produced no chunks")

        total = len(chunks)
        logger.info("Document id=%s: total chunks=%d batch_size=%d", document_id, total, batch_size)

        chunk_ids: List[str] = []
        for start in range(0, total, batch_size):
            batch = chunks[start:start + batch_size]
            ids = [c.id or f"{document_id}::chunk_{i:05d}" for i, c in enumerate(batch, start=start)]
            try:
                self.vector_store.upsert_documents(
                    collection_name=self.collection_name,
                    documents=[c.text for c in batch],
                    ids=ids,
                    metadatas=[c.metadata for c in batch],
                )
            except Exception as e:
                logger.error(
                    "Embedding/storage failed for document id=%s at batch %d..%d: %s",
                    document_id, start + 1, min(start + batch_size, total), str(e),
                )
                raise
            chunk_ids.extend(ids)
            logger.info("Embedding progress: %d/%d", min(start + batch_size, total), total)

        logger.info("Document id=%s stored successfully (%d chunks)", document_id, total)
        return {
            "document_id": document_id,
            "chunks_added": total,
            "chunk_ids": chunk_ids,
            "message": f"Document embedded and added successfully ({total} chunks)",
        }

    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        self.vector_store.clear_collection(self.collection_name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _lexical_tokens(self, text: str) -> set:
        """Extract searchable tokens: identifiers (TEST-003), numbers, words."""
        tokens = set()
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9_-]*|[۰-۹0-9]+", text):
            tokens.add(m.group(0).lower())
        return tokens

    def search_similar_documents(self, embedding: List[float], limit: Optional[int] = None, query: Optional[str] = None) -> Dict[str, Any]:
        """Search for similar documents using the provided embedding.

        ``limit`` defaults to settings.RAG_RETRIEVAL_K (can be larger than the
        final context size). Results include distances (lower = more similar).

        Pipeline:
        1. Vector search over a candidate pool of
           ``settings.RAG_RETRIEVAL_CANDIDATES`` chunks (default 300).
        2. When ``query`` is provided, lexical re-rank the candidate pool:
           chunks containing rare query tokens (identifiers such as
           ``TEST-003``, numbers) float to the top via a distance boost.
        3. Return the top ``limit`` (RAG_RETRIEVAL_K) after re-rank.
        """
        candidates = getattr(settings, "RAG_RETRIEVAL_CANDIDATES", 300)
        if limit is not None:
            candidates = max(candidates, limit)
        result = self.vector_store.search(
            collection_name=self.collection_name,
            query_embeddings=embedding,
            n_results=candidates,
        )
        if not query:
            return result

        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        if not docs:
            return result

        query_tokens = self._lexical_tokens(query)
        # identifiers/numbers are the high-value tokens for exact lookups
        valuable = {t for t in query_tokens if t[0].isalpha() and any(ch.isdigit() for ch in t)} | \
                   {t for t in query_tokens if t.isdigit() and len(t) >= 2}

        def score(i: int) -> float:
            doc_tokens = self._lexical_tokens(docs[i])
            overlap = len(query_tokens & doc_tokens)
            valuable_hit = len(valuable & doc_tokens)
            # lexical bonus: valuable token hit = 2.0 distance boost, word hit = 0.5
            return dists[i] - (valuable_hit * 2.0 + (overlap - valuable_hit) * 0.5)

        order = sorted(range(len(docs)), key=score)
        final_k = limit if limit is not None else settings.RAG_RETRIEVAL_K
        order = [i for i in order][:final_k]
        ids_inner = result.get("ids")
        if ids_inner and isinstance(ids_inner[0], list):
            ids_sorted = [ids_inner[0][i] for i in order]
        else:
            ids_sorted = [ids_inner[i] for i in order] if ids_inner else []
        return {
            "ids": [ids_sorted],
            "documents": [[docs[i] for i in order]],
            "metadatas": [[metas[i] for i in order]] if metas else [],
            "distances": [[dists[i] for i in order]] if dists else [],
        }

    def build_context(self, search_result: Dict[str, Any], limit: Optional[int] = None) -> str:
        """Build a clean, labeled context string from a ChromaDB search result.

        Each block is labeled with document / section / chunk position, without
        leaking raw internal metadata.
        """
        limit = limit if limit is not None else settings.RAG_FINAL_K
        documents = (search_result.get("documents") or [[]])[0]
        metadatas = (search_result.get("metadatas") or [[]])[0]
        distances = search_result.get("distances") or [[None] * len(documents)]

        parts: List[str] = []
        for i, (doc, meta, dist) in enumerate(zip(documents[:limit], metadatas[:limit], distances[0][:limit])):
            if doc is None:
                continue
            label_bits = []
            doc_id = (meta or {}).get("document_id")
            if doc_id:
                label_bits.append(f"Document: {doc_id}")
            section = (meta or {}).get("section")
            if section:
                label_bits.append(f"Section: {section}")
            chunk_idx = (meta or {}).get("chunk_index")
            total = (meta or {}).get("total_chunks")
            if chunk_idx is not None and total:
                label_bits.append(f"Chunk: {chunk_idx + 1}/{total}")
            header = " | ".join(label_bits) if label_bits else f"Source {i + 1}"
            parts.append(f"[{header}]\n{doc}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def _prepare_messages(self, messages: list, context: str) -> List[Dict[str, Any]]:
        """Normalize messages and inject RAG context into the system prompt."""
        normalized = [_as_message_dict(m) for m in messages]
        system_content = "You are a helpful assistant. Use the provided context to answer the user's question. If the context is not relevant, just say 'I don't know'."
        if context:
            system_content = f"{system_content}\n\nUse this context to answer the question:\n{context}"
        result = []
        inserted = False
        for msg in normalized:
            if msg.get("role") == "system":
                merged = f"{msg.get('content', '')}\n\n{system_content}" if msg.get("content") else system_content
                result.append({"role": "system", "content": merged})
                inserted = True
            else:
                result.append(msg)
        if not inserted:
            result.insert(0, {"role": "system", "content": system_content})
        return result

    def _extract_content(self, completion: Any) -> str:
        """Extract assistant text from a provider completion object."""
        try:
            if hasattr(completion, "choices"):
                choice = completion.choices[0]
                msg = getattr(choice, "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if isinstance(content, list):
                        # OpenAI content arrays (e.g. text parts)
                        return "".join(
                            str(part.get("text", "")) if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    if content:
                        return str(content)
                # streaming chunk shape
                delta = getattr(choice, "delta", None)
                if delta is not None and getattr(delta, "content", None):
                    return str(delta.content)
                return ""
            if isinstance(completion, dict):
                return self._extract_content_from_dict(completion)
        except Exception:
            logger.debug("Could not extract content from completion", exc_info=True)
        return str(completion)

    def _extract_content_from_dict(self, completion: dict) -> str:
        choices = completion.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        if isinstance(choice, dict):
            msg = choice.get("message") or choice.get("delta") or {}
            if isinstance(msg, dict):
                return str(msg.get("content") or "")
        return str(choice)

    def _resolve_model(self, model: Optional[str]) -> str:
        """Resolve the chat model for a request.

        Explicit ``model`` wins. Otherwise, pick the provider-appropriate
        default (OpenRouter has its own default model) instead of always
        falling back to ``CHAT_MODEL``.
        """
        if model:
            return model
        if settings.PROVIDER == "openrouter":
            return settings.OPENROUTER_CHAT_MODEL
        return settings.CHAT_MODEL

    def generate_response(self, messages: list, context: str = "", model: Optional[str] = None) -> Any:
        """Generate a response using the AI provider, returning the raw completion."""
        model = self._resolve_model(model)
        messages_dict = self._prepare_messages(messages, context)
        return self.provider.create_chat_completion(messages_dict, model)

    async def generate_stream_response(self, messages: list, context: str = "", model: Optional[str] = None):
        """Generate a streaming response.

        Providers return either a sync iterator (OpenAI SDK) or an async
        iterator; this adapter normalizes both into an async generator of
        OpenAI-style SSE chunks (JSON strings).
        """
        model = self._resolve_model(model)
        messages_dict = self._prepare_messages(messages, context)
        stream = self.provider.create_chat_completion_stream(messages_dict, model)

        if hasattr(stream, "__aiter__"):
            async for chunk in stream:
                yield json.dumps({
                    "choices": [
                        {
                            "delta": {
                                "content": self._extract_content(chunk),
                                "role": "assistant",
                            }
                        }
                    ],
                    "model": getattr(chunk, "model", model) or model,
                    "object": getattr(chunk, "object", "chat.completion.chunk"),
                    "created": getattr(chunk, "created", 0),
                    "id": getattr(chunk, "id", ""),
                })
            return

        for chunk in stream:
            content = self._extract_content(chunk)
            if not content:
                continue
            yield json.dumps({
                "choices": [{"delta": {"content": content, "role": "assistant"}}],
                "model": getattr(chunk, "model", model) or model,
                "object": getattr(chunk, "object", "chat.completion.chunk"),
                "created": getattr(chunk, "created", 0),
                "id": getattr(chunk, "id", ""),
            })


rag_service = RAGService()