from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Document(BaseModel):
    """A document to be embedded and stored."""
    text: str = Field(
        ...,
        description="The text content of the document"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata associated with the document"
    )


class DocumentUpload(BaseModel):
    """Request for uploading a document with chunking support."""
    text: str = Field(..., description="The full document text")
    source: Optional[str] = Field(None, description="Source identifier (e.g., filename, URL)")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for all chunks"
    )


class ChunkResult(BaseModel):
    """Result of a chunk addition."""
    document_id: str = Field(..., description="Unique document identifier")
    chunks_added: int = Field(..., description="Number of chunks added")
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of added chunks")
    message: str = Field(..., description="Status message")


class SearchResultWithScore(BaseModel):
    """A search result with similarity score."""
    id: int = Field(..., description="The unique identifier of the result")
    text: str = Field(..., description="The text content of the result")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata associated with the result")
    score: Optional[float] = Field(None, description="The similarity score")


class RAGResponse(BaseModel):
    """Response from a RAG-enabled chat completion."""
    id: str = Field(..., description="Response ID")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage")

    # RAG-specific metadata
    retrieved_context_used: bool = Field(False, description="Whether RAG context was used")
    retrieval_count: Optional[int] = Field(None, description="Number of documents retrieved")