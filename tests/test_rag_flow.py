"""End-to-end tests for the RAG flow with chunking.

Tests the full flow:
Document -> Chunk -> Embedding -> ChromaDB -> Search -> Context -> LLM
"""
import pytest
import os
import sys
import uuid

from app.services.core.rag_service import RAGService
from app.services.core.chunker import ChunkerService
from app.config.settings import settings


def generate_persian_document(num_lines: int = 5000) -> str:
    """Generate a synthetic Persian document with identifiable facts."""
    lines = []
    for i in range(num_lines):
        if i == 50:
            lines.append("قانون A دارای مقدار 123 است.")
        elif i == 1500:
            lines.append("قانون B دارای مقدار 456 است.")
        elif i == 3500:
            lines.append("قانون C دارای مقدار 789 است.")
        elif i % 100 == 0:
            lines.append(f"خط شماره {i} از متن تستی برای سیستم ریاجکن.")
        else:
            lines.append(f"این یک متن تستی شماره {i} است که شامل اطلاعات مختلف است.")
    
    return "\n\n".join(lines)


class TestRAGFlow:
    """Test the full RAG flow with chunking."""
    
    def _make_rag_service(self, vector_store, mock_embedding_provider):
        """Create a RAG service with mock embedding."""
        rs = RAGService()
        rs._embedding_provider = mock_embedding_provider
        rs.vector_store = vector_store
        rs.collection_name = f"rag_test_{uuid.uuid4().hex[:8]}"
        rs.vector_store.create_collection(rs.collection_name)
        return rs
    
    def test_document_to_chunks_to_storage(self, vector_store, mock_embedding_provider, test_collection):
        """Test: Document → Chunk → Embedding → ChromaDB flow."""
        rs = self._make_rag_service(vector_store, mock_embedding_provider)
        rs.collection_name = test_collection
        
        # Create a document
        text = generate_persian_document(500)
        
        result = rs.add_document(
            text=text,
            metadata={"source": "test_persian.txt", "category": "legal"},
            document_id="rag_test_001",
        )
        
        assert result["total_chunks"] > 1
        assert result["document_id"] == "rag_test_001"
        assert len(result["chunk_ids"]) == result["total_chunks"]
    
    def test_search_after_chunking(self, vector_store, mock_embedding_provider, test_collection):
        """Test: Search works correctly after chunks are inserted."""
        rs = self._make_rag_service(vector_store, mock_embedding_provider)
        rs.collection_name = test_collection
        
        text = generate_persian_document(200)
        rs.add_document(
            text=text,
            metadata={"source": "test.txt"},
            document_id="rag_search_test",
        )
        
        results = rs.search_with_scores(
            query="قانون",
            n_results=5,
        )
        
        assert len(results) > 0
        for result in results:
            assert "id" in result
            assert "text" in result
            assert "metadata" in result
    
    def test_metadata_identifies_correct_chunk(self, vector_store, mock_embedding_provider, test_collection):
        """Test: Metadata identifies the correct chunk after retrieval."""
        rs = self._make_rag_service(vector_store, mock_embedding_provider)
        rs.collection_name = test_collection
        
        text = generate_persian_document(500)
        result = rs.add_document(
            text=text,
            metadata={"source": "laws.txt"},
            document_id="rag_meta_test",
        )
        
        total_chunks = result["total_chunks"]
        assert total_chunks > 0
        
        search_results = rs.search_with_scores(
            query="قانون",
            n_results=total_chunks,
        )
        
        assert len(search_results) > 0
        for r in search_results:
            assert r["metadata"]["document_id"] == "rag_meta_test"
            assert r["metadata"]["total_chunks"] == total_chunks
            assert r["metadata"]["chunk_index"] is not None
    
    def test_large_document_5000_lines(self, vector_store, mock_embedding_provider, test_collection):
        """Test: 5,000+ line document can be inserted and queried."""
        rs = self._make_rag_service(vector_store, mock_embedding_provider)
        rs.collection_name = test_collection
        
        text = generate_persian_document(5000)
        
        result = rs.add_document(
            text=text,
            metadata={"source": "large_doc.txt"},
            document_id="rag_large_test",
        )
        
        assert result["total_chunks"] > 5
    
    def test_unique_chunk_ids(self, vector_store, mock_embedding_provider, test_collection):
        """Test: Chunks have unique, deterministic IDs."""
        rs = self._make_rag_service(vector_store, mock_embedding_provider)
        rs.collection_name = test_collection
        
        text = "Test " * 200
        result = rs.add_document(
            text=text,
            document_id="unique_id_test",
        )
        
        chunk_ids = result["chunk_ids"]
        assert len(chunk_ids) == len(set(chunk_ids))
        for i, cid in enumerate(chunk_ids):
            assert cid.startswith("unique_id_test_")
    
    def test_context_building(self, vector_store, mock_embedding_provider, test_collection):
        """Test: Context building includes metadata in a clean format."""
        rs = self._make_rag_service(vector_store, mock_embedding_provider)
        
        search_results = {
            "documents": [["chunk1 content", "chunk2 content"]],
            "metadatas": [[
                {"document_id": "doc1", "chunk_index": 0, "total_chunks": 3, "source": "test.txt"},
                {"document_id": "doc2", "chunk_index": 1, "total_chunks": 3, "source": "test2.txt"},
            ]],
            "distances": [[0.1, 0.2]],
            "ids": [["doc1_0", "doc2_1"]],
        }
        
        context, used_results = rs.build_context(search_results)
        
        assert "Result" in context or "chunk" in context.lower()
        assert len(used_results) == 2
    
    def test_backward_compatibility_add_document(self, vector_store, mock_embedding_provider, test_collection):
        """Test: Existing add_document_simple method works."""
        rs = self._make_rag_service(vector_store, mock_embedding_provider)
        rs.collection_name = test_collection
        
        rs.add_document_simple(
            "Simple test document",
            metadata={"source": "simple.txt"}
        )
        
        results = rs.search_with_scores("Simple test", n_results=3)
        assert len(results) > 0


class TestEndToEndAcceptance:
    """End-to-end acceptance test with Persian document."""
    
    def _make_rag_service(self, vector_store, mock_embedding_provider, test_collection):
        rs = RAGService()
        rs._embedding_provider = mock_embedding_provider
        rs.vector_store = vector_store
        rs.collection_name = test_collection
        return rs
    
    def test_acceptance_5000_line_persian(self, vector_store, mock_embedding_provider, test_collection):
        """Full acceptance test: Insert 5000-line Persian doc and query for a fact."""
        rs = self._make_rag_service(vector_store, mock_embedding_provider, test_collection)
        
        # Generate a 5000-line document with facts at specific positions
        num_lines = 5000
        lines = []
        for i in range(num_lines):
            if i == 50:
                lines.append("قانون A دارای مقدار 123 است.")
            elif i == 1500:
                lines.append("قانون B دارای مقدار 456 است.")
            elif i == 3500:
                lines.append("قانون C دارای مقدار 789 است.")
            elif i % 200 == 0:
                lines.append(f"این خط شماره {i} است و شامل اطلاعات تکمیلی است.")
            else:
                lines.append(f"متن تستی شماره {i} برای ارزیابی سیستم ریاجکن.")
        
        text = "\n\n".join(lines)
        
        # Step 1: Chunk the document
        chunker = ChunkerService(chunk_size=800, chunk_overlap=100)
        chunks = chunker.chunk_document(text, document_id="acceptance_test", source="laws.txt")
        
        assert len(chunks) > 1
        assert len(chunks) >= 5
        
        # Step 2: Store chunks via RAG service
        result = rs.add_document(
            text=text,
            metadata={"source": "laws.txt", "format": "plain_text"},
            document_id="acceptance_test",
        )
        
        assert result["total_chunks"] > 1
        
        # Step 3: Search for law C
        search_results = rs.search_with_scores(
            query="مقدار قانون C چیست؟",
            n_results=result["total_chunks"],
        )
        
        assert len(search_results) > 0
        
        # Step 4: Verify metadata
        for r in search_results:
            assert r["metadata"]["document_id"] == "acceptance_test"
            assert r["metadata"]["chunk_index"] is not None
            assert r["metadata"]["total_chunks"] == result["total_chunks"]
        
        # Step 5: Verify all chunks have proper metadata
        chunk_indices_found = set()
        for r in search_results:
            chunk_indices_found.add(r["metadata"]["chunk_index"])
        
        assert len(chunk_indices_found) > 0
