"""Tests for the vector store service with chunking."""
import pytest
import os
import sys

from app.services.core.vector_store import VectorStoreService


class TestVectorStoreWithChunks:
    """Test that VectorStoreService properly handles chunks."""
    
    def test_add_chunks_with_metadata(self, vector_store, test_collection):
        """Test adding chunks with full metadata to collection."""
        chunks = [
            {
                "text": "First chunk content about topic A",
                "metadata": {
                    "document_id": "doc_1",
                    "chunk_index": 0,
                    "total_chunks": 2,
                    "source": "test.txt",
                    "category": "test"
                },
                "id": "doc_1_0",
            },
            {
                "text": "Second chunk content about topic B",
                "metadata": {
                    "document_id": "doc_1",
                    "chunk_index": 1,
                    "total_chunks": 2,
                    "source": "test.txt",
                    "category": "test"
                },
                "id": "doc_1_1",
            }
        ]
        
        result = vector_store.add_chunks_with_metadata(
            collection_name=test_collection,
            chunks=chunks,
            batch_size=10
        )
        
        assert result["status"] == "success"
        assert result["total_chunks"] == 2
        assert result["inserted"] == 2
    
    def test_add_documents_with_chunk_ids(self, vector_store, test_collection):
        """Test add_documents with proper chunk IDs."""
        documents = ["Chunk text 1", "Chunk text 2", "Chunk text 3"]
        ids = ["doc_abc_0", "doc_abc_1", "doc_abc_2"]
        metadatas = [
            {"document_id": "doc_abc", "chunk_index": 0},
            {"document_id": "doc_abc", "chunk_index": 1},
            {"document_id": "doc_abc", "chunk_index": 2},
        ]
        
        result = vector_store.add_documents(
            collection_name=test_collection,
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        
        assert result["status"] == "success"
        assert result["documents_added"] == 3
        
        # Verify the data was stored correctly
        coll = vector_store.get_collection(test_collection)
        retrieved = coll.get(ids=["doc_abc_0"])
        assert len(retrieved["documents"]) == 1
        assert retrieved["metadatas"][0]["document_id"] == "doc_abc"
    
    def test_collection_count(self, vector_store, test_collection):
        """Test that collection count works."""
        count = vector_store.get_collection_count(test_collection)
        assert count == 0
        
        vector_store.add_documents(
            collection_name=test_collection,
            documents=["test"],
            ids=["test_1"],
        )
        
        count = vector_store.get_collection_count(test_collection)
        assert count == 1
    
    def test_reset_collection(self, vector_store, test_collection):
        """Test collection reset."""
        vector_store.add_documents(
            collection_name=test_collection,
            documents=["test"],
            ids=["test_1"],
        )
        
        vector_store.reset_collection(test_collection)
        
        count = vector_store.get_collection_count(test_collection)
        assert count == 0
    
    def test_search_returns_metadata(self, vector_store, test_collection):
        """Test that search returns metadata."""
        chunks = [
            {
                "text": "This is chunk about Persian law قانون A",
                "metadata": {"document_id": "doc1", "chunk_index": 0},
                "id": "doc1_0",
            },
            {
                "text": "This is chunk about something else",
                "metadata": {"document_id": "doc1", "chunk_index": 1},
                "id": "doc1_1",
            }
        ]
        
        vector_store.add_chunks_with_metadata(test_collection, chunks, batch_size=10)
        
        # Search using ChromaDB directly with simple embedding
        coll = vector_store.get_collection(test_collection)
        results = coll.query(
            query_embeddings=[[0.1, 0.05, 0.03, 0.01, 0.02]],
            n_results=2
        )
        
        assert len(results["documents"][0]) == 2
        assert results["metadatas"][0][0]["document_id"] == "doc1"
        assert results["metadatas"][0][0]["chunk_index"] == 0
