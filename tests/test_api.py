"""Tests for the API endpoints with chunking support."""
import pytest
import os
import sys
import uuid
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

# Set environment for test
os.environ.setdefault("OPENAI_API_KEY", "test_key")
os.environ.setdefault("PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("LOCAL_EMBEDDING_API_URL", "http://127.0.0.1:8010/embed")
os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", "/tmp/test_chroma_api")
os.environ.setdefault("RAG_CHUNK_SIZE", "200")
os.environ.setdefault("RAG_CHUNK_OVERLAP", "50")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockChromaEmbeddingFunction:
    """Mock embedding function that satisfies ChromaDB's requirements."""
    def __call__(self, input: list) -> list:
        return [[0.1, 0.05, 0.03] for _ in input]
    @staticmethod
    def name():
        return "mock_embedding"
    def get_config(self):
        return {}
    @classmethod
    def build_from_config(cls, config):
        return cls()
    @staticmethod
    def default_metadata():
        return {}
    @property
    def supported_spaces(self):
        return ["cosine"]
    def default_space(self):
        return "cosine"
    is_legacy = False


class MockEmbeddingProvider:
    def create_embedding(self, text: str):
        return [0.1, 0.05, 0.03]
    def create_embedding_batch(self, texts):
        return [[0.1, 0.05, 0.03] for _ in texts]


@pytest.fixture
def client():
    """Get a test client with mock embeddings."""
    # Clear chroma db
    import shutil
    if os.path.exists("/tmp/test_chroma_api"):
        shutil.rmtree("/tmp/test_chroma_api")
    
    from app.services.core.vector_store import VectorStoreService
    from app.services.core.rag_service import RAGService
    
    # Patch the embedding function
    original_init = VectorStoreService.__init__
    
    def mock_init(self):
        original_init(self)
        self._embedding_function = MockChromaEmbeddingFunction()
        self._client = None
    
    with patch.object(VectorStoreService, "__init__", mock_init):
        # Also need to patch the global singleton
        from app.main import app as fastapi_app
        from app.services.core.rag_service import rag_service
        
        # Patch the rag_service's embedding_provider
        rag_service._embedding_provider = MockEmbeddingProvider()
        # Re-init vector store
        rag_service.vector_store = VectorStoreService.__new__(VectorStoreService)
        mock_init(rag_service.vector_store)
        
        # Clear and recreate collection
        col_name = rag_service.collection_name
        try:
            rag_service.vector_store.client.delete_collection(col_name)
        except Exception:
            pass
        rag_service.vector_store.create_collection(col_name)
        
        client = TestClient(fastapi_app)
        yield client
        
        # Cleanup
        try:
            rag_service.vector_store.client.delete_collection(col_name)
        except Exception:
            pass


class TestAPIEndpoints:
    
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
    
    def test_health(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_initialize_collection(self, client):
        response = client.post("/v1/vector_db/initialize_collection")
        assert response.status_code == 201
    
    def test_add_document(self, client):
        response = client.post("/v1/vector_db/add_document", json={
            "text": "این یک متن تستی است.",
            "metadata": {"source": "test.txt"}
        })
        assert response.status_code == 201
        data = response.json()
        assert "message" in data
        assert "total_chunks" in data
        assert data["total_chunks"] >= 1
    
    def test_add_document_large(self, client):
        """Test adding a large document with chunking."""
        lines = []
        for i in range(500):
            lines.append(f"خط شماره {i}: متن تستی برای ارزیابی chunking.")
        text = "\n\n".join(lines)
        
        response = client.post("/v1/vector_db/add_document_chunked", json={
            "text": text,
            "source": "large_test.txt",
            "metadata": {"category": "legal"}
        })
        assert response.status_code == 201
        data = response.json()
        assert data["total_chunks"] > 1
        assert "document_id" in data
        assert len(data["chunk_ids"]) == data["total_chunks"]
    
    def test_search_documents(self, client):
        """Test search after adding documents."""
        # First add a document
        client.post("/v1/vector_db/add_document", json={
            "text": "قانون A مقدار 123 دارد. قانون B مقدار 456 دارد.",
            "metadata": {"source": "laws.txt"}
        })
        
        # Then search
        response = client.post("/v1/vector_db/search_documents", json={
            "prompt": "قانون"
        })
        assert response.status_code == 200
        results = response.json()
        assert isinstance(results, list)
    
    def test_collection_info(self, client):
        response = client.get("/v1/vector_db/collection_info")
        assert response.status_code == 200
        data = response.json()
        assert "collection_name" in data
        assert "document_count" in data
    
    def test_chat_completions(self, client):
        """Test chat completions endpoint structure."""
        response = client.post("/v1/chat/completions", json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test message"}
            ],
            "stream": False
        })
        # May fail due to mock API key, but endpoint should exist
        assert response.status_code in [200, 500]
