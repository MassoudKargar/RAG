"""Pytest configuration and shared fixtures."""
import os
import sys

# Set test environment variables BEFORE any imports
os.environ.setdefault("OPENAI_API_KEY", "test_key")
os.environ.setdefault("PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("LOCAL_EMBEDDING_API_URL", "http://127.0.0.1:8010/embed")
os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", "/tmp/test_chroma_db")
os.environ.setdefault("RAG_CHUNK_SIZE", "800")
os.environ.setdefault("RAG_CHUNK_OVERLAP", "100")
os.environ.setdefault("RAG_RETRIEVAL_K", "10")
os.environ.setdefault("RAG_FINAL_K", "3")
os.environ.setdefault("RAG_SEARCH_LIMIT", "3")

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import uuid
from app.services.core.vector_store import VectorStoreService


@pytest.fixture
def mock_embedding_function():
    """Mock embedding function for ChromaDB that doesn't require external services."""
    class MockEmbeddingFunction:
        def __call__(self, input: list) -> list:
            return [[0.1, 0.05, 0.03, 0.01, 0.02] for _ in input]
        
        @staticmethod
        def name():
            return 'mock_embedding'
        
        def get_config(self):
            return {}
        
        @classmethod
        def build_from_config(cls, config):
            return cls()
        
        @staticmethod
        def default_metadata():
            return {}
    
    return MockEmbeddingFunction()


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider for the RAG service."""
    class MockEmbeddingProvider:
        def create_embedding(self, text: str):
            return [0.1, 0.05, 0.03, 0.01, 0.02]
        
        def create_embedding_batch(self, texts):
            return [[0.1, 0.05, 0.03, 0.01, 0.02] for _ in texts]
    
    return MockEmbeddingProvider()


@pytest.fixture
def vector_store(mock_embedding_function):
    """Vector store with mock embedding function."""
    vs = VectorStoreService()
    vs._embedding_function = mock_embedding_function
    vs._client = None
    return vs


@pytest.fixture
def test_collection(vector_store):
    """Create a unique test collection, yield it, then clean up."""
    col_name = f"test_col_{uuid.uuid4().hex[:8]}"
    vector_store.create_collection(col_name)
    yield col_name
    try:
        vector_store.client.delete_collection(col_name)
        try:
            vector_store.client._pop_collection_from_cache(col_name)
        except Exception:
            pass
    except Exception:
        pass
