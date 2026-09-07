from pydantic_settings import BaseSettings
from typing import Literal, Optional
from functools import lru_cache

class Settings(BaseSettings):
    # AI Provider Settings
    PROVIDER: Literal["openai", "avalai", "openrouter"] = "openai"

    # API Access Control — همهٔ درخواست‌ها باید X-API-Key داشته باشند
    RAG_API_KEY: str = ""
    
    # OpenAI Settings
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"
    
    # AvalAI Settings / میتوانید از وب سرویس aval ai استفاده کنید
    AVALAI_API_KEY: Optional[str] = None
    AVALAI_BASE_URL: str = "https://api.avalapis.ir/v1"
    
    # OpenRouter Settings
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_CHAT_MODEL: str = "deepseek/deepseek-v4-flash-0731"
    OPENROUTER_EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
    
    # Local Embedding Settings / امبدینگ محلی با HuggingFace
    # وقتی EMBEDDING_PROVIDER=local باشد، مدل sentence-transformers زیر برای امبدینگ بارگذاری می‌شود
    EMBEDDING_PROVIDER: Optional[Literal["openai", "avalai", "openrouter", "local"]] = None
    LOCAL_EMBEDDING_MODEL: str = "xmanii/maux-gte-persian"
    LOCAL_EMBEDDING_API_URL: str = "http://127.0.0.1:8010/embed"
    
    # Vector Store Settings
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    RAG_SEARCH_LIMIT: int = 3

    # RAG Search Settings
    RAG_RETRIEVAL_K: int = 10  # Number of results to retrieve from vector DB
    RAG_FINAL_K: int = 3       # Number of results to send to LLM

    # Candidate pool scanned before lexical re-rank. Large documents (hundreds of
    # chunks) need a bigger pool so rare exact-value chunks (e.g. an identifier)
    # survive dense ranking and reach the lexical boost step.
    RAG_RETRIEVAL_CANDIDATES: int = 300

    # Document Chunking Settings
    RAG_CHUNK_SIZE: int = 800      # Characters per chunk (not tokens - chars are simpler)
    RAG_CHUNK_OVERLAP: int = 100   # Characters of overlap between chunks

    # Embedding batching (chunks are embedded+stored in batches of this size)
    RAG_EMBEDDING_BATCH_SIZE: int = 32
    
    # System Settings
    SYSTEM_PROMPT: str = (
        "You are a helpful assistant. Use the provided context to answer "
        "the user's question. If the context is not relevant, just say 'I don't know'"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True

    def validate_api_keys(self):
        """Validate that the required API key is present based on the selected provider"""
        if self.PROVIDER == "avalai" and not self.AVALAI_API_KEY:
            raise ValueError("AVALAI_API_KEY is required when using AvalAI provider")
        elif self.PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
        elif self.PROVIDER == "openrouter" and not self.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required when using OpenRouter provider")

    @property
    def effective_embedding_provider(self) -> str:
        """Resolve the embedding provider. Falls back to PROVIDER when EMBEDDING_PROVIDER is unset, preserving backward compatibility."""
        return self.EMBEDDING_PROVIDER if self.EMBEDDING_PROVIDER else self.PROVIDER

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    settings.validate_api_keys()
    return settings

settings = get_settings()