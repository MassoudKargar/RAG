from pydantic_settings import BaseSettings
from typing import Literal, Optional
from functools import lru_cache

class Settings(BaseSettings):
    # AI Provider Settings
    PROVIDER: Literal["openai", "avalai", "openrouter"] = "openai"
    
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
    OPENROUTER_CHAT_MODEL: str = "nvidia/nemotron-3-nano-30b-a3b:free"
    OPENROUTER_EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
    
    # Local Embedding Settings / امبدینگ محلی با HuggingFace
    # وقتی EMBEDDING_PROVIDER=local باشد، مدل sentence-transformers زیر برای امبدینگ بارگذاری می‌شود
    EMBEDDING_PROVIDER: Optional[Literal["openai", "avalai", "openrouter", "local"]] = None
    LOCAL_EMBEDDING_MODEL: str = "xmanii/maux-gte-persian"
    LOCAL_EMBEDDING_API_URL: str = "http://127.0.0.1:8010/embed"
    
    # Vector Store Settings
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    RAG_SEARCH_LIMIT: int = 3
    
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