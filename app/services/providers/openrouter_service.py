from openai import OpenAI
from app.config.settings import settings
from app.services.base import BaseAIProvider
from typing import List, AsyncGenerator, Any


class OpenRouterProvider(BaseAIProvider):
    """Provider for the OpenRouter API (OpenAI-compatible)."""

    def __init__(self):
        """Initialize OpenRouter provider using the OpenAI client with OpenRouter base URL."""
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required when using OpenRouter provider")

        self.client = OpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "https://github.com/MauxPlatform/Maux-RAG-API",
                "X-Title": "Maux RAG API",
            },
        )

    def create_embedding(self, text: str) -> List[float]:
        """Create embeddings using OpenRouter's API."""
        result = self.client.embeddings.create(
            input=text,
            model=settings.OPENROUTER_EMBEDDING_MODEL,
        )
        return result.data[0].embedding

    def create_chat_completion(self, messages: list, model: str) -> Any:
        """Create chat completion using OpenRouter's API."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response

    def create_chat_completion_stream(self, messages: list, model: str) -> AsyncGenerator[Any, None]:
        """Create streaming chat completion using OpenRouter's API."""
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        return stream

