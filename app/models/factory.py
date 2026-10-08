"""Factory for model providers."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.models.base import ModelProvider
from app.models.mock_provider import MockModelProvider

_provider: ModelProvider | None = None


def create_provider(settings: Settings) -> ModelProvider:
    provider = settings.model_provider
    if provider == "mock":
        return MockModelProvider()
    if provider == "openai":
        from app.models.openai_provider import OpenAIModelProvider

        return OpenAIModelProvider(settings)
    if provider == "ollama":
        from app.models.ollama_provider import OllamaModelProvider

        return OllamaModelProvider(settings)
    raise ValueError(f"Unsupported MODEL_PROVIDER: {provider}")


def get_model_provider() -> ModelProvider:
    global _provider
    if _provider is None:
        _provider = create_provider(get_settings())
    return _provider


def reset_model_provider() -> None:
    global _provider
    _provider = None
