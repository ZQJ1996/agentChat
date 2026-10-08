"""OpenAI-compatible chat & embedding provider."""

from __future__ import annotations

from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.config import Settings
from app.models.base import ChatMessage, ChatResult, ModelProvider


def _to_openai_messages(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    return [
        cast(ChatCompletionMessageParam, cast(object, {"role": m.role, "content": m.content}))
        for m in messages
    ]


class OpenAIModelProvider(ModelProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.chat_model = settings.openai_chat_model
        self.embed_model = settings.openai_embed_model

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=_to_openai_messages(messages),
            temperature=temperature if temperature is not None else self.settings.temperature,
            max_tokens=max_tokens or self.settings.max_tokens_per_request,
        )
        choice = resp.choices[0].message.content or ""
        usage = resp.usage
        return ChatResult(
            content=choice,
            model=self.chat_model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw={"id": resp.id},
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        stream = self.client.chat.completions.create(
            model=self.chat_model,
            messages=_to_openai_messages(messages),
            temperature=temperature if temperature is not None else self.settings.temperature,
            max_tokens=max_tokens or self.settings.max_tokens_per_request,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [item.embedding for item in resp.data]
