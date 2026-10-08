"""Ollama local model provider via HTTP API."""

from __future__ import annotations

import httpx

from app.config import Settings
from app.models.base import ChatMessage, ChatResult, ModelProvider


class OllamaModelProvider(ModelProvider):
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.chat_model = settings.ollama_chat_model
        self.embed_model = settings.ollama_embed_model

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        payload = {
            "model": self.chat_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature
                if temperature is not None
                else self.settings.temperature,
                "num_predict": max_tokens or self.settings.max_tokens_per_request,
            },
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = (data.get("message") or {}).get("content", "")
        return ChatResult(
            content=content,
            model=self.chat_model,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            raw=data,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        payload = {
            "model": self.chat_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature
                if temperature is not None
                else self.settings.temperature,
                "num_predict": max_tokens or self.settings.max_tokens_per_request,
            },
        }
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    import json

                    data = json.loads(line)
                    msg = data.get("message") or {}
                    chunk = msg.get("content")
                    if chunk:
                        yield chunk

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=120.0) as client:
            for text in texts:
                resp = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                )
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
        return vectors
