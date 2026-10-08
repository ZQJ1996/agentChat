"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_provider: Literal["mock", "openai", "ollama"] = "mock"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    max_graph_steps: int = 12
    max_tool_retries: int = 1
    retrieval_top_k: int = 5
    rerank_top_n: int = 3
    evidence_min_score: float = 0.08
    max_tokens_per_request: int = 4096
    temperature: float = 0.2

    knowledge_dir: str = "data/knowledge"
    chroma_dir: str = "data/chroma"
    sqlite_path: str = "data/sqlite/agent.db"
    golden_set_path: str = "data/eval/golden_set.jsonl"
    eval_report_dir: str = "reports"

    @property
    def knowledge_path(self) -> Path:
        return PROJECT_ROOT / self.knowledge_dir

    @property
    def chroma_path(self) -> Path:
        return PROJECT_ROOT / self.chroma_dir

    @property
    def sqlite_file(self) -> Path:
        return PROJECT_ROOT / self.sqlite_path

    @property
    def golden_set_file(self) -> Path:
        return PROJECT_ROOT / self.golden_set_path

    @property
    def eval_report_path(self) -> Path:
        return PROJECT_ROOT / self.eval_report_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
