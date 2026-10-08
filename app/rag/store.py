"""Chroma vector store + hybrid BM25 retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import chromadb
from chromadb.api.types import PyEmbeddings
from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.models import get_model_provider
from app.rag.chunking import DocumentChunk, chunk_text, load_markdown_files


@dataclass
class Evidence:
    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    score: float
    rank: int

    def citation(self) -> str:
        return f"[{self.rank}] {self.title}/{self.section} ({self.source})"


class KnowledgeStore:
    COLLECTION = "enterprise_knowledge"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict[str, Any]] = []
        self._rebuild_bm25_from_collection()

    def ingest_directory(self, directory=None) -> int:
        directory = directory or self.settings.knowledge_path
        docs = load_markdown_files(directory)
        chunks: list[DocumentChunk] = []
        for source, title, content in docs:
            chunks.extend(chunk_text(content, source=source, title=title))
        return self.upsert_chunks(chunks)

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        provider = get_model_provider()
        texts = [c.text for c in chunks]
        embeddings: PyEmbeddings = [cast(Sequence[float], vec) for vec in provider.embed(texts)]
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],
        )
        self._rebuild_bm25_from_collection()
        return len(chunks)

    def _rebuild_bm25_from_collection(self) -> None:
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        self._corpus = []
        tokenized: list[list[str]] = []
        for i, doc_id in enumerate(ids):
            text = docs[i] or ""
            meta = metas[i] or {}
            self._corpus.append(
                {
                    "chunk_id": doc_id,
                    "text": text,
                    "source": meta.get("source", ""),
                    "title": meta.get("title", ""),
                    "section": meta.get("section", ""),
                }
            )
            tokenized.append(_tokenize(text))
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def hybrid_search(self, query: str, top_k: int | None = None) -> list[Evidence]:
        top_k = top_k or self.settings.retrieval_top_k
        if not self._corpus:
            return []

        vector_hits = self._vector_search(query, top_k=top_k * 2)
        bm25_hits = self._bm25_search(query, top_k=top_k * 2)
        fused = _rrf_fuse(vector_hits, bm25_hits)
        reranked = _score_rerank(query, fused)[: self.settings.rerank_top_n]
        if reranked:
            max_score = max(float(x.get("score") or 0.0) for x in reranked) or 1.0
            for item in reranked:
                item["score"] = float(item.get("score") or 0.0) / max_score
        evidence: list[Evidence] = []
        for i, item in enumerate(reranked, start=1):
            evidence.append(
                Evidence(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    source=item["source"],
                    title=item["title"],
                    section=item["section"],
                    score=float(item["score"]),
                    rank=i,
                )
            )
        return evidence

    def _vector_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        provider = get_model_provider()
        emb = provider.embed([query])[0]
        result = self._collection.query(
            query_embeddings=cast(PyEmbeddings, [cast(Sequence[float], emb)]),
            n_results=min(top_k, max(1, len(self._corpus))),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for i, doc_id in enumerate(ids):
            dist = float(dists[i]) if i < len(dists) else 1.0
            score = 1.0 / (1.0 + dist)
            meta = metas[i] or {}
            hits.append(
                {
                    "chunk_id": doc_id,
                    "text": docs[i] or "",
                    "source": meta.get("source", ""),
                    "title": meta.get("title", ""),
                    "section": meta.get("section", ""),
                    "score": score,
                }
            )
        return hits

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        hits: list[dict[str, Any]] = []
        max_score = max((s for _, s in ranked), default=1.0) or 1.0
        for idx, score in ranked:
            item = dict(self._corpus[idx])
            item["score"] = float(score) / float(max_score)
            hits.append(item)
        return hits


def _tokenize(text: str) -> list[str]:
    import re

    text = text.lower()
    tokens: list[str] = []
    # latin/digit words
    tokens.extend(re.findall(r"[a-z0-9_]+", text))
    # CJK chars + bigrams for better BM25 on Chinese
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens or ["empty"]


def _rrf_fuse(
    list_a: list[dict[str, Any]],
    list_b: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    payload: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(list_a):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        payload[cid] = item
    for rank, item in enumerate(list_b):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        payload[cid] = item
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused: list[dict[str, Any]] = []
    for cid, score in ordered:
        item = dict(payload[cid])
        item["score"] = score
        fused.append(item)
    return fused


def _score_rerank(query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q_tokens = set(_tokenize(query))
    reranked: list[dict[str, Any]] = []
    for item in items:
        t_tokens = set(_tokenize(item["text"]))
        overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens))
        # Prefer lexical match strongly for mock embeddings
        score = 0.35 * float(item.get("score", 0.0)) + 0.65 * overlap
        new_item = dict(item)
        new_item["score"] = score
        reranked.append(new_item)
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked


_store: KnowledgeStore | None = None


def get_knowledge_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store
