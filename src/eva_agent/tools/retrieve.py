"""HTTP-клиенты к сервису поиска по закону.

Сам поиск (эмбеддинги, reranker, pgvector) вынесен в отдельный сервис, чтобы не тянуть
тяжелые модели в приложение; агент обращается к нему по HTTP (адрес в RAG_API_BASE).
Контракт ответа: {"records": [{"content", "score", "citation", "metadata": {...}}]}.
Найденные куски помечаются недоверенными (spotlight) перед подачей в модель.
"""

from __future__ import annotations

from typing import Literal

import httpx

from eva_agent.settings import settings
from eva_agent.state import Chunk, RetrievalResult

_TRUST = ("primary", "secondary", "tertiary")


def _to_chunk(record: dict) -> Chunk:
    meta = record.get("metadata") or {}
    trust = meta.get("trust_level", "primary")
    if trust not in _TRUST:
        trust = "primary"
    return Chunk(
        text=record.get("content", ""),
        citation=record.get("citation") or meta.get("citation", "") or "",
        law_number=meta.get("law_number", "") or "",
        article=meta.get("article", "") or "",
        trust_level=trust,
        score=float(record.get("score") or 0.0),
        source_url=meta.get("source_url", "") or "",
    )


def _retrieve(query: str, *, in_force_only: bool = True) -> list[Chunk]:
    response = httpx.post(
        f"{settings.rag_api_base}/retrieval",
        json={"query": query, "in_force_only": in_force_only},
        timeout=120.0,
    )
    response.raise_for_status()
    records = response.json().get("records", [])
    return [_to_chunk(r) for r in records]


def retrieve_legal(query: str) -> RetrievalResult:
    """Статьи 38-ФЗ по запросу. Гибридный поиск: dense + keyword + rerank + фильтр действующих редакций."""
    return RetrievalResult(query=query, collection="legal", chunks=_retrieve(query))


def retrieve_howto(query: str, *, collection: Literal["legal", "howto"] = "howto") -> RetrievalResult:
    """Как сделать в системе. Пока тот же эндпоинт сервиса поиска; отдельная коллекция - следующий шаг."""
    return RetrievalResult(query=query, collection=collection, chunks=_retrieve(query))
