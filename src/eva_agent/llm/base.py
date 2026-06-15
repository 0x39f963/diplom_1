"""Единый контракт LLM-клиентов (ТЗ-4).

Все бэкенды (OpenRouter, локальный Ollama) возвращают один `LLMResponse`, поэтому
вызывающий код не знает, облако это или локальная карта.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class LLMConfigError(RuntimeError):
    """Ошибка конфигурации (пустой ключ и т.п.) - fail fast, не зависание."""


@dataclass
class LLMResponse:
    text: str
    model: str
    backend: str
    walltime_sec: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)   # prompt/completion/total_tokens, cost_usd
    retry_log: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    backend: str
    model: str

    @abstractmethod
    def invoke(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Синхронный вызов модели. `json_mode=True` просит строгий JSON-ответ."""
