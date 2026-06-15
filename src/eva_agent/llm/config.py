"""Выбор LLM-клиента под роль узла графа.

Какая модель и backend на каждой роли - настраивается через .env, без правок кода.
"""

from __future__ import annotations

from eva_agent import metrics
from eva_agent.llm.base import LLMClient, LLMResponse
from eva_agent.llm.ollama_local import OllamaLocalClient
from eva_agent.llm.openrouter import OpenRouterClient
from eva_agent.settings import Role, settings


class _TrackedClient(LLMClient):
    """Обертка: на каждый invoke пишет usage в run-учет (metrics)."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.backend = inner.backend
        self.model = inner.model

    def invoke(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        response = self._inner.invoke(system, user, temperature=temperature, json_mode=json_mode)
        metrics.record(response.usage)
        return response


def get_client(role: Role) -> LLMClient:
    backend = settings.role_backend(role)
    model = settings.role_model(role)
    if backend == "openrouter":
        inner: LLMClient = OpenRouterClient(
            model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.llm_call_timeout_sec,
            max_retries=settings.llm_provider_max_retries,
            provider_only=settings.openrouter_provider_only,
            thinking=settings.qwen_thinking_force_on,
        )
    else:
        inner = OllamaLocalClient(
            model,
            base_url=settings.local_llm_base_url,
            keep_alive=settings.ollama_request_keep_alive,
            timeout=settings.llm_call_timeout_sec,
        )
    return _TrackedClient(inner)
