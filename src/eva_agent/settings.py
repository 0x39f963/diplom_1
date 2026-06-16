"""Конфигурация eva-agent-lab из окружения (.env).

Единая точка чтения env. Здесь же выбираю модель под каждую роль узла,
чтобы переключать модели без правок кода.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Backend = Literal["openrouter", "local", "claude_cli", "codex_cli"]
Effort = Literal["low", "medium", "high"]
Role = Literal["reasoning", "default", "guard"]
MockMode = Literal["fixtures", "random"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_provider_only: str = ""
    qwen_thinking_force_on: bool = False

    # Локальный Ollama
    local_llm_base_url: str = "http://host.docker.internal:11434/v1"
    host_local_llm_base_url: str = "http://127.0.0.1:11434/v1"
    local_llm_api_key: str = "not-needed"
    ollama_keep_alive: str = "30m"
    ollama_request_keep_alive: str = "30m"

    # Роли: backend и модель на каждую роль
    llm_backend_reasoning: Backend = "openrouter"
    llm_model_reasoning: str = "qwen/qwen3.5-9b"
    llm_backend_default: Backend = "local"
    llm_model_default: str = "qwen3.5:9b"
    llm_backend_guard: Backend = "openrouter"
    llm_model_guard: str = "qwen/qwen3.5-9b"
    llm_effort_reasoning: Effort = "medium"
    llm_effort_default: Effort = "medium"
    llm_effort_guard: Effort = "medium"
    llm_call_timeout_sec: int = 300
    llm_provider_max_retries: int = 2

    # CLI agents
    eva_cli_claude_bin: str = "claude"
    eva_cli_codex_bin: str = "codex"

    # Поиск по закону работает отдельным сервисом (адрес в RAG_API_BASE), агент ходит по HTTP
    rag_api_base: str = "http://localhost:8077"

    # Внутреннюю систему в этот репозиторий выложить нельзя. Если ее нет (mock или сервис недоступен),
    # данные подменяются встроенными заглушками.
    eva_mock_mode: MockMode = "fixtures"
    eva_api_base: str = "mock"
    eva_login: str = "demo@example.local"
    eva_password: str = ""

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Граф
    max_loops: int = Field(default=2, ge=0, le=5)

    def role_backend(self, role: Role) -> Backend:
        return getattr(self, f"llm_backend_{role}")

    def role_model(self, role: Role) -> str:
        return getattr(self, f"llm_model_{role}")

    def role_effort(self, role: Role) -> Effort:
        return getattr(self, f"llm_effort_{role}")


settings = Settings()
