"""Локальный Ollama-коннектор (портировано из соседнего проекта).

Native `/api/chat` (стабильный контент vs `/v1`, который у части моделей отдает только
reasoning). `think:false`, `keep_alive` (удержание модели в VRAM), per-model lock.

Локальные вызовы не перехватывает langfuse.openai (это работает только для облачного OpenAI SDK),
поэтому при включенном LangFuse создаем generation вручную - чтобы у локальной модели в трейсе были
вход, выход и токены, как у облачных моделей.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from eva_agent.llm.base import LLMClient, LLMResponse
from eva_agent.llm.observability import langfuse_enabled

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _model_lock(model: str) -> threading.Lock:
    """Сериализация вызовов на модель: не запускать два runner-load одновременно."""
    with _locks_guard:
        return _locks.setdefault(model, threading.Lock())


class OllamaLocalClient(LLMClient):
    backend = "local"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        keep_alive: str = "30m",
        timeout: int = 300,
    ) -> None:
        self.model = model
        self._keep_alive = keep_alive
        self._timeout = timeout
        # base_url может прийти как .../v1 (OpenAI-совместимый) - native-ручка без /v1.
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        self._chat_url = f"{root}/api/chat"

    def invoke(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "keep_alive": self._keep_alive,
            "options": {"temperature": 0.1 if temperature is None else temperature},
        }
        if schema is not None:
            payload["format"] = schema
        elif json_mode:
            payload["format"] = "json"

        started = time.monotonic()
        data = self._post(payload, system, user)
        return LLMResponse(
            text=(data.get("message") or {}).get("content", ""),
            model=self.model,
            backend=self.backend,
            walltime_sec=time.monotonic() - started,
            usage={
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
                "cost_usd": 0.0,
            },
            raw=data,
        )

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._chat_url, data=body, headers={"Content-Type": "application/json"}
        )
        with _model_lock(self.model), urllib.request.urlopen(
            request, timeout=self._timeout
        ) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return result

    def _post(self, payload: dict[str, Any], system: str, user: str) -> dict[str, Any]:
        """Вызов модели + ручной generation в LangFuse (если включен)."""
        if not langfuse_enabled():
            return self._call(payload)
        from langfuse import get_client

        client = get_client()
        with client.start_as_current_observation(
            as_type="generation",
            name="ollama-generation",
            model=self.model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        ):
            data = self._call(payload)
            client.update_current_generation(
                output=(data.get("message") or {}).get("content", ""),
                usage_details={
                    "input": int(data.get("prompt_eval_count") or 0),
                    "output": int(data.get("eval_count") or 0),
                },
                cost_details={"input": 0.0, "output": 0.0, "total": 0.0},
            )
            return data
