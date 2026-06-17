"""OpenRouter-коннектор (портировано из соседнего проекта).

OpenAI SDK с выключенным SDK-ретраем (`max_retries=0`) + свой явный retry-loop;
provider-pin; глушение qwen3-thinking (иначе 11-54 с/вызов); извлечение usage/cost.
"""

from __future__ import annotations

import time
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError

from eva_agent.llm.base import LLMClient, LLMConfigError, LLMResponse
from eva_agent.llm.observability import make_openai_class

# LangFuse-трейсинг включается через .env (LANGFUSE_*); иначе обычный OpenAI-клиент.
OpenAI = make_openai_class()

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_BACKOFF_CAP_SEC = 30.0


class OpenRouterClient(LLMClient):
    backend = "openrouter"

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str,
        timeout: int = 300,
        max_retries: int = 2,
        provider_only: str = "",
        thinking: bool = False,
    ) -> None:
        if not api_key:
            raise LLMConfigError("OPENROUTER_API_KEY пуст - заполни .env")
        self.model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._provider_only = provider_only
        self._thinking = thinking
        # SDK-ретрай выключен намеренно: скрытые ретраи съедают walltime.
        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)

    def _extra_body(self) -> dict[str, Any]:
        # usage.include=true - OpenRouter возвращает фактическую стоимость вызова в usage.cost
        # (нужно для cost-метрик eval; без него часть моделей cost не отдает).
        extra: dict[str, Any] = {"usage": {"include": True}}
        if self._provider_only:
            extra["provider"] = {"only": [self._provider_only], "allow_fallbacks": False}
        # Глушим qwen3-thinking по умолчанию (вернуть - thinking=True).
        if "qwen3" in self.model.lower() and not self._thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
            extra["reasoning"] = {"enabled": False}
        return extra

    def invoke(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1 if temperature is None else temperature,
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "frame", "schema": schema, "strict": True},
            }
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs["extra_body"] = self._extra_body()

        retry_log: list[str] = []
        started = time.monotonic()
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return self._to_response(resp, time.monotonic() - started, retry_log)
            except (APIConnectionError, APITimeoutError) as exc:
                retry_log.append(f"attempt {attempt}: network {type(exc).__name__}")
            except APIStatusError as exc:
                if exc.status_code not in _RETRYABLE_STATUS or attempt == self._max_retries:
                    raise
                retry_log.append(f"attempt {attempt}: status {exc.status_code}")
                self._sleep(getattr(exc, "response", None), attempt)
                continue
            if attempt == self._max_retries:
                raise
            self._sleep(None, attempt)
        raise RuntimeError("unreachable")

    @staticmethod
    def _sleep(response: Any, attempt: int) -> None:
        retry_after = 0.0
        if response is not None:
            raw = response.headers.get("retry-after") if hasattr(response, "headers") else None
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = 0.0
        delay = retry_after or min(_BACKOFF_CAP_SEC, 1.5 * (2**attempt))
        time.sleep(delay)

    def _to_response(self, resp: Any, walltime: float, retry_log: list[str]) -> LLMResponse:
        data = resp.model_dump()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        return LLMResponse(
            text=text,
            model=self.model,
            backend=self.backend,
            walltime_sec=walltime,
            usage=_extract_usage(data),
            retry_log=retry_log,
            raw=data,
        )


def _extract_usage(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": usage.get("cost") or (data.get("usage_accounting") or {}).get("cost"),
        "provider": data.get("provider"),
    }
