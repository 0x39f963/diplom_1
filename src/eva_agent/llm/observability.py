"""LangFuse-трейсинг (опционально).

Если в .env заданы LANGFUSE_PUBLIC_KEY/SECRET_KEY - возвращаем OpenAI-класс из
`langfuse.openai` (drop-in: автоматически шлет в LangFuse cost/tokens/latency каждого вызова).
Иначе - обычный openai.OpenAI (трейсинг выключен, без предупреждений).
"""

from __future__ import annotations

import os

from eva_agent.settings import settings


def langfuse_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def make_openai_class() -> type:
    if langfuse_enabled():
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
        try:
            from langfuse.openai import OpenAI as TracedOpenAI

            return TracedOpenAI
        except ImportError:
            pass
    from openai import OpenAI

    return OpenAI
