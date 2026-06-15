"""Тесты идут в детерминированном мок-режиме независимо от .env (который может смотреть на реальный backend)."""

from __future__ import annotations

import pytest

from eva_agent.settings import settings


@pytest.fixture(autouse=True)
def _force_mock_mode():
    api_base, mock_mode = settings.eva_api_base, settings.eva_mock_mode
    lf_pub, lf_sec = settings.langfuse_public_key, settings.langfuse_secret_key
    settings.eva_api_base = "mock"
    settings.eva_mock_mode = "fixtures"
    # Тесты офлайн: выключаем LangFuse, чтобы мок-прогон не дергал сеть/трейсинг.
    settings.langfuse_public_key = ""
    settings.langfuse_secret_key = ""
    yield
    settings.eva_api_base, settings.eva_mock_mode = api_base, mock_mode
    settings.langfuse_public_key, settings.langfuse_secret_key = lf_pub, lf_sec
