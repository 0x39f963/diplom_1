"""Прокладка к внешнему backend API (read-only) с авто-fallback на мок.

EVA_API_BASE=mock        -> реальный сервис не трогаем, все через мок (data.py).
EVA_API_BASE=http://...  -> логинимся dev-сессией и читаем реальные данные; если сервис
                            недоступен / логин не прошел / эндпоинт упал - возвращаем None,
                            и вызывающий tool сам падает на мок. Строго read-only (только GET).
"""

from __future__ import annotations

from typing import Any

import httpx

from eva_agent.settings import settings


class EvaReadClient:
    def __init__(self) -> None:
        self._client: httpx.Client | None = None
        self._login_failed = False

    @property
    def real_mode(self) -> bool:
        return settings.eva_api_base.startswith("http")

    def _session(self) -> httpx.Client | None:
        if not self.real_mode or self._login_failed:
            return None
        if self._client is not None:
            return self._client
        try:
            client = httpx.Client(base_url=settings.eva_api_base, timeout=10.0)
            response = client.post(
                "/api/auth/login",
                json={"email": settings.eva_login, "password": settings.eva_password, "remember": True},
            )
            if response.status_code == 200:
                # cookie eva_session помечена Secure - по http httpx ее не шлет, ставим явно.
                token = client.cookies.get("eva_session")
                if token:
                    client.headers["Cookie"] = f"eva_session={token}"
                self._client = client
                return client
            client.close()
        except httpx.HTTPError:
            pass
        self._login_failed = True
        return None

    def get(self, path: str, params: dict | None = None) -> Any | None:
        client = self._session()
        if client is None:
            return None
        try:
            response = client.get(path, params=params)
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError:
            pass
        return None


_client = EvaReadClient()


def real_get(path: str, params: dict | None = None) -> Any | None:
    """GET к внешнему backend, либо None (значит - fallback на мок)."""
    return _client.get(path, params)


def real_available() -> bool:
    return _client._session() is not None
