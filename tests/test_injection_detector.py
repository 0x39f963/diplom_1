from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from eva_agent.security import injection_detector


class _FakeClient:
    def __init__(self) -> None:
        self.system = ""
        self.user = ""

    def invoke(self, system: str, user: str, **kwargs: Any) -> SimpleNamespace:
        self.system = system
        self.user = user
        return SimpleNamespace(
            text='{"decision":"allow","risk_score":0,"categories":[],"reason":"ok"}'
        )


def test_detect_injection_adds_domain_context(monkeypatch) -> None:
    client = _FakeClient()
    monkeypatch.setattr(injection_detector, "get_client", lambda role: client)

    verdict = injection_detector.detect_injection("Открой CP-1", domain_signals=["CP-1"])

    assert verdict.decision == "allow"
    assert "CT-/CP-/DOC-/CR-/PL-" in client.system
    assert "ДОМЕННЫЙ КОНТЕКСТ" in client.user
    assert "CP-1" in client.user
