from __future__ import annotations

import eva_agent.tracing as tracing


def test_log_span_event_noop_without_langfuse(monkeypatch) -> None:
    monkeypatch.setattr(tracing, "langfuse_enabled", lambda: False)

    tracing.log_span_event({"guard": {"decision": "allow"}})
