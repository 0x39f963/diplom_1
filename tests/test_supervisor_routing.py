from __future__ import annotations

import json
from typing import Any

import eva_agent.nodes.agents as agents_module
from eva_agent.graph import _route_after_supervisor
from eva_agent.llm.base import LLMResponse
from eva_agent.state import AgentState


class _StaticClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.text = json.dumps(payload, ensure_ascii=False)

    def invoke(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        return LLMResponse(text=self.text, model="fake", backend="fake")


def _patch_supervisor_client(monkeypatch: Any, payload: dict[str, Any]) -> None:
    client = _StaticClient(payload)

    def fake_get_client(role: str) -> _StaticClient:
        assert role == "reasoning"
        return client

    monkeypatch.setattr(agents_module, "get_client", fake_get_client)


def test_supervisor_overrides_clarification_for_domain_signal(monkeypatch: Any) -> None:
    _patch_supervisor_client(
        monkeypatch,
        {
            "kind": "need_clarification",
            "confidence": 0.4,
            "rationale": "слишком общая формулировка",
            "needed_inputs": ["договор"],
        },
    )

    result = agents_module.supervisor(
        AgentState(user_input_raw="Покажи список договоров, которые пока не доведены до конца.")
    )

    assert result["intent"].kind == "mixed_diagnostic"
    assert result["intent"].needed_inputs == ["договор"]
    assert "high-recall override" in result["intent"].rationale
    assert _route_after_supervisor(
        AgentState(user_input_raw="x", intent=result["intent"])
    ) == "domain_selector"


def test_supervisor_keeps_out_of_scope_without_domain_signal(monkeypatch: Any) -> None:
    _patch_supervisor_client(
        monkeypatch,
        {
            "kind": "out_of_scope",
            "confidence": 0.8,
            "rationale": "не по теме",
            "needed_inputs": [],
        },
    )

    result = agents_module.supervisor(AgentState(user_input_raw="Как сварить кофе?"))

    assert result["intent"].kind == "out_of_scope"
    assert result["intent"].rationale == "не по теме"
