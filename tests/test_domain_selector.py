from __future__ import annotations

import json
from typing import Any

import eva_agent.nodes.agents as agents_module
import eva_agent.nodes.domain_nodes as domain_nodes
import eva_agent.nodes.frame_parser as frame_parser_module
import eva_agent.nodes.guards as guards_module
from eva_agent.graph import build_graph
from eva_agent.llm.base import LLMResponse
from eva_agent.security.verdict import GuardVerdict
from eva_agent.settings import settings
from eva_agent.state import AgentState, Intent, RetrievalResult


class _StaticClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "json_mode": json_mode,
                "schema": schema,
            }
        )
        return LLMResponse(text=self.text, model="fake", backend="fake")


class _AgentClient:
    def invoke(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del schema
        if "оркестратор" in system:
            payload = {
                "kind": "mixed_diagnostic",
                "confidence": 0.95,
                "rationale": "нужно состояние данных",
                "needed_inputs": [],
            }
            return LLMResponse(text=json.dumps(payload, ensure_ascii=False), model="fake", backend="fake")
        if "критик" in system:
            return LLMResponse(
                text='{"decision":"accept","target":null,"reason":""}',
                model="fake",
                backend="fake",
            )
        return LLMResponse(text="Данные собраны.", model="fake", backend="fake")


def test_domain_selector_returns_slice_and_checklist(monkeypatch: Any) -> None:
    client = _StaticClient('{"entities":["ContractParty","Counterparty","Missing"]}')

    def fake_get_client(role: str) -> _StaticClient:
        assert role == "domain"
        return client

    monkeypatch.setattr(domain_nodes, "get_client", fake_get_client)
    state = AgentState(
        user_input_raw="кто заказчик по договору CT-1",
        intent=Intent(kind="mixed_diagnostic", confidence=0.9),
    )

    result = domain_nodes.domain_selector(state)

    domain_slice = result["domain_slice"]
    checklist = result["checklist"]
    assert client.calls and client.calls[0]["json_mode"] is True
    assert domain_slice.entities == ["ContractParty", "Counterparty", "Contract"]
    assert domain_slice.tools
    assert checklist.intent == "mixed_diagnostic"
    assert checklist.entities == domain_slice.entities
    assert checklist.access == domain_slice.tools
    assert checklist.resolution == "proceed"
    assert any(count.entity == "Contract" and count.ref_count == 1 for count in checklist.cardinality)


def test_domain_selector_fallback_on_broken_llm_response(monkeypatch: Any) -> None:
    client = _StaticClient("{not-json")

    def fake_get_client(role: str) -> _StaticClient:
        assert role == "domain"
        return client

    monkeypatch.setattr(domain_nodes, "get_client", fake_get_client)
    state = AgentState(
        user_input_raw="собери обзор данных",
        intent=Intent(kind="mixed_diagnostic", confidence=0.9),
    )

    result = domain_nodes.domain_selector(state)

    assert len(client.calls) == 1
    assert len(result["domain_slice"].entities) == 7
    assert result["checklist"].entities == result["domain_slice"].entities
    assert result["checklist"].access


def test_supervisor_prompt_keeps_data_question_as_mixed(monkeypatch: Any) -> None:
    client = _StaticClient(
        json.dumps(
            {
                "kind": "mixed_diagnostic",
                "confidence": 0.9,
                "rationale": "нужно посмотреть договор",
                "needed_inputs": [],
            },
            ensure_ascii=False,
        )
    )

    def fake_get_client(role: str) -> _StaticClient:
        assert role == "reasoning"
        return client

    monkeypatch.setattr(agents_module, "get_client", fake_get_client)

    result = agents_module.supervisor(AgentState(user_input_raw="какой статус договора CT-1?"))

    assert result["intent"].kind == "mixed_diagnostic"
    assert "Любой такой запрос" in client.calls[0]["system"]
    assert "какой статус договора CT-1" in client.calls[0]["system"]


def test_graph_mixed_route_runs_frame_parser_compiler_and_data_gather(monkeypatch: Any) -> None:
    monkeypatch.setattr(settings, "planner_use_protocol_compiler", True)
    frame_payload = {
        "operation": "read",
        "target": "ContractParty",
        "relation": "parties",
        "fields": [],
        "filters": {"date_hint": "none", "status": []},
        "cardinality": "one",
        "selector": {"contract_id": "CT-1", "role": "customer"},
        "output": "card",
        "subtasks": [],
        "needs_clarification": False,
        "clarify_reason": "",
        "confidence": 0.9,
    }
    domain_client = _StaticClient('{"entities":["Contract","ContractParty","Counterparty"]}')
    frame_client = _StaticClient(json.dumps(frame_payload, ensure_ascii=False))

    def fake_detect_injection(user_input: str, untrusted_data: str = "") -> GuardVerdict:
        assert user_input
        assert untrusted_data == ""
        return GuardVerdict(decision="allow", risk_score=0.0, reason="ok")

    def fake_agent_get_client(role: str) -> _AgentClient:
        assert role in {"reasoning", "default"}
        return _AgentClient()

    def fake_domain_get_client(role: str) -> _StaticClient:
        assert role == "domain"
        return domain_client

    def fake_frame_get_client(role: str) -> _StaticClient:
        assert role == "domain"
        return frame_client

    def fake_retrieve_howto(query: str) -> RetrievalResult:
        return RetrievalResult(query=query, collection="howto", chunks=[])

    monkeypatch.setattr(guards_module, "detect_injection", fake_detect_injection)
    monkeypatch.setattr(agents_module, "get_client", fake_agent_get_client)
    monkeypatch.setattr(domain_nodes, "get_client", fake_domain_get_client)
    monkeypatch.setattr(frame_parser_module, "get_client", fake_frame_get_client)
    monkeypatch.setattr(frame_parser_module, "retrieve_examples", lambda query, k=5: [])
    monkeypatch.setattr(agents_module, "retrieve_howto", fake_retrieve_howto)

    result = build_graph().invoke({"user_input_raw": "кто заказчик по договору CT-1"})

    assert result["intent"].kind == "mixed_diagnostic"
    assert result["frame"].target == "ContractParty"
    assert result["domain_slice"].entities == ["ContractParty", "Contract", "Counterparty"]
    assert result["checklist"].resolution == "proceed"
    assert result["todo_plan"].strategy.startswith("compiled:")
    assert result["todo_plan"].status == "answered"
    assert [finding.tool for finding in result["api_findings"]] == [
        "eva_get_contract_parties",
        "eva_get_counterparty",
    ]
    assert frame_client.calls[0]["schema"]["$defs"]["PlanningFrame"]["properties"]["target"]["type"] == "string"
