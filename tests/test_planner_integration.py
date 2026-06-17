from __future__ import annotations

import json
from typing import Any

import eva_agent.nodes.agents as agents_module
from eva_agent.domain.checklist import EntityCount, PlanningChecklist
from eva_agent.domain.plan import PlanStep, TodoItem, TodoPlan
from eva_agent.domain.slice import DomainSlice
from eva_agent.graph import _route_after_data_gather
from eva_agent.llm.base import LLMResponse
from eva_agent.planner import build as planner_build
from eva_agent.planner.trace import trace_plan
from eva_agent.state import AgentState, ApiFinding, CriticVerdict, Intent


class _FakePlannerClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def invoke(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls += 1
        assert "КАТАЛОГ ДОСТУПНЫХ TODO" in system
        assert user
        assert temperature == 0.0
        assert json_mode is True
        return LLMResponse(text=self.text, model="fake", backend="fake")


def _planner_payload() -> dict[str, Any]:
    return {
        "goal": "кто заказчик по договору CT-1",
        "protocol_id": "party_lookup",
        "strategy": "получить стороны договора",
        "items": [
            {
                "id": "resolve_party_role",
                "type": "blocking",
                "order": 1,
                "inputs": {"contract_id": "CT-1", "role": "customer"},
                "tool_calls": [
                    {
                        "order": 1,
                        "tool": "eva_get_contract_parties",
                        "args": {"contract_id": "CT-1"},
                        "date_hint": "none",
                        "status_hint": "none",
                        "reason": "получить стороны",
                    }
                ],
            }
        ],
        "status": "in_progress",
        "confidence": 0.9,
        "clarify_question": "",
    }


def _executable_plan() -> TodoPlan:
    return TodoPlan(
        goal="кто заказчик по договору CT-1",
        protocol_id="party_lookup",
        strategy="получить стороны договора",
        status="in_progress",
        confidence=0.9,
        items=[
            TodoItem(
                id="resolve_party_role",
                order=1,
                inputs={"contract_id": "CT-1", "role": "customer"},
                tool_calls=[
                    PlanStep(
                        order=1,
                        tool="eva_get_contract_parties",
                        args={"contract_id": "CT-1"},
                    )
                ],
            )
        ],
    )


def test_data_gather_builds_and_executes_planner_plan(monkeypatch: Any) -> None:
    client = _FakePlannerClient(json.dumps(_planner_payload(), ensure_ascii=False))

    def fake_get_client(role: str) -> _FakePlannerClient:
        assert role == "planner"
        return client

    monkeypatch.setattr(planner_build, "get_client", fake_get_client)
    state = AgentState(
        user_input_raw="кто заказчик по договору CT-1",
        intent=Intent(kind="mixed_diagnostic", confidence=0.9),
        checklist=PlanningChecklist(
            intent="mixed_diagnostic",
            entities=["ContractParty"],
            cardinality=[EntityCount(entity="ContractParty", ref_count=0)],
            access=["eva_get_contract_parties"],
        ),
    )

    result = agents_module.data_gather(state)

    assert client.calls == 1
    assert "intent" not in result
    assert result["plan_attempts"] == 0
    assert isinstance(result["todo_plan"], TodoPlan)
    assert result["todo_plan"].status == "answered"
    assert [finding.tool for finding in result["api_findings"]] == ["eva_get_contract_parties"]
    assert result["checklist"].resolution == "proceed"
    assert result["checklist"].cardinality[0].result_count == 2


def test_data_gather_routes_awaiting_clarification_to_intent(monkeypatch: Any) -> None:
    plan = TodoPlan(
        goal="нужны данные",
        status="awaiting_clarification",
        confidence=0.2,
        clarify_question="Уточните договор.",
    )

    def fake_build_plan(
        query: str,
        *,
        prior_meaning: str = "",
        domain_slice: DomainSlice | None = None,
        intent_kind: str | None = None,
    ) -> TodoPlan:
        assert query
        assert prior_meaning == ""
        assert domain_slice is None
        assert intent_kind == "mixed_diagnostic"
        return plan

    monkeypatch.setattr(agents_module, "build_plan", fake_build_plan)
    state = AgentState(
        user_input_raw="кто заказчик",
        intent=Intent(kind="mixed_diagnostic", confidence=0.8),
        checklist=PlanningChecklist(
            intent="mixed_diagnostic",
            entities=["Contract"],
            cardinality=[EntityCount(entity="Contract", ref_count=0)],
            access=["eva_get_contract"],
        ),
    )

    result = agents_module.data_gather(state)

    assert "api_findings" not in result
    assert result["intent"].kind == "need_clarification"
    assert result["intent"].needed_inputs == ["Уточните договор."]
    assert result["todo_plan"] is plan
    assert result["checklist"].resolution == "clarify"
    assert result["checklist"].clarify_reason == "Уточните договор."


def test_route_after_data_gather_uses_plan_or_intent() -> None:
    clarify_plan = TodoPlan(status="awaiting_clarification", clarify_question="Уточните договор.")
    assert _route_after_data_gather(AgentState(user_input_raw="x", todo_plan=clarify_plan)) == "clarify"

    clarify_intent = Intent(kind="need_clarification", confidence=0.5, needed_inputs=["договор"])
    assert _route_after_data_gather(AgentState(user_input_raw="x", intent=clarify_intent)) == "clarify"

    assert _route_after_data_gather(AgentState(user_input_raw="x")) == "interface_agent"


def test_data_gather_reuses_existing_plan_on_rework(monkeypatch: Any) -> None:
    calls = 0

    def fail_build_plan(
        query: str,
        *,
        prior_meaning: str = "",
        domain_slice: DomainSlice | None = None,
        intent_kind: str | None = None,
    ) -> TodoPlan:
        nonlocal calls
        calls += 1
        raise AssertionError((query, prior_meaning))

    monkeypatch.setattr(agents_module, "build_plan", fail_build_plan)
    finding = ApiFinding(
        tool="eva_get_contract_parties",
        args={"contract_id": "CT-1"},
        data={"parties": []},
    )
    plan = _executable_plan()
    plan.status = "answered"
    plan.items[0].status = "done"
    state = AgentState(
        user_input_raw="кто заказчик по договору CT-1",
        api_findings=[finding],
        todo_plan=plan,
        plan_attempts=agents_module.MAX_PLAN_REBUILDS,
        critic=CriticVerdict(decision="rework", target="data_gather", reason="need more data"),
    )

    result = agents_module.data_gather(state)

    assert calls == 0
    assert result["api_findings"] == [finding]
    assert result["plan_attempts"] == agents_module.MAX_PLAN_REBUILDS
    assert result["plan_attempts"] <= agents_module.MAX_PLAN_REBUILDS


def test_data_gather_passes_slice_relations_to_executor(monkeypatch: Any) -> None:
    plan = _executable_plan()
    relations = [
        {
            "source_entity": "ContractParty",
            "target_entity": "Counterparty",
            "source_tool": "eva_get_contract_parties",
            "source_path": "parties[].counterparty_id",
            "target_tool": "eva_get_counterparty",
            "target_arg": "counterparty_id",
            "cardinality": "many",
        }
    ]
    domain_slice = DomainSlice.model_validate(
        {
            "entities": ["ContractParty", "Counterparty"],
            "relations": relations,
            "tools": ["eva_get_contract_parties", "eva_get_counterparty"],
        }
    )
    captured: dict[str, Any] = {}

    def fake_build_plan(
        query: str,
        *,
        prior_meaning: str = "",
        domain_slice: DomainSlice | None = None,
        intent_kind: str | None = None,
    ) -> TodoPlan:
        captured["domain_slice"] = domain_slice
        captured["intent_kind"] = intent_kind
        return plan

    def fake_execute_plan(
        incoming: TodoPlan,
        relations: list[Any] | None = None,
    ) -> tuple[list[ApiFinding], TodoPlan]:
        captured["relations"] = relations
        incoming.status = "answered"
        incoming.items[0].status = "done"
        return [], incoming

    monkeypatch.setattr(agents_module, "build_plan", fake_build_plan)
    monkeypatch.setattr(agents_module, "execute_plan", fake_execute_plan)
    state = AgentState(
        user_input_raw="кто заказчик по договору CT-1",
        intent=Intent(kind="mixed_diagnostic", confidence=0.8),
        domain_slice=domain_slice,
    )

    result = agents_module.data_gather(state)

    assert captured["domain_slice"] is domain_slice
    assert captured["intent_kind"] == "mixed_diagnostic"
    assert captured["relations"] == domain_slice.relations
    assert result["todo_plan"].status == "answered"


def test_trace_plan_payload_contains_audit_fields() -> None:
    plan = _executable_plan()
    plan.items[0].status = "done"
    plan.items[0].result_ref = "eva_get_contract_parties"
    finding = ApiFinding(tool="eva_get_contract_parties", args={"contract_id": "CT-1"}, data={})

    payload = trace_plan(plan, [finding], plan_reused=True, plan_attempts=1)

    assert payload["protocol_id"] == "party_lookup"
    assert payload["dialog_status"] == "in_progress"
    assert payload["findings_tools"] == ["eva_get_contract_parties"]
    assert payload["todos"][0]["id"] == "resolve_party_role"
    assert payload["todos"][0]["status"] == "done"
    assert payload["todos"][0]["tool"] == "eva_get_contract_parties"
    assert payload["plan_reused"] is True
