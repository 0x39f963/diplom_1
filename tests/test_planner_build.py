from __future__ import annotations

import json
from typing import Any

from eva_agent.domain.plan import PlanStep, TodoItem, TodoPlan
from eva_agent.llm.base import LLMResponse
from eva_agent.planner import build as planner_build


class _FakeClient:
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
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "json_mode": json_mode,
            }
        )
        return LLMResponse(text=self.text, model="fake", backend="fake")


def _patch_client(monkeypatch: Any, text: str) -> _FakeClient:
    client = _FakeClient(text)

    def fake_get_client(role: str) -> _FakeClient:
        assert role == "planner"
        return client

    monkeypatch.setattr(planner_build, "get_client", fake_get_client)
    return client


def test_build_plan_uses_planner_client_and_returns_todoplan(monkeypatch: Any) -> None:
    payload = {
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
    client = _patch_client(monkeypatch, json.dumps(payload, ensure_ascii=False))

    plan = planner_build.build_plan("кто заказчик по договору CT-1")

    assert isinstance(plan, TodoPlan)
    assert plan.protocol_id == "party_lookup"
    assert plan.items[0].id == "resolve_party_role"
    assert plan.items[0].inputs["contract_id"] == "CT-1"
    assert len(client.calls) == 1
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["json_mode"] is True
    assert "КАТАЛОГ ДОСТУПНЫХ TODO" in client.calls[0]["system"]
    assert "CT-1" in client.calls[0]["system"]


def test_build_plan_invalid_json_awaits_clarification(monkeypatch: Any) -> None:
    _patch_client(monkeypatch, "{not-json")

    plan = planner_build.build_plan("помоги")

    assert plan.status == "awaiting_clarification"
    assert plan.clarify_question


def test_build_plan_low_confidence_awaits_clarification(monkeypatch: Any) -> None:
    payload = {
        "goal": "покажи карточку договора CT-1",
        "protocol_id": "contract_card",
        "strategy": "получить договор",
        "items": [
            {
                "id": "get_contract",
                "type": "blocking",
                "order": 1,
                "inputs": {"contract_id": "CT-1"},
                "tool_calls": [
                    {
                        "order": 1,
                        "tool": "eva_get_contract",
                        "args": {"contract_id": "CT-1"},
                        "date_hint": "none",
                        "status_hint": "none",
                        "reason": "получить карточку",
                    }
                ],
            }
        ],
        "status": "in_progress",
        "confidence": 0.1,
        "clarify_question": "",
    }
    _patch_client(monkeypatch, json.dumps(payload, ensure_ascii=False))

    plan = planner_build.build_plan("покажи карточку договора CT-1")

    assert plan.status == "awaiting_clarification"
    assert plan.confidence < planner_build.PLANNER_MIN_CONFIDENCE


def test_replan_resolved_input_unblocks_target_todo() -> None:
    prev = TodoPlan(
        goal="кто заказчик по договору",
        protocol_id="party_lookup",
        items=[
            TodoItem(
                id="resolve_party_role",
                order=1,
                status="blocked",
                blockers=["нет contract_id"],
                tool_calls=[
                    PlanStep(
                        order=1,
                        tool="eva_get_contract_parties",
                        args={},
                    )
                ],
            )
        ],
        status="awaiting_clarification",
        confidence=0.9,
        clarify_question="Уточните договор.",
    )

    plan = planner_build.replan(prev, "это CT-1", resolved_inputs={"contract_id": "CT-1"})

    assert plan.status == "in_progress"
    assert plan.items[0].status == "pending"
    assert plan.items[0].blockers == []
    assert plan.items[0].inputs["contract_id"] == "CT-1"
    assert plan.items[0].tool_calls[0].args["contract_id"] == "CT-1"
    assert prev.items[0].status == "blocked"
