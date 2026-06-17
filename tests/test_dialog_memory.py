from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import eva_agent.dialog.memory_agent as memory_agent_module
import eva_agent.nodes.agents as agents_module
import eva_agent.nodes.domain_nodes as domain_nodes
import eva_agent.nodes.guards as guards_module
from eva_agent.dialog.models import DialogMeaning
from eva_agent.dialog.store import DialogStore, get_store, reset_store
from eva_agent.domain.plan import PlanStep, TodoItem, TodoPlan
from eva_agent.graph import build_graph
from eva_agent.llm.base import LLMResponse
from eva_agent.nodes.dialog_nodes import load_context, save_turn
from eva_agent.security.verdict import GuardVerdict
from eva_agent.state import AgentState
from eva_agent.tracing import run_request


@pytest.fixture
def dialog_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "dialog.db"
    monkeypatch.setenv("EVA_DIALOG_DB", str(db_path))
    reset_store()
    yield db_path
    reset_store()


class _StaticClient:
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
        assert system
        assert user
        assert temperature == 0.0
        assert json_mode is True
        return LLMResponse(text=self.text, model="fake", backend="fake")


class _AgentClient:
    def invoke(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        assert user
        if "оркестратор" in system:
            payload = {
                "kind": "mixed_diagnostic",
                "confidence": 0.9,
                "rationale": "нужны данные договора",
                "needed_inputs": [],
            }
            return LLMResponse(text=json.dumps(payload, ensure_ascii=False), model="fake", backend="fake")
        if "критик" in system:
            return LLMResponse(
                text='{"decision":"accept","target":null,"reason":""}',
                model="fake",
                backend="fake",
            )
        return LLMResponse(
            text="Исполнитель по договору Д-2025/249 найден.",
            model="fake",
            backend="fake",
        )


def _blocked_party_plan() -> TodoPlan:
    return TodoPlan(
        goal="кто исполнитель в договоре",
        protocol_id="party_lookup",
        strategy="получить стороны договора после уточнения",
        items=[
            TodoItem(
                id="resolve_party_role",
                order=1,
                status="blocked",
                inputs={"role": "executor"},
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
        clarify_question="Уточните номер договора.",
    )


def test_store_roundtrip(dialog_db: Path) -> None:
    store = DialogStore(str(dialog_db))
    try:
        session = store.get_or_create_session("s1")
        again = store.get_or_create_session("s1")
        assert session.session_id == "s1"
        assert again.turn == 0

        updated = store.update_session(
            "s1",
            status="awaiting_clarification",
            last_intent="need_clarification",
            bump_turn=True,
        )
        assert updated.status == "awaiting_clarification"
        assert updated.turn == 1

        store.append_message("s1", "user", "кто исполнитель в договоре?")
        store.append_message("s1", "clarification", "Уточните номер договора.")
        messages = store.list_messages("s1")
        assert [message.role for message in messages] == ["user", "clarification"]

        plan = _blocked_party_plan()
        snapshot = DialogMeaning(
            session_id="s1",
            turn=updated.turn,
            summary="запрошено уточнение",
            open_question="Уточните номер договора.",
            accumulated_meaning="пользователь хочет узнать исполнителя по договору",
            reasoning="нужен contract_id",
            todo_list=plan.model_dump(mode="json"),
            dialog_status="awaiting_clarification",
        )
        stored = store.add_snapshot("s1", snapshot)
        latest = store.latest_snapshot("s1")

        assert stored.ts
        assert latest is not None
        assert latest.todo_list is not None
        assert latest.todo_list["protocol_id"] == "party_lookup"
        assert latest.dialog_status == "awaiting_clarification"
        assert latest.open_question == "Уточните номер договора."
    finally:
        store.close()


def test_memory_agent_continuation_lifts_previous_plan(
    dialog_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = DialogStore(str(dialog_db))
    try:
        store.get_or_create_session("s1")
        session = store.update_session("s1", status="awaiting_clarification", bump_turn=True)
        store.add_snapshot(
            "s1",
            DialogMeaning(
                session_id="s1",
                turn=session.turn,
                open_question="номер договора",
                accumulated_meaning="пользователь хочет узнать исполнителя по договору",
                todo_list=_blocked_party_plan().model_dump(mode="json"),
                dialog_status="awaiting_clarification",
            ),
        )
        payload = {
            "is_continuation": True,
            "merged_context": "кто исполнитель в договоре Д-2025/249?",
            "accumulated_meaning": "пользователь хочет узнать исполнителя по договору Д-2025/249",
            "resolved_inputs": {"contract_id": "Д-2025/249"},
        }
        client = _StaticClient(json.dumps(payload, ensure_ascii=False))

        def fake_get_client(role: str) -> _StaticClient:
            assert role == "memory"
            return client

        monkeypatch.setattr(memory_agent_module, "get_client", fake_get_client)

        decision = memory_agent_module.run_memory(store, session, "Д-2025/249")

        assert client.calls == 1
        assert decision.is_continuation is True
        assert decision.prev_todo_plan is not None
        assert decision.prev_todo_plan.status == "awaiting_clarification"
        assert decision.resolved_inputs == {"contract_id": "Д-2025/249"}
        assert "Д-2025/249" in decision.merged_context
    finally:
        store.close()


def test_multi_turn_continuation_replans_and_saves_history(
    dialog_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls = 0

    def fake_detect_injection(user_input: str, untrusted_data: str = "") -> GuardVerdict:
        assert user_input
        assert untrusted_data == ""
        return GuardVerdict(decision="allow", risk_score=0.0, reason="ok")

    def fake_get_agent_client(role: str) -> _AgentClient:
        assert role in {"reasoning", "default"}
        return _AgentClient()

    def fake_build_plan(
        query: str,
        *,
        prior_meaning: str = "",
        domain_slice: Any | None = None,
        intent_kind: str | None = None,
    ) -> TodoPlan:
        nonlocal build_calls
        build_calls += 1
        assert query == "кто исполнитель в договоре?"
        assert prior_meaning == ""
        assert domain_slice is not None
        assert intent_kind == "mixed_diagnostic"
        return _blocked_party_plan()

    memory_payload = {
        "is_continuation": True,
        "merged_context": "кто исполнитель в договоре Д-2025/249?",
        "accumulated_meaning": "пользователь хочет узнать исполнителя по договору Д-2025/249",
        "resolved_inputs": {"contract_id": "Д-2025/249"},
    }
    memory_client = _StaticClient(json.dumps(memory_payload, ensure_ascii=False))

    def fake_get_memory_client(role: str) -> _StaticClient:
        assert role == "memory"
        return memory_client

    domain_client = _StaticClient('{"entities":["Contract","ContractParty","Counterparty"]}')

    def fake_get_domain_client(role: str) -> _StaticClient:
        assert role == "domain"
        return domain_client

    monkeypatch.setattr(guards_module, "detect_injection", fake_detect_injection)
    monkeypatch.setattr(agents_module, "get_client", fake_get_agent_client)
    monkeypatch.setattr(agents_module, "build_plan", fake_build_plan)
    monkeypatch.setattr(memory_agent_module, "get_client", fake_get_memory_client)
    monkeypatch.setattr(domain_nodes, "get_client", fake_get_domain_client)

    graph = build_graph()
    sid = "multi-turn"

    first = run_request(graph, "кто исполнитель в договоре?", session_id=sid)
    store = get_store()
    first_session = store.get_or_create_session(sid)
    first_snapshot = store.latest_snapshot(sid)

    assert first["intent"].kind == "need_clarification"
    assert first["final"].startswith("Чтобы ответить точно")
    assert first_session.status == "awaiting_clarification"
    assert first_snapshot is not None
    assert first_snapshot.open_question == "Уточните номер договора."
    assert first_snapshot.todo_list is not None
    assert first_snapshot.dialog_status == "awaiting_clarification"

    second = run_request(graph, "Д-2025/249", session_id=sid)
    second_session = store.get_or_create_session(sid)
    second_snapshot = store.latest_snapshot(sid)
    messages = store.list_messages(sid)

    assert build_calls == 1
    assert memory_client.calls == 1
    assert second["intent"].kind == "mixed_diagnostic"
    assert second["memory"].prev_todo_plan is not None
    assert second["todo_plan"].status == "answered"
    assert second["todo_plan"].items[0].status == "done"
    assert second_session.status == "active"
    assert second_snapshot is not None
    assert second_snapshot.open_question is None
    assert second_snapshot.dialog_status == "answered"
    assert [message.role for message in messages] == [
        "user",
        "clarification",
        "user",
        "assistant",
    ]


def test_no_session_id_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_store() -> DialogStore:
        raise AssertionError("store must not be used without session_id")

    monkeypatch.setattr("eva_agent.nodes.dialog_nodes.get_store", fail_get_store)
    state = AgentState(user_input_raw="запрос", user_input_clean="запрос")

    assert load_context(state) == {"user_input_clean": "запрос"}
    assert save_turn(state) == {}

    class _Graph:
        def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
            return payload

    result = run_request(_Graph(), "запрос")
    assert result == {"user_input_raw": "запрос", "session_id": None}
