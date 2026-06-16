"""Узлы графа для загрузки и сохранения диалоговой памяти."""

from __future__ import annotations

from typing import Any

from eva_agent.dialog.memory_agent import run_memory
from eva_agent.dialog.models import DialogMeaning, DialogMessage, MessageRole, SessionStatus
from eva_agent.dialog.store import get_store
from eva_agent.domain.plan import DialogStatus
from eva_agent.state import AgentState


def load_context(state: AgentState) -> dict[str, object]:
    """Поднять контекст сессии и склеить уточнение в самодостаточный запрос."""

    if not state.session_id:
        return {"user_input_clean": state.user_input_clean or state.user_input_raw}

    store = get_store()
    session = store.get_or_create_session(state.session_id)
    decision = run_memory(store, session, state.user_input_raw)
    store.append_message(state.session_id, "user", state.user_input_raw)
    next_status = "active" if session.status == "awaiting_clarification" else session.status
    store.update_session(state.session_id, status=next_status, bump_turn=True)
    return {
        "memory": decision,
        "user_input_clean": decision.merged_context,
    }


def save_turn(state: AgentState) -> dict[str, object]:
    """Записать ответ хода и снимок смысла в историю диалога."""

    if not state.session_id:
        return {}

    store = get_store()
    _ensure_user_message(state)

    final = state.final or ""
    is_clarify = _is_clarification(state)
    role: MessageRole = "clarification" if is_clarify else "assistant"
    store.append_message(state.session_id, role, final)

    dialog_status = _dialog_status(state)
    session_status: SessionStatus = (
        "awaiting_clarification" if dialog_status == "awaiting_clarification" else "active"
    )
    store.update_session(
        state.session_id,
        status=session_status,
        last_intent=state.intent.kind if state.intent else None,
    )

    session = store.get_or_create_session(state.session_id)
    store.add_snapshot(state.session_id, _build_snapshot(state, turn=session.turn))
    return {}


def _ensure_user_message(state: AgentState) -> None:
    if not state.session_id:
        return
    store = get_store()
    latest = store.list_messages(state.session_id, limit=1)
    if latest and _is_same_user_message(latest[0], state.user_input_raw):
        return
    store.append_message(state.session_id, "user", state.user_input_raw)
    store.update_session(state.session_id, bump_turn=True)


def _is_same_user_message(message: DialogMessage, text: str) -> bool:
    return message.role == "user" and message.text == text


def _is_clarification(state: AgentState) -> bool:
    if state.todo_plan is not None and state.todo_plan.status == "awaiting_clarification":
        return True
    return state.intent is not None and state.intent.kind == "need_clarification"


def _dialog_status(state: AgentState) -> DialogStatus:
    if state.todo_plan is not None:
        return state.todo_plan.status
    return "awaiting_clarification" if _is_clarification(state) else "answered"


def _build_snapshot(state: AgentState, *, turn: int) -> DialogMeaning:
    status = _dialog_status(state)
    return DialogMeaning(
        session_id=state.session_id or "",
        turn=turn,
        summary=_truncate(state.final or ""),
        open_question=_open_question(state) if status == "awaiting_clarification" else None,
        accumulated_meaning=_accumulated_meaning(state),
        reasoning=_reasoning(state),
        todo_list=_todo_list(state),
        dialog_status=status,
    )


def _open_question(state: AgentState) -> str | None:
    if state.open_question:
        return state.open_question
    if state.todo_plan is not None and state.todo_plan.clarify_question:
        return state.todo_plan.clarify_question
    if state.intent is not None and state.intent.needed_inputs:
        return "; ".join(state.intent.needed_inputs)
    return None


def _accumulated_meaning(state: AgentState) -> str:
    query = state.user_input_clean or state.user_input_raw
    if state.memory is None:
        return query
    prior = state.memory.accumulated_meaning.strip()
    merged = state.memory.merged_context.strip() or query
    if prior and merged and merged not in prior:
        return f"{prior}\nТекущий запрос: {merged}"
    return prior or merged


def _reasoning(state: AgentState) -> str:
    parts: list[str] = []
    if state.intent is not None and state.intent.rationale:
        parts.append(state.intent.rationale)
    if state.todo_plan is not None:
        parts.append(f"protocol={state.todo_plan.protocol_id}; status={state.todo_plan.status}")
        blockers = [
            blocker
            for todo in state.todo_plan.ordered()
            for blocker in todo.blockers
        ]
        if blockers:
            parts.append("blockers=" + "; ".join(blockers[:3]))
    return _truncate(" | ".join(parts), limit=1000)


def _todo_list(state: AgentState) -> dict[str, Any] | None:
    if state.todo_plan is None:
        return None
    return state.todo_plan.model_dump(mode="json")


def _truncate(text: str, *, limit: int = 500) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."
