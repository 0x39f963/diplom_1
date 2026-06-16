"""Memory-agent: решает продолжение диалога и поднимает прошлый todo-план."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from eva_agent.dialog.models import DialogMeaning, MemoryDecision, Session
from eva_agent.dialog.store import DialogStore
from eva_agent.domain.plan import TodoPlan
from eva_agent.llm.config import get_client
from eva_agent.settings import settings

_MEMORY_SYS = (
    "Ты - модуль памяти диалога ИИ-помощника по рекламному праву (38-ФЗ) и работе в кабинете системы. "
    "На вход ты получаешь историю диалога, накопленный смысл предыдущих ходов, последний уточняющий "
    "вопрос системы и новое сообщение пользователя. Определи, новый это запрос или продолжение, и "
    "собери полный контекст для следующего узла. Верни СТРОГО JSON: "
    '{"is_continuation":true|false,"merged_context":"полный самодостаточный запрос",'
    '"accumulated_meaning":"глобальный смысл диалога",'
    '"resolved_inputs":{"что спрашивали":"что прислал пользователь"}}. '
    "Если был уточняющий вопрос и новое сообщение отвечает на него, is_continuation=true. "
    "merged_context всегда самодостаточен. Ничего не выдумывай."
)


def run_memory(store: DialogStore, session: Session, new_message: str) -> MemoryDecision:
    """Вернуть решение памяти для нового сообщения в сессии."""

    latest = store.latest_snapshot(session.session_id)
    if session.status != "awaiting_clarification":
        return MemoryDecision(
            is_continuation=False,
            merged_context=new_message,
            accumulated_meaning=latest.accumulated_meaning if latest else "",
        )

    response_text = _invoke_memory(store, session, latest, new_message)
    data = _safe_json(response_text)
    decision = _decision_from_data(data, new_message)
    if decision is None:
        return MemoryDecision(is_continuation=False, merged_context=new_message)

    prev_plan = _prev_todo_plan(latest) if decision.is_continuation else None
    accumulated = decision.accumulated_meaning or (latest.accumulated_meaning if latest else "")
    return decision.model_copy(update={"prev_todo_plan": prev_plan, "accumulated_meaning": accumulated})


def _invoke_memory(
    store: DialogStore,
    session: Session,
    latest: DialogMeaning | None,
    new_message: str,
) -> str:
    context = _memory_context(store, session, latest, new_message)
    try:
        return get_client("memory").invoke(
            _MEMORY_SYS,
            context,
            temperature=0.0,
            json_mode=True,
        ).text
    except Exception:
        return ""


def _memory_context(
    store: DialogStore,
    session: Session,
    latest: DialogMeaning | None,
    new_message: str,
) -> str:
    messages = store.list_messages(
        session.session_id,
        limit=max(1, settings.eva_dialog_summarize_after),
    )
    payload: dict[str, Any] = {
        "session": {
            "session_id": session.session_id,
            "status": session.status,
            "turn": session.turn,
        },
        "latest_snapshot": latest.model_dump(mode="json") if latest else None,
        "messages": [
            {"role": message.role, "text": message.text, "ts": message.ts}
            for message in messages
        ],
        "new_message": new_message,
    }
    return json.dumps(payload, ensure_ascii=False)


def _safe_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _decision_from_data(data: dict[str, Any], new_message: str) -> MemoryDecision | None:
    if not data:
        return None
    data.setdefault("merged_context", new_message)
    data.setdefault("is_continuation", False)
    try:
        return MemoryDecision.model_validate(data)
    except ValidationError:
        return None


def _prev_todo_plan(snapshot: DialogMeaning | None) -> TodoPlan | None:
    if snapshot is None or snapshot.todo_list is None:
        return None
    try:
        return TodoPlan.model_validate(snapshot.todo_list)
    except ValidationError:
        return None

