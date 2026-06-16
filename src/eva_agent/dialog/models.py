"""Pydantic-контракты истории диалога и решения memory-agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from eva_agent.domain.plan import DialogStatus, TodoPlan

SessionStatus = Literal["active", "awaiting_clarification", "closed"]
MessageRole = Literal["user", "assistant", "clarification"]


class Session(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    status: SessionStatus = "active"
    last_intent: str | None = None
    turn: int = 0


class DialogMessage(BaseModel):
    id: int | None = None
    session_id: str
    role: MessageRole
    text: str
    ts: str


class DialogMeaning(BaseModel):
    """Снимок смысла, статуса и плана за один ход диалога."""

    session_id: str
    turn: int
    summary: str = ""
    open_question: str | None = None
    accumulated_meaning: str = ""
    reasoning: str = ""
    todo_list: dict[str, Any] | None = None
    dialog_status: DialogStatus | None = None
    ts: str | None = None


class MemoryDecision(BaseModel):
    """Выход memory-agent для графа и планировщика."""

    is_continuation: bool = False
    merged_context: str
    accumulated_meaning: str = ""
    prev_todo_plan: TodoPlan | None = None
    resolved_inputs: dict[str, str] = Field(default_factory=dict)

