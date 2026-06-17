"""Контракты плана планировщика.

TodoPlan описывает стратегию ответа по протоколу. TodoItem хранит стандартное действие
из каталога, а PlanStep задает конкретный вызов инструмента внутри этого действия.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PlanTool = Literal[
    "eva_get_contract",
    "eva_get_contract_parties",
    "eva_get_counterparty",
    "eva_get_creative_status",
    "eva_list_placements",
    "eva_list_contract_documents",
    "eva_list_unsigned_contracts",
    "eva_search_contracts",
    "retrieve_legal",
    "eva_doc_read",
    "eva_doc_download",
    "eva_doc_attach",
]

DateHint = Literal["none", "yesterday", "last_week", "last_month"]
StatusHint = Literal["none", "unsigned", "draft", "registered"]

TodoType = Literal["blocking", "non_blocking", "dependent"]
TodoStatus = Literal["pending", "done", "blocked", "skipped"]
DialogStatus = Literal["answered", "awaiting_clarification", "in_progress"]


class PlanStep(BaseModel):
    """Один вызов инструмента.

    order - глобальный порядок шага по всему todo-листу. Значение уникально в плане и
    возрастает по ходу исполнения. Ссылки на прошлый результат задаются в args как
    {"$from": {"step": <global_order>, "path": "..."}}.
    """

    order: int = Field(ge=1)
    tool: PlanTool
    args: dict[str, Any] = Field(default_factory=dict)
    date_hint: DateHint = "none"
    status_hint: StatusHint = "none"
    reason: str = ""


class StepResult(BaseModel):
    """Результат исполнения PlanStep для последующей $from-подстановки."""

    order: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)


class TodoItem(BaseModel):
    """Один пункт todo-листа с входами, блокерами и вызовами инструментов."""

    id: str
    type: TodoType = "blocking"
    order: int = Field(ge=1)
    status: TodoStatus = "pending"
    depends_on: list[int] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    result_ref: str = ""
    tool_calls: list[PlanStep] = Field(default_factory=list)

    @property
    def is_mandatory_pending(self) -> bool:
        return self.type == "blocking" and self.status == "pending"


class TodoPlan(BaseModel):
    """План решения под выбранный протокол и текущий статус диалога."""

    goal: str = ""
    protocol_id: str = "clarify_first"
    strategy: str = ""
    items: list[TodoItem] = Field(default_factory=list)
    status: DialogStatus = "in_progress"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    clarify_question: str = ""
    trace: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.items

    def ordered(self) -> list[TodoItem]:
        return sorted(self.items, key=lambda item: item.order)

    def mandatory_items(self, mandatory_ids: list[str]) -> list[TodoItem]:
        return [item for item in self.items if item.id in mandatory_ids]
