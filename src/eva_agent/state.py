"""Pydantic-контракты графа (ТЗ-2 §2).

Все границы между узлами типизированы. Structured-output моделей возвращает эти же
типы - это дает легкий tool-call / intent eval.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from eva_agent.dialog.models import MemoryDecision
from eva_agent.domain.plan import TodoPlan
from eva_agent.security.verdict import GuardVerdict

IntentKind = Literal[
    "legal_consult",
    "interface_consult",
    "mixed_diagnostic",
    "need_clarification",
    "out_of_scope",
]
RouteTarget = Literal["legal_agent", "interface_agent", "data_gather"]


class Chunk(BaseModel):
    """Извлеченный фрагмент корпуса с метаданными для цитирования и фильтрации."""

    text: str
    citation: str = ""           # человекочитаемая ссылка: «п. 2 ч. 3 ст. 18.1»
    law_number: str = ""
    article: str = ""
    trust_level: Literal["primary", "secondary", "tertiary"] = "primary"
    score: float = 0.0
    source_url: str = ""


class Intent(BaseModel):
    kind: IntentKind
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    needed_inputs: list[str] = Field(default_factory=list)   # для need_clarification


class RetrievalResult(BaseModel):
    query: str
    collection: Literal["legal", "howto"]
    chunks: list[Chunk] = Field(default_factory=list)


class ApiFinding(BaseModel):
    """Результат read-only обертки вокруг внешнего backend API."""

    tool: str
    args: dict = Field(default_factory=dict)
    data: dict = Field(default_factory=dict)
    owner_ref: str = ""          # скоуп владельца -> для детерминированного ownership_check


class CriticVerdict(BaseModel):
    """Вердикт критика. Критик НЕ маршрутизирует - решение о повторе принимает supervisor."""

    decision: Literal["accept", "rework"]
    target: RouteTarget | None = None
    reason: str = ""
    missing: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """Состояние графа LangGraph (guards -> supervisor -> agents -> critic -> guards -> finalize)."""

    user_input_raw: str
    session_id: str | None = None
    messages: list = Field(default_factory=list)

    user_input_clean: str | None = None     # после input_filter
    memory: MemoryDecision | None = None
    guard_in: GuardVerdict | None = None
    guard_out: GuardVerdict | None = None

    intent: Intent | None = None
    retrieved: dict[str, RetrievalResult] = Field(default_factory=dict)   # "legal"/"howto"
    api_findings: list[ApiFinding] = Field(default_factory=list)
    todo_plan: TodoPlan | None = None
    plan_attempts: int = 0
    drafts: dict[str, str] = Field(default_factory=dict)                  # ответы агентов
    critic: CriticVerdict | None = None

    loop_count: int = 0
    citations: list[str] = Field(default_factory=list)
    final: str | None = None
    open_question: str | None = None
