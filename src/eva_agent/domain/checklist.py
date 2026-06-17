"""Контракт распределенного чек-листа планирования."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EntityCount(BaseModel):
    """Три счетчика разных слоев для одной сущности."""

    entity: str
    intent_count: int | None = None
    ref_count: int = 0
    result_count: int | None = None


class PlanningChecklist(BaseModel):
    """Явный протокол рассуждения по агентам."""

    intent: str = ""
    entities: list[str] = Field(default_factory=list)
    cardinality: list[EntityCount] = Field(default_factory=list)
    access: list[str] = Field(default_factory=list)
    needs_chain: bool = False
    delegate: list[str] = Field(default_factory=list)
    resolution: Literal["proceed", "clarify"] = "proceed"
    clarify_reason: str = ""
