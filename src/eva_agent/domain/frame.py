"""Typed semantic frame for deterministic planning."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from eva_agent.nlu.preprocess import DateHint

FrameOperation = Literal["read", "list", "compare", "open", "download", "attach", "diagnose"]
FrameCardinality = Literal["one", "all", "n"]
FrameOutput = Literal["value", "card", "list", "summary"]


class FrameFilters(BaseModel):
    """Normalized filters copied from NLU hints."""

    date_hint: DateHint = "none"
    status: list[str] = Field(default_factory=list)


class PlanningFrame(BaseModel):
    """Compact semantic frame filled by the parser and consumed by the compiler."""

    operation: FrameOperation = "read"
    target: str = ""
    relation: str | None = None
    fields: list[str] = Field(default_factory=list)
    filters: FrameFilters = Field(default_factory=FrameFilters)
    cardinality: FrameCardinality = "one"
    selector: dict[str, str] = Field(default_factory=dict)
    output: FrameOutput = "summary"
    subtasks: list[PlanningFrame] = Field(default_factory=list)
    needs_clarification: bool = False
    clarify_reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_factors: dict[str, float] = Field(default_factory=dict)
    trace: list[str] = Field(default_factory=list)


__all__ = [
    "FrameCardinality",
    "FrameFilters",
    "FrameOperation",
    "FrameOutput",
    "PlanningFrame",
]
