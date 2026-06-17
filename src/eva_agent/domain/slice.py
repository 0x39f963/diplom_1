"""Контракты доменного среза для планировщика."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RelationSpec(BaseModel):
    """Одно машинное FK-ребро для авто-связывания цепочек."""

    source_entity: str
    target_entity: str
    source_tool: str
    source_path: str
    target_tool: str
    target_arg: str
    selector: str | None = None
    cardinality: Literal["one", "many"] = "one"


class DomainSlice(BaseModel):
    """Маленький релевантный подграф домена, который селектор отдает планировщику."""

    entities: list[str] = Field(default_factory=list)
    relations: list[RelationSpec] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    hint: str = ""
    scope: str = ""
