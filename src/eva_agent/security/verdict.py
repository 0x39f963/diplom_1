"""GuardVerdict - общий контракт всех guard-узлов защитного слоя (ТЗ-3).

Один вердикт на вход-фильтр, детектор инъекций и выход-фильтр. Маршрутизация графа
идет по полю `decision`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GuardDecision = Literal["allow", "sanitize", "block", "escalate"]
RiskType = Literal[
    "none",
    "prompt_injection",
    "policy_forbidden",
    "pii_exfiltration",
    "unknown",
]


class GuardVerdict(BaseModel):
    """Решение guard-узла.

    - allow    - пропустить как есть;
    - sanitize - пропустить очищенный `sanitized_text`;
    - block    - заблокировать -> refuse;
    - escalate - в проде human-in-the-loop, в дипломе = block + лог.
    """

    decision: GuardDecision
    risk_score: float = Field(ge=0.0, le=1.0)
    categories: list[str] = Field(default_factory=list)
    reason: str = ""
    sanitized_text: str | None = None
    risk_type: RiskType = "none"
    matched_rules: list[str] = Field(default_factory=list)
    domain_signals: list[str] = Field(default_factory=list)
    safe_read_action: bool = False

    @property
    def passed(self) -> bool:
        return self.decision in ("allow", "sanitize")
