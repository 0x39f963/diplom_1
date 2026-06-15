"""Выход-фильтр (ТЗ-3 §2.3): ownership + PII/leak + grounding.

Порядок: детерминированный ownership-check (вне модели, ~0 FP) -> canary-leak -> PII-маскировка
-> мягкая проверка обоснованности. Приоритет решений: block > sanitize > allow.
"""

from __future__ import annotations

from eva_agent.security.ru_pii import mask_pii
from eva_agent.security.verdict import GuardVerdict
from eva_agent.state import ApiFinding

# Канарейка: подставляется в системный промпт; ее появление в ответе = утечка промпта.
CANARY = "CNRY-7Qx9"


def ownership_ok(findings: list[ApiFinding], owner_scope: str) -> bool:
    """Все собранные через API данные принадлежат текущему пользователю/workspace."""
    return all((not f.owner_ref) or f.owner_ref == owner_scope for f in findings)


def filter_output(
    answer: str,
    *,
    findings: list[ApiFinding] | None = None,
    owner_scope: str = "",
    retrieved_citations: list[str] | None = None,
) -> GuardVerdict:
    findings = findings or []
    retrieved_citations = retrieved_citations or []

    if owner_scope and not ownership_ok(findings, owner_scope):
        return GuardVerdict(
            decision="block",
            risk_score=0.95,
            categories=["ownership_violation"],
            reason="В ответе данные другого пользователя/workspace.",
        )

    if CANARY in answer:
        return GuardVerdict(
            decision="block",
            risk_score=0.95,
            categories=["prompt_leak"],
            reason="Обнаружена утечка системного промпта (canary).",
        )

    masked, found = mask_pii(answer)
    if found:
        return GuardVerdict(
            decision="sanitize",
            risk_score=0.6,
            categories=["pii"],
            reason=f"Замаскированы PII: {sorted({t for t, _ in found})}",
            sanitized_text=masked,
        )

    categories: list[str] = []
    risk = 0.0
    # Мягкая проверка обоснованности: есть нормативные цитаты ↔ ответ на них опирается.
    if retrieved_citations and not any(c and c in answer for c in retrieved_citations):
        categories.append("ungrounded")
        risk = 0.3

    return GuardVerdict(decision="allow", risk_score=risk, categories=categories, reason="ok")
