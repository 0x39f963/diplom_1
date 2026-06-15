"""Узлы защитного слоя графа (ТЗ-3): вход (фильтр + детектор инъекций) и выход (ownership/PII/grounding)."""

from __future__ import annotations

from eva_agent.mock.data import OWNER_SCOPE
from eva_agent.security.injection_detector import detect_injection
from eva_agent.security.input_filter import filter_input
from eva_agent.security.output_filter import filter_output
from eva_agent.state import AgentState


def input_guard(state: AgentState) -> dict:
    raw = state.user_input_raw
    deterministic = filter_input(raw)
    if deterministic.decision == "block":
        return {"guard_in": deterministic, "user_input_clean": raw}
    clean = deterministic.sanitized_text or raw
    judged = detect_injection(clean)
    verdict = judged if judged.decision == "block" else deterministic
    return {"guard_in": verdict, "user_input_clean": clean}


def output_guard(state: AgentState) -> dict:
    verdict = filter_output(
        state.final or "",
        findings=state.api_findings,
        owner_scope=OWNER_SCOPE,
        retrieved_citations=state.citations,
    )
    if verdict.decision == "block":
        return {"guard_out": verdict, "final": "Ответ отклонен выходным контролем безопасности."}
    if verdict.decision == "sanitize" and verdict.sanitized_text is not None:
        return {"guard_out": verdict, "final": verdict.sanitized_text}
    return {"guard_out": verdict}
