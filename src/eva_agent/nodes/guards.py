"""Узлы защитного слоя графа (ТЗ-3): вход (фильтр + детектор инъекций) и выход (ownership/PII/grounding)."""

from __future__ import annotations

from typing import Any

from eva_agent.mock.data import OWNER_SCOPE
from eva_agent.security.injection_detector import detect_injection
from eva_agent.security.input_filter import filter_input
from eva_agent.security.output_filter import filter_output
from eva_agent.security.verdict import GuardVerdict
from eva_agent.state import AgentState
from eva_agent.tools.entity_ref import extract_refs, has_domain_signal
from eva_agent.tracing import log_span_event


def input_guard(state: AgentState) -> dict[str, Any]:
    raw = state.user_input_raw
    domain_signals = _domain_signals(raw)
    deterministic = filter_input(raw)
    safe_read_action = bool(domain_signals) and deterministic.decision != "block"
    if deterministic.decision == "block":
        verdict = _with_guard_context(
            deterministic,
            domain_signals=domain_signals,
            safe_read_action=safe_read_action,
        )
        return {
            "guard_in": verdict,
            "user_input_clean": raw,
            "debug": {**state.debug, "guard": _guard_debug(verdict)},
        }
    clean = deterministic.sanitized_text or raw
    judged = detect_injection(clean)
    verdict = judged if judged.decision == "block" else deterministic
    verdict = _with_guard_context(
        verdict,
        domain_signals=domain_signals,
        safe_read_action=safe_read_action,
    )
    return {
        "guard_in": verdict,
        "user_input_clean": clean,
        "debug": {**state.debug, "guard": _guard_debug(verdict)},
    }


def _domain_signals(raw: str) -> list[str]:
    refs = extract_refs(raw)
    if refs.all_ids:
        return refs.all_ids
    return ["domain_signal"] if has_domain_signal(raw) else []


def _with_guard_context(
    verdict: GuardVerdict,
    *,
    domain_signals: list[str],
    safe_read_action: bool,
) -> GuardVerdict:
    return verdict.model_copy(
        update={
            "domain_signals": domain_signals,
            "safe_read_action": safe_read_action,
        }
    )


def _guard_debug(verdict: GuardVerdict) -> dict[str, Any]:
    payload = {
        "decision": verdict.decision,
        "risk_type": verdict.risk_type,
        "matched_rules": list(verdict.matched_rules),
        "domain_signals": list(verdict.domain_signals),
        "safe_read_action": verdict.safe_read_action,
        "risk_score": verdict.risk_score,
    }
    log_span_event({"guard": payload})
    return payload


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
