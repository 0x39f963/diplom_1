"""Узлы защитного слоя графа (ТЗ-3): вход (фильтр + детектор инъекций) и выход (ownership/PII/grounding)."""

from __future__ import annotations

from typing import Any

from eva_agent.mock.data import OWNER_SCOPE
from eva_agent.nlu.preprocess import is_read_only_domain_command, preprocess
from eva_agent.security.injection_detector import detect_injection
from eva_agent.security.input_filter import filter_input
from eva_agent.security.output_filter import filter_output
from eva_agent.security.verdict import GuardVerdict
from eva_agent.state import AgentState
from eva_agent.tools.entity_ref import extract_refs, has_domain_signal
from eva_agent.tracing import log_span_event

_REAL_INJECTION_RISK_SCORE = 0.85
_REAL_INJECTION_MARKERS = (
    "ignore",
    "disregard",
    "jailbreak",
    "developer mode",
    "system prompt",
    "reveal prompt",
    "hidden instruction",
    "instruction",
    "override",
    "bypass",
    "roleplay",
    "exfil",
    "игнор",
    "забудь",
    "джейлбрейк",
    "инструкц",
    "системный промпт",
    "системного промпта",
    "скрытые инструкции",
    "переопредел",
    "обойти правила",
    "раскрыть системный",
    "передать данные наружу",
    "эксфильтр",
    "credential",
    "secret",
)


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
    judged = detect_injection(clean, domain_signals=domain_signals)
    if _should_allow_domain_read_after_judge(
        judged,
        clean,
        domain_signals=domain_signals,
        safe_read_action=safe_read_action,
    ):
        verdict = _override_domain_read(judged)
    else:
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


def _should_allow_domain_read_after_judge(
    verdict: GuardVerdict,
    query: str,
    *,
    domain_signals: list[str],
    safe_read_action: bool,
) -> bool:
    return (
        verdict.decision == "block"
        and safe_read_action
        and bool(domain_signals)
        and is_read_only_domain_command(query)
        and not _judge_reports_real_injection(verdict)
    )


def _judge_reports_real_injection(verdict: GuardVerdict) -> bool:
    if verdict.risk_type not in {"prompt_injection", "pii_exfiltration"}:
        return False
    if verdict.risk_score >= _REAL_INJECTION_RISK_SCORE:
        return True
    text = " ".join([verdict.reason, *verdict.categories, *verdict.matched_rules]).lower()
    return any(marker in text for marker in _REAL_INJECTION_MARKERS)


def _override_domain_read(verdict: GuardVerdict) -> GuardVerdict:
    matched_rules = [*verdict.matched_rules, "override:domain_read_after_judge"]
    payload = {
        "from_decision": verdict.decision,
        "to_decision": "allow",
        "risk_type": verdict.risk_type,
        "risk_score": verdict.risk_score,
        "matched_rules": matched_rules,
    }
    log_span_event({"guard_override": payload})
    return verdict.model_copy(update={"decision": "allow", "matched_rules": matched_rules})


def _domain_signals(raw: str) -> list[str]:
    refs = extract_refs(raw)
    if refs.all_ids:
        return refs.all_ids
    return ["domain_signal"] if has_domain_signal(raw) and _has_domain_content_signal(raw) else []


def _has_domain_content_signal(raw: str) -> bool:
    features = preprocess(raw)
    return bool(features.entities or features.roles or features.statuses)


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
