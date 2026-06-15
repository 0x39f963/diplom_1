"""Узлы-агенты графа (ТЗ-2 §3): supervisor, консультанты, data_gather, critic, finalize, clarify, refuse."""

from __future__ import annotations

import json
from typing import Any

from eva_agent.llm.config import get_client
from eva_agent.mock.data import eva_get_creative_status, eva_list_unsigned_contracts
from eva_agent.security.spotlight import SPOTLIGHT_INSTRUCTION, spotlight
from eva_agent.state import AgentState, CriticVerdict, Intent
from eva_agent.tools.retrieve import retrieve_howto, retrieve_legal

_INTENT_KINDS = {
    "legal_consult", "interface_consult", "mixed_diagnostic", "need_clarification", "out_of_scope",
}
_ROUTE_TARGETS = {"legal_agent", "interface_agent", "data_gather"}


def _safe_json(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _str_list(value: Any) -> list[str]:
    """Привести к списку строк: модель может вернуть пункты словарями ({name, description})."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("name") or item.get("field") or item.get("id")
            if name:
                out.append(str(name))
        elif item is not None:
            out.append(str(item))
    return out


_SUPERVISOR_SYS = (
    "Ты - оркестратор ИИ-помощника по рекламному праву (38-ФЗ). "
    "Определи тип запроса пользователя и верни СТРОГО JSON "
    '{"kind":"legal_consult|interface_consult|mixed_diagnostic|need_clarification|out_of_scope",'
    '"confidence":0..1,"rationale":"...","needed_inputs":[...]}. '
    "legal_consult - вопрос про нормы/закон; interface_consult - как сделать в кабинете; "
    "mixed_diagnostic - нужно посмотреть состояние данных (договоры/креативы); "
    "need_clarification - не хватает данных; out_of_scope - не по теме закона о рекламе."
)


def supervisor(state: AgentState) -> dict:
    query = state.user_input_clean or state.user_input_raw
    response = get_client("reasoning").invoke(_SUPERVISOR_SYS, query, temperature=0.0, json_mode=True)
    data = _safe_json(response.text)
    kind = data.get("kind") if data.get("kind") in _INTENT_KINDS else "out_of_scope"
    try:
        confidence = float(data.get("confidence", 0.5) or 0.5)
    except (TypeError, ValueError):
        confidence = 0.5
    intent = Intent(
        kind=kind,
        confidence=confidence,
        rationale=str(data.get("rationale", "")),
        needed_inputs=_str_list(data.get("needed_inputs")),
    )
    return {"intent": intent}


_LEGAL_SYS = (
    "Ты - юрист-консультант по 38-ФЗ «О рекламе». Отвечай кратко и по-русски, опираясь ТОЛЬКО на "
    "приведенные НОРМЫ, обязательно с цитатой (ст./ч./п.). Если в нормах нет ответа - честно скажи. "
    + SPOTLIGHT_INSTRUCTION
)


def legal_agent(state: AgentState) -> dict:
    query = state.user_input_clean or state.user_input_raw
    result = retrieve_legal(query)
    norms = "\n\n".join(spotlight(f"[{c.citation}] {c.text[:1200]}") for c in result.chunks[:5])
    answer = get_client("reasoning").invoke(
        _LEGAL_SYS, f"ВОПРОС: {query}\n\nНОРМЫ:\n{norms or '(не найдено)'}", temperature=0.1
    )
    citations = [c.citation for c in result.chunks[:5] if c.citation]
    return {
        "retrieved": {**state.retrieved, "legal": result},
        "drafts": {**state.drafts, "legal": answer.text},
        "citations": citations,
    }


_INTERFACE_SYS = (
    "Ты - консультант по работе в кабинете системы. Объясни шаги в системе "
    "кратко и по делу, опираясь на справку и данные системы. " + SPOTLIGHT_INSTRUCTION
)


def interface_agent(state: AgentState) -> dict:
    query = state.user_input_clean or state.user_input_raw
    result = retrieve_howto(query)
    context = "\n\n".join(spotlight(c.text[:800]) for c in result.chunks[:4])
    findings = ""
    if state.api_findings:
        payload = json.dumps([f.data for f in state.api_findings], ensure_ascii=False)
        findings = "\n\nДАННЫЕ СИСТЕМЫ:\n" + spotlight(payload[:1500])
    answer = get_client("default").invoke(
        _INTERFACE_SYS, f"ВОПРОС: {query}\n\nСПРАВКА:\n{context or '(нет)'}{findings}", temperature=0.2
    )
    return {
        "retrieved": {**state.retrieved, "howto": result},
        "drafts": {**state.drafts, "interface": answer.text},
    }


def data_gather(state: AgentState) -> dict:
    """Диагностика (read-only): собрать состояние системы под готовность размещения."""
    findings = [eva_list_unsigned_contracts(), eva_get_creative_status("CR-2")]
    return {"api_findings": state.api_findings + findings}


_CRITIC_SYS = (
    "Ты - критик-проверяющий ответа ИИ-помощника по рекламному праву (38-ФЗ). Оцени черновик: верен ли "
    "по сути, есть ли цитаты на нормы, нет ли явных юридических рисков. Верни СТРОГО JSON "
    '{"decision":"accept|rework","target":"legal_agent|interface_agent|data_gather|null","reason":"..."}.'
)


def critic(state: AgentState) -> dict:
    draft = "\n\n".join(f"[{key}] {value}" for key, value in state.drafts.items()) or "(пусто)"
    response = get_client("reasoning").invoke(_CRITIC_SYS, draft, temperature=0.0, json_mode=True)
    data = _safe_json(response.text)
    decision = "rework" if data.get("decision") == "rework" else "accept"
    target = data.get("target") if data.get("target") in _ROUTE_TARGETS else None
    verdict = CriticVerdict(
        decision=decision,
        target=target,
        reason=str(data.get("reason", "")),
    )
    return {"critic": verdict, "loop_count": state.loop_count + (1 if decision == "rework" else 0)}


def finalize(state: AgentState) -> dict:
    body = "\n\n".join(state.drafts.values()) or "Не удалось сформировать ответ."
    citations = list(dict.fromkeys(c for c in state.citations if c))
    if citations:
        body += "\n\nИсточники: " + ", ".join(citations)
    return {"final": body}


def clarify(state: AgentState) -> dict:
    needed = state.intent.needed_inputs if state.intent else []
    ask = "; ".join(needed) if needed else "уточните детали запроса"
    return {"final": f"Чтобы ответить точно, уточните, пожалуйста: {ask}."}


def refuse(state: AgentState) -> dict:
    if state.guard_in and state.guard_in.decision == "block":
        return {"final": "Запрос отклонен входным контролем безопасности."}
    return {
        "final": "Я - помощник маркетолога: отвечаю по закону о рекламе и помогаю с договорами, "
        "контрагентами и размещениями в вашей системе. Переформулируйте вопрос по теме."
    }
