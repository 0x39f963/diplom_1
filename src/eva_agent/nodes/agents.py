"""Узлы-агенты графа (ТЗ-2 §3): supervisor, консультанты, data_gather, critic, finalize, clarify, refuse."""

from __future__ import annotations

import json
from typing import Any

from eva_agent.domain.checklist import EntityCount, PlanningChecklist
from eva_agent.domain.plan import TodoPlan
from eva_agent.llm.config import get_client
from eva_agent.llm.observability import langfuse_enabled
from eva_agent.planner.build import build_plan, replan
from eva_agent.planner.compile import compile_plan
from eva_agent.planner.execute import execute_plan
from eva_agent.planner.trace import trace_plan
from eva_agent.security.spotlight import SPOTLIGHT_INSTRUCTION, spotlight
from eva_agent.settings import settings
from eva_agent.state import AgentState, ApiFinding, CriticVerdict, Intent
from eva_agent.tools.entity_ref import has_domain_signal
from eva_agent.tools.retrieve import retrieve_howto, retrieve_legal

_INTENT_KINDS = {
    "legal_consult", "interface_consult", "mixed_diagnostic", "need_clarification", "out_of_scope",
}
_ROUTE_TARGETS = {"legal_agent", "interface_agent", "data_gather"}
MAX_PLAN_REBUILDS = 1
_REBUILD_HINTS = (
    "rebuild",
    "replan",
    "missing step",
    "missing todo",
    "перестро",
    "переплан",
    "не хватает шага",
    "не хватает todo",
)


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
    "mixed_diagnostic - нужно посмотреть состояние данных системы: договоры, стороны, контрагенты, "
    "креативы, размещения или документы. Любой такой запрос классифицируй как mixed_diagnostic, "
    "даже если он сформулирован вопросом. Примеры: 'какой статус договора CT-1?' -> mixed_diagnostic; "
    "'кто заказчик по договору CT-1?' -> mixed_diagnostic; "
    "'почему креатив CR-1 не готов?' -> mixed_diagnostic. "
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
    intent = _apply_high_recall_override(query, intent)
    return {"intent": intent}


def _apply_high_recall_override(query: str, intent: Intent) -> Intent:
    if intent.kind not in {"need_clarification", "out_of_scope"} or not has_domain_signal(query):
        return intent
    original_kind = intent.kind
    rationale = intent.rationale.strip()
    marker = f"high-recall override: {original_kind}->mixed_diagnostic"
    intent = intent.model_copy(
        update={
            "kind": "mixed_diagnostic",
            "rationale": f"{rationale}; {marker}" if rationale else marker,
        }
    )
    _trace_supervisor_override(original_kind, intent.kind)
    return intent


def _trace_supervisor_override(original_kind: str, final_kind: str) -> None:
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client as get_langfuse_client

        get_langfuse_client().update_current_span(
            metadata={
                "supervisor": {
                    "high_recall_override": True,
                    "original_kind": original_kind,
                    "final_kind": final_kind,
                }
            }
        )
    except Exception:
        return


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
    """Диагностика (read-only): построить todo-план, исполнить его и собрать findings."""
    query = state.user_input_clean or state.user_input_raw
    plan, plan_attempts, source, rebuild_reason = _plan_for_state(state, query)

    current_findings: list[ApiFinding] = []
    all_findings = state.api_findings
    if plan.status != "awaiting_clarification":
        if source == "reuse" and state.api_findings:
            current_findings = state.api_findings
        else:
            current_findings, plan = execute_plan(
                plan,
                relations=state.domain_slice.relations if state.domain_slice else None,
            )
            all_findings = state.api_findings + current_findings

    trace_plan(
        plan,
        current_findings,
        plan_reused=source == "reuse",
        plan_attempts=plan_attempts,
        rebuild_reason=rebuild_reason,
    )

    checklist = _update_checklist(state.checklist, plan, all_findings)

    if plan.status == "awaiting_clarification":
        result = {
            "intent": _clarification_intent(state, plan),
            "todo_plan": plan,
            "plan_attempts": plan_attempts,
        }
        if checklist is not None:
            result["checklist"] = checklist
        return result
    result = {"api_findings": all_findings, "todo_plan": plan, "plan_attempts": plan_attempts}
    if checklist is not None:
        result["checklist"] = checklist
    return result


def _plan_for_state(state: AgentState, query: str) -> tuple[TodoPlan, int, str, str]:
    memory = getattr(state, "memory", None)
    if state.todo_plan is None and _use_protocol_compiler(state) and state.frame is not None:
        return (
            compile_plan(state.frame, domain_slice=state.domain_slice),
            state.plan_attempts,
            "compile",
            "",
        )
    if (
        state.todo_plan is None
        and memory is not None
        and memory.is_continuation
        and memory.prev_todo_plan is not None
    ):
        return (
            replan(memory.prev_todo_plan, query, resolved_inputs=memory.resolved_inputs),
            state.plan_attempts,
            "replan",
            "",
        )

    if state.todo_plan is None:
        return (
            build_plan(
                query,
                prior_meaning=_prior_meaning(state),
                domain_slice=state.domain_slice,
                intent_kind=state.intent.kind if state.intent else None,
            ),
            state.plan_attempts,
            "build",
            "",
        )

    reason = _explicit_rebuild_reason(state)
    if not reason or state.plan_attempts >= MAX_PLAN_REBUILDS:
        return state.todo_plan.model_copy(deep=True), state.plan_attempts, "reuse", ""
    if _use_protocol_compiler(state) and state.frame is not None:
        return (
            compile_plan(state.frame, domain_slice=state.domain_slice),
            state.plan_attempts + 1,
            "compile_rebuild",
            reason,
        )
    return (
        build_plan(
            query,
            prior_meaning=_prior_meaning(state),
            domain_slice=state.domain_slice,
            intent_kind=state.intent.kind if state.intent else None,
        ),
        state.plan_attempts + 1,
        "rebuild" if reason else "build",
        reason,
    )


def _use_protocol_compiler(state: AgentState) -> bool:
    return settings.planner_use_protocol_compiler and state.frame is not None


_FINDING_TOOL_ENTITY = {
    "eva_get_contract": "Contract",
    "eva_search_contracts": "Contract",
    "eva_list_unsigned_contracts": "Contract",
    "eva_get_contract_parties": "ContractParty",
    "eva_get_counterparty": "Counterparty",
    "eva_get_creative_status": "Creative",
    "eva_list_placements": "Placement",
    "eva_list_contract_documents": "Document",
    "eva_doc_read": "Document",
    "eva_doc_download": "Document",
}


def _update_checklist(
    checklist: PlanningChecklist | None,
    plan: TodoPlan,
    findings: list[ApiFinding],
) -> PlanningChecklist | None:
    if checklist is None:
        return None
    result_counts = _result_counts(findings)
    seen: set[str] = set()
    cardinality: list[EntityCount] = []
    for count in checklist.cardinality:
        seen.add(count.entity)
        cardinality.append(
            count.model_copy(update={"result_count": result_counts.get(count.entity, 0)})
        )
    for entity in checklist.entities:
        if entity not in seen:
            cardinality.append(EntityCount(entity=entity, result_count=result_counts.get(entity, 0)))

    resolution = "clarify" if plan.status == "awaiting_clarification" else "proceed"
    clarify_reason = _clarify_reason(plan) if resolution == "clarify" else ""
    return checklist.model_copy(
        update={
            "cardinality": cardinality,
            "needs_chain": checklist.needs_chain or _plan_needs_chain(plan),
            "resolution": resolution,
            "clarify_reason": clarify_reason,
        }
    )


def _result_counts(findings: list[ApiFinding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        entity = _FINDING_TOOL_ENTITY.get(finding.tool)
        if entity is None:
            continue
        counts[entity] = counts.get(entity, 0) + _finding_count(finding)
    return counts


def _finding_count(finding: ApiFinding) -> int:
    for value in finding.data.values():
        if isinstance(value, list):
            return len(value)
    return 1 if finding.data else 0


def _plan_needs_chain(plan: TodoPlan) -> bool:
    if any("auto-wire" in event for event in plan.trace):
        return True
    for todo in plan.items:
        if todo.depends_on or _has_from_ref(todo.inputs):
            return True
        if any(_has_from_ref(step.args) for step in todo.tool_calls):
            return True
    return False


def _has_from_ref(value: Any) -> bool:
    if isinstance(value, dict):
        if "$from" in value:
            return True
        return any(_has_from_ref(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_from_ref(child) for child in value)
    return False


def _clarify_reason(plan: TodoPlan) -> str:
    for todo in plan.ordered():
        if todo.blockers:
            return todo.blockers[0]
    return plan.clarify_question


def _explicit_rebuild_reason(state: AgentState) -> str:
    if state.todo_plan is None or state.critic is None or state.critic.target != "data_gather":
        return ""
    reason = state.critic.reason.strip()
    lowered = reason.lower()
    if any(hint in lowered for hint in _REBUILD_HINTS):
        return reason[:200]
    return ""


def _prior_meaning(state: AgentState) -> str:
    memory = getattr(state, "memory", None)
    if memory is None:
        return ""
    return memory.accumulated_meaning.strip()


def _clarification_intent(state: AgentState, plan: TodoPlan) -> Intent:
    question = plan.clarify_question or "Уточните, по какой сущности нужны данные."
    return Intent(
        kind="need_clarification",
        confidence=state.intent.confidence if state.intent else 0.0,
        rationale=state.intent.rationale if state.intent else "",
        needed_inputs=[question],
    )


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
    return {"final": f"Чтобы ответить точно, уточните, пожалуйста: {ask}.", "open_question": ask}


def refuse(state: AgentState) -> dict:
    if state.guard_in and state.guard_in.decision == "block":
        return {"final": "Запрос отклонен входным контролем безопасности."}
    return {
        "final": "Я - помощник маркетолога: отвечаю по закону о рекламе и помогаю с договорами, "
        "контрагентами и размещениями в вашей системе. Переформулируйте вопрос по теме."
    }
