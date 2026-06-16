"""LLM builder for planner todo plans."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from eva_agent.domain.entity_map import render_entity_map
from eva_agent.domain.plan import TodoItem, TodoPlan
from eva_agent.llm.config import get_client
from eva_agent.planner.catalog import AVAILABLE_TOOLS_DEFAULT, CATALOG, render_catalog
from eva_agent.planner.protocols import ProtocolId, render_protocol, select_protocol
from eva_agent.planner.validate import PLANNER_MIN_CONFIDENCE, validate_plan
from eva_agent.tools.entity_ref import EntityRefs, extract_refs

_PLANNER_SYS = """Ты - планировщик ИИ-помощника по работе с внутренней системой маркетингового агентства.
Твоя задача: разобрать запрос пользователя и построить todo-лист - упорядоченный план решения цели.
Ты НЕ отвечаешь пользователю и НЕ исполняешь шаги, ты только строишь план. Данные только на чтение.

ЦЕЛЬ:
{{GOAL}}

СМЫСЛ ПРОШЛОГО РАУНДА:
{{PRIOR_MEANING}}

РАСПОЗНАННЫЕ ССЫЛКИ:
{{REFS}}

ПРОТОКОЛ (обязательный чек-лист шагов под этот тип задачи, обязательные шаги пропускать нельзя):
{{PROTOCOL}}

КАТАЛОГ ДОСТУПНЫХ TODO (бери ТОЛЬКО эти id, других нет; каждый todo привязан к инструменту):
{{CATALOG}}

КАРТА СУЩНОСТЕЙ И СВЯЗЕЙ (источник истины, опирайся только на нее):
{{ENTITY_MAP}}

КАК СТРОИТЬ TODO-ЛИСТ:
- Сначала parse_goal, затем обязательные шаги протокола в порядке, в конце summarize_answer.
- Каждый todo: id из каталога, type (blocking, non_blocking, dependent), order по порядку, inputs,
  blockers и при необходимости tool_calls.
- Если у обязательного шага не хватает входа, не выдумывай id: добавь todo clarify, опиши blocker,
  задай clarify_question, status = awaiting_clarification.

КАК ПЕРЕДАВАТЬ РЕЗУЛЬТАТ МЕЖДУ ШАГАМИ:
В tool_calls аргумент из вывода прошлого шага задавай ссылкой:
{"$from": {"step": <номер прошлого шага>, "path": "<путь в его данных>"}}.

ФИЛЬТРЫ:
Не вычисляй даты и не перечисляй статусы сам. В tool_calls ставь только метки:
- date_hint: one of none|yesterday|last_week|last_month.
- status_hint: one of none|unsigned|draft|registered.

ФОРМАТ ОТВЕТА - СТРОГО JSON, без пояснений вокруг:
{
  "goal": "<глобальный смысл запроса>",
  "protocol_id": "<id протокола из чек-листа выше>",
  "strategy": "<кратко как решаешь>",
  "items": [
    {"id": "<todo_id из каталога>", "type": "blocking|non_blocking|dependent", "order": 1,
     "depends_on": [], "inputs": {}, "blockers": [],
     "tool_calls": [{"order": 1, "tool": "<имя инструмента>", "args": {},
                     "date_hint": "none", "status_hint": "none", "reason": "..."}]}
  ],
  "status": "answered|awaiting_clarification|in_progress",
  "confidence": 0.0,
  "clarify_question": ""
}
order todo начинается с 1. id строго из каталога. Если решение исполнимо, clarify_question пустой.
Если обязательного входа нет, добавь обязательный todo clarify, status = awaiting_clarification,
задай clarify_question коротким вопросом."""


def build_plan(query: str, *, prior_meaning: str = "") -> TodoPlan:
    """Build a todo plan with one planner LLM call."""

    refs = extract_refs(query)
    protocol_id = select_protocol(query, has_entity=refs.has_any)
    system = _build_system_prompt(protocol_id, query=query, refs=refs, prior_meaning=prior_meaning)
    response = get_client("planner").invoke(
        system,
        "Построй JSON TodoPlan для цели из системного сообщения.",
        temperature=0.0,
        json_mode=True,
    )
    plan = _parse_plan(response.text, protocol_id=protocol_id, query=query)
    _fill_missing_inputs(plan, refs=refs, query=query)
    return validate_plan(plan)


def replan(
    prev_plan: TodoPlan,
    new_message: str,
    *,
    resolved_inputs: dict[str, str] | None = None,
) -> TodoPlan:
    """Update a previous plan after a clarification message."""

    plan = prev_plan.model_copy(deep=True)
    refs = extract_refs(new_message)
    inputs = _inputs_from_refs(refs, query=new_message)
    if resolved_inputs:
        inputs.update(resolved_inputs)

    for todo in plan.items:
        _apply_inputs(todo, inputs)
        if todo.result_ref:
            todo.status = "done"
            continue
        if todo.status == "skipped":
            continue
        todo.blockers = _remaining_blockers(todo, inputs)
        todo.status = "blocked" if todo.blockers else "pending"

    plan.status = "in_progress"
    plan.clarify_question = ""
    return validate_plan(plan)


def _build_system_prompt(
    protocol_id: ProtocolId,
    *,
    query: str,
    refs: EntityRefs,
    prior_meaning: str,
) -> str:
    prior = prior_meaning.strip() if prior_meaning.strip() else "-"
    return (
        _PLANNER_SYS.replace("{{GOAL}}", query)
        .replace("{{PRIOR_MEANING}}", prior)
        .replace("{{REFS}}", _render_refs(refs))
        .replace("{{PROTOCOL}}", render_protocol(protocol_id))
        .replace("{{CATALOG}}", render_catalog(AVAILABLE_TOOLS_DEFAULT))
        .replace("{{ENTITY_MAP}}", render_entity_map())
    )


def _parse_plan(text: str, *, protocol_id: ProtocolId, query: str) -> TodoPlan:
    empty = _fallback_plan(query=query, protocol_id=protocol_id)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return empty
    if not isinstance(raw, dict):
        return empty
    raw.setdefault("goal", query)
    raw.setdefault("protocol_id", protocol_id)
    try:
        return TodoPlan.model_validate(raw)
    except ValidationError:
        return empty


def _fallback_plan(*, query: str, protocol_id: ProtocolId) -> TodoPlan:
    return TodoPlan(
        goal=query,
        protocol_id=protocol_id,
        status="awaiting_clarification",
        confidence=0.0,
        clarify_question="Уточните цель запроса.",
    )


def _fill_missing_inputs(plan: TodoPlan, *, refs: EntityRefs, query: str) -> None:
    inputs = _inputs_from_refs(refs, query=query)
    for todo in plan.items:
        _apply_inputs(todo, inputs)


def _apply_inputs(todo: TodoItem, inputs: dict[str, Any]) -> None:
    spec = CATALOG.get(todo.id)
    if spec is None:
        return
    allowed = set(spec.inputs_required + spec.inputs_optional)
    for key in allowed:
        value = inputs.get(key)
        if value in (None, ""):
            continue
        if not _has_value(todo.inputs.get(key)):
            todo.inputs[key] = value
        for step in todo.tool_calls:
            if not _has_value(step.args.get(key)):
                step.args[key] = value


def _inputs_from_refs(refs: EntityRefs, *, query: str) -> dict[str, Any]:
    inputs: dict[str, Any] = {"query": query}
    if refs.primary_contract:
        inputs["contract_id"] = refs.primary_contract
    if refs.primary_creative:
        inputs["creative_id"] = refs.primary_creative
    if refs.primary_counterparty:
        inputs["counterparty_id"] = refs.primary_counterparty
    if refs.document_ids:
        inputs["doc_id"] = refs.document_ids[0]
    if refs.placement_ids:
        inputs["placement_id"] = refs.placement_ids[0]
    role = _role_from_text(query)
    if role:
        inputs["role"] = role
    return inputs


def _role_from_text(text: str) -> str:
    lowered = text.lower()
    if "заказчик" in lowered:
        return "customer"
    if "исполнитель" in lowered:
        return "executor"
    return ""


def _remaining_blockers(todo: TodoItem, inputs: dict[str, Any]) -> list[str]:
    spec = CATALOG.get(todo.id)
    required = spec.inputs_required if spec is not None else []
    if any(not _has_required(todo, key) for key in required):
        return todo.blockers
    return [
        blocker
        for blocker in todo.blockers
        if not any(_blocker_matches_input(blocker, key) for key in inputs)
    ]


def _has_required(todo: TodoItem, key: str) -> bool:
    if _has_value(todo.inputs.get(key)):
        return True
    return any(_has_value(step.args.get(key)) for step in todo.tool_calls)


def _blocker_matches_input(blocker: str, key: str) -> bool:
    return key in blocker or blocker == f"нет {key}" or blocker == f"missing input: {key}"


def _has_value(value: Any) -> bool:
    return value not in (None, "")


def _render_refs(refs: EntityRefs) -> str:
    return json.dumps(
        {
            "contract_ids": refs.contract_ids,
            "creative_ids": refs.creative_ids,
            "counterparty_ids": refs.counterparty_ids,
            "document_ids": refs.document_ids,
            "placement_ids": refs.placement_ids,
            "contract_numbers": refs.contract_numbers,
            "counterparty_hints": refs.counterparty_hints,
        },
        ensure_ascii=False,
    )


__all__ = ["PLANNER_MIN_CONFIDENCE", "build_plan", "replan"]
