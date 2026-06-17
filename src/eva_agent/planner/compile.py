"""Deterministic compiler from PlanningFrame to TodoPlan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from eva_agent.domain.frame import PlanningFrame
from eva_agent.domain.plan import DateHint, PlanStep, StatusHint, TodoItem, TodoPlan
from eva_agent.domain.slice import DomainSlice
from eva_agent.planner.catalog import CATALOG
from eva_agent.planner.protocols import PROTOCOL_CARDS, PROTOCOLS, ProtocolCard
from eva_agent.planner.validate import validate_plan

_INTERNAL_TODOS = frozenset({"parse_goal", "clarify", "summarize_answer"})
_DATE_HINTS: frozenset[str] = frozenset({"none", "yesterday", "last_week", "last_month"})
_STATUS_HINTS: frozenset[str] = frozenset({"none", "unsigned", "draft", "registered"})
_SLOT_ALIASES: dict[str, str] = {
    "contract": "contract_id",
    "contract_ref": "contract_id",
    "contract_number": "contract_id",
    "creative": "creative_id",
    "counterparty": "counterparty_id",
    "document": "doc_id",
    "document_id": "doc_id",
    "doc": "doc_id",
    "placement": "placement_id",
}


@dataclass(frozen=True)
class ProtocolRank:
    card: ProtocolCard
    score: int
    covered_slots: tuple[str, ...]
    missing_slots: tuple[str, ...]


def compile_plan(frame: PlanningFrame, domain_slice: DomainSlice | None = None) -> TodoPlan:
    """Build an executable TodoPlan from a typed semantic frame."""

    if frame.needs_clarification:
        return _clarify_plan(frame, frame.clarify_reason or "недостаточно входных данных")
    if not frame.target:
        return _clarify_plan(frame, "не определена целевая сущность")
    if frame.target not in {card.target for card in PROTOCOL_CARDS}:
        return _clarify_plan(frame, "нет покрытого протокола для целевой сущности")

    ranked = rank_protocol_cards(frame, domain_slice=domain_slice)
    if not ranked or ranked[0].score <= 0:
        return _clarify_plan(frame, "нет покрытого протокола для запроса")

    selected = ranked[0]
    slots = _slots(frame)
    ambiguity = _ambiguity_reason(frame, selected.card, slots)
    if ambiguity:
        return _clarify_plan(frame, ambiguity)

    plan = _build_plan(frame, selected.card, slots)
    plan.trace.extend(_rank_trace(ranked[:3], selected))
    return validate_plan(plan)


def compile_frame(frame: PlanningFrame, domain_slice: DomainSlice | None = None) -> TodoPlan:
    """Alias kept for call sites that name the source object explicitly."""

    return compile_plan(frame, domain_slice=domain_slice)


def rank_protocol_cards(
    frame: PlanningFrame,
    *,
    domain_slice: DomainSlice | None = None,
    cards: tuple[ProtocolCard, ...] = PROTOCOL_CARDS,
) -> list[ProtocolRank]:
    """Rank protocol cards by coverage of frame operation, target, relation and slots."""

    slots = _slots(frame)
    scoped_targets = set(domain_slice.entities) if domain_slice is not None else set()
    ranked = [_rank_card(card, frame, slots, scoped_targets) for card in cards]
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def _rank_card(
    card: ProtocolCard,
    frame: PlanningFrame,
    slots: dict[str, str],
    scoped_targets: set[str],
) -> ProtocolRank:
    covered = tuple(slot for slot in card.required_slots if _has_slot(slots, slot))
    missing = tuple(slot for slot in card.required_slots if not _has_slot(slots, slot))
    score = card.priority

    score += _operation_score(card.operation, frame)
    score += _target_score(card.target, frame.target, scoped_targets)
    score += _relation_score(card.relation, frame.relation)
    score += len(covered) * 12
    score -= len(missing) * 10

    if frame.cardinality == "all" and card.operation == "list":
        score += 16
    if frame.cardinality == "all" and card.id.endswith("_all"):
        score += 10
    if frame.cardinality == "one" and card.id.endswith("_role") and _has_slot(slots, "role"):
        score += 10
    if "unsigned" in frame.filters.status and card.id == "unsigned_overview":
        score += 18
    if "missing" in frame.fields and card.id == "missing_documents":
        score += 14

    return ProtocolRank(card=card, score=score, covered_slots=covered, missing_slots=missing)


def _operation_score(operation: str, frame: PlanningFrame) -> int:
    if operation == frame.operation:
        return 50
    if frame.cardinality == "all" and operation == "list":
        return 28
    if frame.operation == "diagnose" and operation == "read":
        return 18
    if frame.operation == "read" and operation == "diagnose":
        return 10
    return -35


def _target_score(target: str, frame_target: str, scoped_targets: set[str]) -> int:
    if target == frame_target:
        return 60
    if target in scoped_targets and not frame_target:
        return 12
    if target in scoped_targets:
        return 4
    return -55


def _relation_score(card_relation: str | None, frame_relation: str | None) -> int:
    if _norm(card_relation) == _norm(frame_relation):
        return 28 if card_relation else 10
    if card_relation and not frame_relation:
        return 0
    if not card_relation and frame_relation:
        return -12
    return -24


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def _has_slot(slots: dict[str, str], slot: str) -> bool:
    return slots.get(slot, "") != ""


def _slots(frame: PlanningFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_key, raw_value in frame.selector.items():
        value = str(raw_value).strip()
        if not value:
            continue
        key = _SLOT_ALIASES.get(str(raw_key).strip(), str(raw_key).strip())
        out[key] = value
    return out


def _ambiguity_reason(frame: PlanningFrame, card: ProtocolCard, slots: dict[str, str]) -> str:
    if (
        card.protocol_id == "party_lookup"
        and frame.cardinality == "one"
        and not _has_slot(slots, "role")
    ):
        return "нужно указать роль стороны: заказчик или исполнитель"
    return ""


def _build_plan(frame: PlanningFrame, card: ProtocolCard, slots: dict[str, str]) -> TodoPlan:
    items: list[TodoItem] = []
    step_order = 1
    for todo_order, todo_id in enumerate(_todo_template(card), start=1):
        item, step_order = _build_item(
            todo_id,
            todo_order=todo_order,
            step_order=step_order,
            frame=frame,
            card=card,
            slots=slots,
        )
        items.append(item)

    return TodoPlan(
        goal=_goal(frame),
        protocol_id=card.protocol_id,
        strategy=f"compiled:{card.id}",
        items=items,
        status="in_progress",
        confidence=max(frame.confidence, 0.75),
        trace=[f"compiler selected {card.id}"],
    )


def _todo_template(card: ProtocolCard) -> list[str]:
    template = list(dict.fromkeys(card.todo_template))
    mandatory = PROTOCOLS[card.protocol_id].mandatory
    for todo_id in mandatory:
        if todo_id not in template:
            insert_at = max(0, len(template) - 1) if todo_id == "summarize_answer" else len(template)
            template.insert(insert_at, todo_id)
    return template


def _build_item(
    todo_id: str,
    *,
    todo_order: int,
    step_order: int,
    frame: PlanningFrame,
    card: ProtocolCard,
    slots: dict[str, str],
) -> tuple[TodoItem, int]:
    inputs = _todo_inputs(todo_id, frame, slots)
    item_type = "dependent" if _is_dependent(todo_id, card) else "blocking"
    depends_on = _depends_on(todo_id, card)
    calls: list[PlanStep] = []

    spec = CATALOG.get(todo_id)
    if spec is not None and todo_id not in _INTERNAL_TODOS:
        for tool in spec.tools:
            calls.append(
                PlanStep(
                    order=step_order,
                    tool=tool,
                    args=_tool_args(todo_id, inputs),
                    date_hint=_date_hint(frame),
                    status_hint=_status_hint(frame),
                    reason=f"compiled:{card.id}:{todo_id}",
                )
            )
            step_order += 1

    return (
        TodoItem(
            id=todo_id,
            type=item_type,
            order=todo_order,
            depends_on=depends_on,
            inputs=inputs,
            tool_calls=calls,
        ),
        step_order,
    )


def _todo_inputs(todo_id: str, frame: PlanningFrame, slots: dict[str, str]) -> dict[str, Any]:
    spec = CATALOG.get(todo_id)
    if spec is None:
        return {}
    allowed = set(spec.inputs_required + spec.inputs_optional)
    inputs: dict[str, Any] = {key: slots[key] for key in allowed if _has_slot(slots, key)}
    if todo_id == "resolve_party_role" and _has_slot(slots, "role"):
        inputs["role"] = slots["role"]
    if _needs_fan_out(todo_id, frame):
        inputs["fan_out"] = True
    return inputs


def _tool_args(todo_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
    spec = CATALOG.get(todo_id)
    if spec is None:
        return {}
    allowed = set(spec.inputs_required + spec.inputs_optional)
    return {key: value for key, value in inputs.items() if key in allowed}


def _is_dependent(todo_id: str, card: ProtocolCard) -> bool:
    return bool(_depends_on(todo_id, card))


def _depends_on(todo_id: str, card: ProtocolCard) -> list[int]:
    if todo_id == "get_counterparty" and "resolve_party_role" in card.todo_template:
        return [_todo_order(card, "resolve_party_role")]
    if todo_id == "get_contract" and "get_creative_status" in card.todo_template:
        return [_todo_order(card, "get_creative_status")]
    return []


def _todo_order(card: ProtocolCard, todo_id: str) -> int:
    try:
        return _todo_template(card).index(todo_id) + 1
    except ValueError:
        return 0


def _needs_fan_out(todo_id: str, frame: PlanningFrame) -> bool:
    return todo_id == "get_counterparty" and frame.cardinality in {"all", "n"}


def _date_hint(frame: PlanningFrame) -> DateHint:
    hint = frame.filters.date_hint
    return cast(DateHint, hint) if hint in _DATE_HINTS else "none"


def _status_hint(frame: PlanningFrame) -> StatusHint:
    for status in frame.filters.status:
        normalized = status.strip().lower()
        if normalized in _STATUS_HINTS:
            return cast(StatusHint, normalized)
        if normalized in {"unregistered", "unsigned", "not_registered"}:
            return "unsigned"
    return "none"


def _goal(frame: PlanningFrame) -> str:
    relation = f".{frame.relation}" if frame.relation else ""
    return f"{frame.operation}:{frame.target}{relation}".strip(":")


def _clarify_plan(frame: PlanningFrame, reason: str) -> TodoPlan:
    question = reason if reason.endswith("?") else f"Уточните входные данные: {reason}."
    return TodoPlan(
        goal=_goal(frame) or "clarify",
        protocol_id="clarify_first",
        strategy="compiled:clarify",
        status="awaiting_clarification",
        confidence=min(frame.confidence, 0.4),
        clarify_question=question,
        items=[
            TodoItem(id="parse_goal", order=1),
            TodoItem(id="clarify", order=2, blockers=[reason]),
        ],
        trace=["compiler clarify", reason],
    )


def _rank_trace(ranked: list[ProtocolRank], selected: ProtocolRank) -> list[str]:
    events = [f"compiler rank {item.card.id}={item.score}" for item in ranked]
    events.append(
        "compiler slots "
        f"covered={','.join(selected.covered_slots) or '-'} "
        f"missing={','.join(selected.missing_slots) or '-'}"
    )
    return events


__all__ = [
    "ProtocolRank",
    "compile_frame",
    "compile_plan",
    "rank_protocol_cards",
]
