"""Deterministic compiler from PlanningFrame to TodoPlan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from eva_agent.domain.frame import PlanningFrame
from eva_agent.domain.plan import DateHint, PlanStep, StatusHint, TodoItem, TodoPlan
from eva_agent.domain.slice import DomainSlice
from eva_agent.planner.catalog import CATALOG
from eva_agent.planner.protocols import PROTOCOL_CARDS, PROTOCOLS, ProtocolCard
from eva_agent.planner.validate import validate_plan

_INTERNAL_TODOS = frozenset({"parse_goal", "clarify", "resolve_party_role", "summarize_answer"})
_DATE_HINTS: frozenset[str] = frozenset({"none", "yesterday", "last_week", "last_month"})
_STATUS_HINTS: frozenset[str] = frozenset({"none", "unsigned", "draft", "registered"})
TAU_CLARIFY = 0.45
_WRITE_OPERATIONS = frozenset({"attach", "download"})
_COMPOSITE_LIMIT = 5
_OVERVIEW_DATA_CARDS = frozenset({"unsigned_overview", "readiness_overview"})
_SEARCH_CARDS = frozenset({"contract_search_filtered", "contract_search_read"})
_SPECIFIC_RELATIONS = frozenset({"parties", "documents", "placements", "creative"})
_SPECIFIC_SELECTOR_SLOTS = frozenset(
    {"contract_id", "creative_id", "doc_id", "counterparty_id", "placement_id"}
)
_SPECIFIC_OPERATIONS = frozenset({"compare", "open", "download", "attach"})
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
_SLOT_LABELS: dict[str, str] = {
    "contract_id": "договор",
    "creative_id": "креатив",
    "counterparty_id": "контрагента",
    "doc_id": "документ",
    "placement_id": "размещение",
    "role": "роль стороны",
}

ClarifyCode = Literal[
    "unknown_target",
    "missing_selector",
    "ambiguous_role",
    "ambiguous_protocol",
    "write_confirm",
    "low_confidence",
]


@dataclass(frozen=True)
class ProtocolRank:
    card: ProtocolCard
    score: int
    covered_slots: tuple[str, ...]
    missing_slots: tuple[str, ...]


def compile_plan(frame: PlanningFrame, domain_slice: DomainSlice | None = None) -> TodoPlan:
    """Build an executable TodoPlan from a typed semantic frame."""

    if frame.subtasks:
        composite = _compile_composite_plan(frame, domain_slice=domain_slice)
        if composite is not None:
            return composite
    return _compile_single_plan(frame, domain_slice=domain_slice)


def _compile_single_plan(
    frame: PlanningFrame,
    *,
    domain_slice: DomainSlice | None = None,
    validate: bool = True,
) -> TodoPlan:
    if frame.needs_clarification:
        if _can_compile_global_overview(frame):
            frame = _without_clarification(frame, "clarify overridden: global overview")
        else:
            plan = _clarify_override_plan(
                frame,
                domain_slice=domain_slice,
                validate=validate,
            )
            if plan is not None:
                return plan
            return _clarify_plan(
                frame,
                frame.clarify_reason or "недостаточно входных данных",
                code=_code_from_reason(frame.clarify_reason),
            )
    return _compile_resolved_plan(frame, domain_slice=domain_slice, validate=validate)


def _compile_resolved_plan(
    frame: PlanningFrame,
    *,
    domain_slice: DomainSlice | None,
    validate: bool,
) -> TodoPlan:
    slots = _slots(frame)
    frame = _missing_documents_frame(frame, slots)
    frame = _roleless_party_all_frame(frame, slots)
    slots = _slots(frame)

    if not frame.target:
        return _clarify_plan(frame, "не определена целевая сущность", code="unknown_target")
    if frame.target not in {card.target for card in PROTOCOL_CARDS}:
        fallback = _fallback_read_plan(frame, domain_slice=domain_slice)
        if fallback is not None:
            return fallback
        return _clarify_plan(
            frame,
            "нет покрытого протокола для целевой сущности",
            code="ambiguous_protocol",
        )

    ranked = rank_protocol_cards(frame, domain_slice=domain_slice)
    if not ranked or ranked[0].score <= 0:
        fallback = _fallback_read_plan(frame, domain_slice=domain_slice)
        if fallback is not None:
            return fallback
        return _clarify_plan(
            frame,
            "нет покрытого протокола для запроса",
            code="ambiguous_protocol",
        )

    selected = ranked[0]
    ambiguity = _ambiguity_reason(frame, selected.card, slots)
    if ambiguity:
        return _clarify_plan(frame, ambiguity, code="ambiguous_role", missing_slot="role")
    write_slot = _missing_write_slot(frame, selected)
    if write_slot:
        return _clarify_plan(
            frame,
            f"нет {write_slot}",
            code="write_confirm",
            missing_slot=write_slot,
        )
    low_confidence_overridden = False
    if frame.confidence < TAU_CLARIFY:
        if not _has_domain_plan_signal(frame, slots):
            return _clarify_plan(frame, "низкая уверенность разбора", code="low_confidence")
        low_confidence_overridden = True

    plan = _build_plan(frame, selected.card, slots)
    if low_confidence_overridden:
        plan.confidence = max(plan.confidence, TAU_CLARIFY)
        plan.trace.append("clarify overridden: low confidence with domain signal")
    plan.trace.extend(_rank_trace(ranked[:3], selected))
    if validate:
        plan = validate_plan(plan)
        _ensure_clarify_code(plan, frame, selected)
    return plan


def _clarify_override_plan(
    frame: PlanningFrame,
    *,
    domain_slice: DomainSlice | None,
    validate: bool,
) -> TodoPlan | None:
    slots = _slots(frame)
    if not _can_try_clarify_override(frame, slots):
        return None
    candidate = _without_clarification(frame, "clarify overridden: safe read-only plan exists")
    plan = _compile_resolved_plan(candidate, domain_slice=domain_slice, validate=validate)
    if _plan_is_usable(plan):
        plan.trace.append("clarify overridden: safe read-only plan exists")
        return plan
    fallback = _fallback_read_plan(candidate, domain_slice=domain_slice)
    if fallback is not None and _plan_is_usable(fallback):
        fallback.trace.append("clarify overridden: safe read-only plan exists")
        return fallback
    return None


def _without_clarification(frame: PlanningFrame, event: str) -> PlanningFrame:
    trace = list(frame.trace)
    trace.append(event)
    return frame.model_copy(
        update={
            "needs_clarification": False,
            "clarify_reason": "",
            "trace": _unique(trace),
        }
    )


def _plan_is_usable(plan: TodoPlan) -> bool:
    return plan.status != "awaiting_clarification" and not plan.is_empty


def _can_try_clarify_override(frame: PlanningFrame, slots: dict[str, str]) -> bool:
    if _invalid_clarify_reason(frame, slots):
        return True
    return _has_domain_plan_signal(frame, slots)


def _invalid_clarify_reason(frame: PlanningFrame, slots: dict[str, str]) -> bool:
    reason = frame.clarify_reason.lower()
    if "роль" in reason and frame.relation == "parties" and frame.cardinality in {"all", "n"}:
        return True
    if ("id" in reason or "договор" in reason) and frame.operation == "list" and frame.cardinality == "all":
        return True
    if ("id" in reason or "договор" in reason) and _has_search_signal(frame, slots):
        return True
    return ("увер" in reason or "confidence" in reason) and _has_explicit_selector(slots)


def _has_domain_plan_signal(frame: PlanningFrame, slots: dict[str, str]) -> bool:
    if _has_explicit_selector(slots):
        return True
    if _norm(frame.relation) in _SPECIFIC_RELATIONS and _has_slot(slots, "contract_id"):
        return True
    if _has_search_signal(frame, slots):
        return True
    return frame.operation == "list" and frame.cardinality == "all"


def _has_explicit_selector(slots: dict[str, str]) -> bool:
    return any(_has_slot(slots, slot) for slot in _SPECIFIC_SELECTOR_SLOTS)


def compile_frame(frame: PlanningFrame, domain_slice: DomainSlice | None = None) -> TodoPlan:
    """Alias kept for call sites that name the source object explicitly."""

    return compile_plan(frame, domain_slice=domain_slice)


def _compile_composite_plan(
    frame: PlanningFrame,
    *,
    domain_slice: DomainSlice | None = None,
) -> TodoPlan | None:
    frames = _composite_frames(frame)
    if len(frames) < 2 or len(frames) > _COMPOSITE_LIMIT:
        return None
    plans: list[TodoPlan] = []
    parent_slots: dict[str, str] = {}
    for item in frames:
        subframe = _canonical_composite_frame(item, parent_slots=parent_slots)
        parent_slots.update(_slots(subframe))
        plan = _compile_single_plan(subframe, domain_slice=domain_slice, validate=False)
        if plan.status == "awaiting_clarification" or plan.is_empty:
            return None
        plans.append(plan)
    merged = _merge_composite_plans(frame, plans)
    return validate_plan(merged) if merged is not None else None


def _composite_frames(frame: PlanningFrame) -> list[PlanningFrame]:
    subtasks = []
    for item in frame.subtasks:
        subtasks.extend(_flatten_subtasks(item))
    head = frame.model_copy(update={"subtasks": []})
    if subtasks and _same_task(head, subtasks[0]):
        return subtasks
    return [head, *subtasks]


def _flatten_subtasks(frame: PlanningFrame) -> list[PlanningFrame]:
    current = frame.model_copy(update={"subtasks": []})
    out = [current]
    for item in frame.subtasks:
        out.extend(_flatten_subtasks(item))
    return out


def _canonical_composite_frame(
    frame: PlanningFrame,
    *,
    parent_slots: dict[str, str],
) -> PlanningFrame:
    selector = dict(frame.selector)
    for raw_key, raw_value in list(selector.items()):
        alias = _SLOT_ALIASES.get(str(raw_key).strip())
        if alias and alias not in selector:
            selector[alias] = raw_value
    for slot, value in parent_slots.items():
        if slot in _SPECIFIC_SELECTOR_SLOTS and slot not in selector:
            selector[slot] = value

    operation = "read" if frame.operation == "open" and frame.target == "Contract" else frame.operation
    cardinality = "all" if frame.cardinality == "n" else frame.cardinality
    relation = _canonical_relation(frame.target, frame.relation)
    return frame.model_copy(
        update={
            "operation": operation,
            "cardinality": cardinality,
            "relation": relation,
            "selector": selector,
            "subtasks": [],
        }
    )


def _canonical_relation(target: str, relation: str | None) -> str | None:
    normalized = _norm(relation)
    if target == "ContractParty" and normalized in {"", "contract_id", "entity_id", "contracts"}:
        return "parties"
    if target == "Counterparty" and normalized in {"contract_id", "contract_parties", "contracts"}:
        return "parties"
    if target == "Document" and normalized in {"", "contract_id", "entity_id", "attachments"}:
        return "documents"
    if target in {"Placement", "Creative"} and normalized in {"contract_id", "entity_id"}:
        return "placements"
    return relation


def _roleless_party_all_frame(frame: PlanningFrame, slots: dict[str, str]) -> PlanningFrame:
    if _norm(frame.relation) != "parties":
        return frame
    if _has_slot(slots, "role") or not _has_slot(slots, "contract_id"):
        return frame
    if frame.target not in {"Contract", "ContractParty", "Counterparty"}:
        return frame
    if frame.operation == "list" and frame.cardinality == "all":
        return frame
    trace = list(frame.trace)
    trace.append("compiler party without role: all parties")
    target = "ContractParty" if frame.target == "Contract" else frame.target
    return frame.model_copy(
        update={
            "operation": "list",
            "target": target,
            "cardinality": "all",
            "output": "list",
            "trace": _unique(trace),
        }
    )


def _missing_documents_frame(frame: PlanningFrame, slots: dict[str, str]) -> PlanningFrame:
    if _norm(frame.relation) != "documents":
        return frame
    if "missing" not in frame.fields or not _has_slot(slots, "contract_id") or _has_slot(slots, "doc_id"):
        return frame
    if frame.operation == "diagnose" and frame.cardinality == "all":
        return frame
    trace = list(frame.trace)
    trace.append("compiler documents missing: diagnose all")
    return frame.model_copy(
        update={
            "operation": "diagnose",
            "cardinality": "all",
            "output": "list",
            "trace": _unique(trace),
        }
    )


def _same_task(left: PlanningFrame, right: PlanningFrame) -> bool:
    return (
        left.operation == right.operation
        and left.target == right.target
        and left.relation == right.relation
        and left.cardinality == right.cardinality
        and left.selector == right.selector
    )


def _merge_composite_plans(frame: PlanningFrame, plans: list[TodoPlan]) -> TodoPlan | None:
    items: list[TodoItem] = []
    seen_data_todos: set[str] = set()
    todo_order = 1
    step_order = 1
    last_data_order: int | None = None
    trace: list[str] = ["compiler composite"]

    for plan_index, plan in enumerate(plans):
        plan_items, step_order = _copy_plan_items(
            plan,
            plan_index=plan_index,
            todo_order=todo_order,
            step_order=step_order,
            last_data_order=last_data_order,
            seen_data_todos=seen_data_todos,
        )
        if plan_items is None:
            return None
        items.extend(plan_items)
        todo_order += len(plan_items)
        data_orders = [item.order for item in plan_items if item.tool_calls and item.id not in _INTERNAL_TODOS]
        if data_orders:
            last_data_order = data_orders[-1]
        trace.extend(plan.trace)

    if not any(item.id == "summarize_answer" for item in items):
        depends_on = [last_data_order] if last_data_order is not None else []
        items.append(
            TodoItem(
                id="summarize_answer",
                type="dependent" if depends_on else "blocking",
                order=todo_order,
                depends_on=depends_on,
            )
        )

    return TodoPlan(
        goal=_goal(frame),
        protocol_id=plans[0].protocol_id,
        strategy="compiled:composite",
        items=items,
        status="in_progress",
        confidence=min(plan.confidence for plan in plans),
        clarify_code="",
        trace=_unique(trace),
    )


def _copy_plan_items(
    plan: TodoPlan,
    *,
    plan_index: int,
    todo_order: int,
    step_order: int,
    last_data_order: int | None,
    seen_data_todos: set[str],
) -> tuple[list[TodoItem] | None, int]:
    copied: list[tuple[TodoItem, list[int], bool]] = []
    old_to_new: dict[int, int] = {}
    first_data_seen = False
    next_todo_order = todo_order
    next_step_order = step_order

    for source in plan.ordered():
        if source.id == "summarize_answer":
            continue
        if source.id == "parse_goal" and (plan_index > 0 or next_todo_order > 1):
            continue
        if source.id not in _INTERNAL_TODOS:
            if source.id in seen_data_todos:
                return None, step_order
            seen_data_todos.add(source.id)

        calls: list[PlanStep] = []
        for call in sorted(source.tool_calls, key=lambda item: item.order):
            calls.append(call.model_copy(update={"order": next_step_order}))
            next_step_order += 1
        is_first_data = bool(source.tool_calls and source.id not in _INTERNAL_TODOS and not first_data_seen)
        first_data_seen = first_data_seen or is_first_data
        copied_item = source.model_copy(
            deep=True,
            update={"order": next_todo_order, "tool_calls": calls},
        )
        old_to_new[source.order] = next_todo_order
        copied.append((copied_item, list(source.depends_on), is_first_data))
        next_todo_order += 1

    out: list[TodoItem] = []
    for item, old_depends, is_first_data in copied:
        depends = [old_to_new[value] for value in old_depends if value in old_to_new]
        if is_first_data and plan_index > 0 and last_data_order is not None:
            depends.append(last_data_order)
        depends = list(dict.fromkeys(depends))
        item = item.model_copy(
            update={
                "depends_on": depends,
                "type": "dependent" if depends else item.type,
            }
        )
        out.append(item)
    return out, next_step_order


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
    if missing and not slots:
        score -= len(missing) * 45

    if frame.cardinality == "all" and card.operation == "list":
        score += 16
    if frame.cardinality == "all" and card.id.endswith("_all"):
        score += 10
    if frame.cardinality == "one" and card.id.endswith("_role") and _has_slot(slots, "role"):
        score += 10
    if "unsigned" in frame.filters.status and card.id == "unsigned_overview":
        score += 18
    if frame.filters.date_hint != "none" and card.id == "unsigned_overview":
        score -= 55
    if "missing" in frame.fields and card.id == "missing_documents":
        score += 14
    if "missing" in frame.fields and card.id == "readiness_overview":
        score += 40
    if card.id in _SEARCH_CARDS:
        score += 45 if _has_search_signal(frame, slots) else -45
    if card.id in _OVERVIEW_DATA_CARDS and _has_specific_signal(frame, slots):
        score -= 40

    return ProtocolRank(card=card, score=score, covered_slots=covered, missing_slots=missing)


def _has_specific_signal(frame: PlanningFrame, slots: dict[str, str]) -> bool:
    if _norm(frame.relation) in _SPECIFIC_RELATIONS:
        return True
    if any(_has_slot(slots, slot) for slot in _SPECIFIC_SELECTOR_SLOTS):
        return True
    if frame.operation in _SPECIFIC_OPERATIONS:
        return True
    return bool(frame.subtasks)


def _has_search_signal(frame: PlanningFrame, slots: dict[str, str]) -> bool:
    if frame.filters.date_hint != "none":
        return True
    if any(_has_slot(slots, key) for key in ("query", "search_query", "name", "title")):
        return True
    if frame.cardinality == "one" and not _has_slot(slots, "contract_id"):
        return any(field in {"status", "latest", "last"} for field in frame.fields)
    return False


def _can_compile_global_overview(frame: PlanningFrame) -> bool:
    if frame.target != "Contract" or frame.relation:
        return False
    if frame.selector or frame.cardinality != "all":
        return False
    if frame.operation == "list":
        return True
    return _has_overview_scope_signal(frame)


def _has_overview_scope_signal(frame: PlanningFrame) -> bool:
    if frame.filters.date_hint != "none" or frame.filters.status:
        return True
    return any(field in {"missing", "readiness", "ready", "status"} for field in frame.fields)


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


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


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
    del frame, card, slots
    return ""


def _missing_write_slot(frame: PlanningFrame, selected: ProtocolRank) -> str:
    if frame.operation not in _WRITE_OPERATIONS:
        return ""
    for slot in selected.missing_slots:
        return slot
    return ""


def _fallback_read_plan(
    frame: PlanningFrame,
    *,
    domain_slice: DomainSlice | None,
) -> TodoPlan | None:
    slots = _slots(frame)
    for operation in ("read", "list"):
        cardinality = "all" if operation == "list" else frame.cardinality
        candidate = frame.model_copy(
            update={
                "operation": operation,
                "cardinality": cardinality,
                "subtasks": [],
                "needs_clarification": False,
                "clarify_reason": "",
            }
        )
        ranked = rank_protocol_cards(candidate, domain_slice=domain_slice)
        if not ranked or ranked[0].score <= 0:
            continue
        selected = ranked[0]
        if selected.card.target != frame.target:
            continue
        if any(not _has_slot(slots, slot) for slot in selected.card.required_slots):
            continue
        plan = _build_plan(candidate, selected.card, slots)
        plan.trace.append("compiler fallback: minimal read")
        if plan.confidence < TAU_CLARIFY and _has_domain_plan_signal(candidate, slots):
            plan.confidence = max(plan.confidence, TAU_CLARIFY)
            plan.trace.append("clarify overridden: low confidence with domain signal")
        plan.trace.extend(_rank_trace(ranked[:3], selected))
        return validate_plan(plan)
    return None


def _ensure_clarify_code(
    plan: TodoPlan,
    frame: PlanningFrame,
    selected: ProtocolRank,
) -> None:
    if plan.status != "awaiting_clarification" or plan.clarify_code:
        return
    if _ambiguity_reason(frame, selected.card, _slots(frame)):
        plan.clarify_code = "ambiguous_role"
        return
    if selected.missing_slots:
        plan.clarify_code = "missing_selector"
        return
    plan.clarify_code = "low_confidence" if plan.confidence < TAU_CLARIFY else "ambiguous_protocol"


def _code_from_reason(reason: str) -> ClarifyCode:
    lowered = reason.lower()
    if "сущ" in lowered or "target" in lowered:
        return "unknown_target"
    if "роль" in lowered:
        return "ambiguous_role"
    if "увер" in lowered:
        return "low_confidence"
    return "missing_selector"


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
        confidence=max(0.0, min(1.0, frame.confidence)),
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
                    args=_tool_args(todo_id, inputs, frame=frame, card=card, slots=slots),
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
    if todo_id == "search_contracts":
        inputs["query"] = _search_query(frame, slots)
    if _needs_fan_out(todo_id, frame):
        inputs["fan_out"] = True
    return inputs


def _tool_args(
    todo_id: str,
    inputs: dict[str, Any],
    *,
    frame: PlanningFrame,
    card: ProtocolCard,
    slots: dict[str, str],
) -> dict[str, Any]:
    spec = CATALOG.get(todo_id)
    if spec is None:
        return {}
    allowed = set(spec.inputs_required + spec.inputs_optional)
    args = {key: value for key, value in inputs.items() if key in allowed}
    if todo_id == "search_contracts" and "query" in args:
        return {"q": args["query"]}
    if todo_id == "attach_document":
        args.setdefault("file", _attachment_file_arg(inputs))
    if todo_id == "get_counterparty" and "counterparty_id" not in args:
        ref = _counterparty_ref(frame, card, slots)
        if ref:
            args["counterparty_id"] = {"$from": ref}
    return args


def _attachment_file_arg(inputs: dict[str, Any]) -> dict[str, str]:
    doc_id = str(inputs.get("doc_id") or "document").strip() or "document"
    doc_type = str(inputs.get("doc_type") or "annex").strip() or "annex"
    return {
        "file_name": f"{doc_id}.txt",
        "content_b64": "",
        "mime_type": "text/plain",
        "doc_type": doc_type,
    }


def _counterparty_ref(
    frame: PlanningFrame,
    card: ProtocolCard,
    slots: dict[str, str],
) -> dict[str, Any]:
    if "get_contract_parties" not in _todo_template(card):
        return {}
    ref: dict[str, Any] = {
        "todo": "get_contract_parties",
        "path": "parties[].counterparty_id",
        "selector": "role",
        "cardinality": "many",
    }
    if _has_slot(slots, "role"):
        ref["selector_value"] = slots["role"]
    elif frame.cardinality in {"all", "n"}:
        ref["fan_out"] = True
    return ref


def _search_query(frame: PlanningFrame, slots: dict[str, str]) -> str:
    for key in ("query", "search_query", "name", "title"):
        if _has_slot(slots, key):
            return slots[key]
    if _has_slot(slots, "contract_id") and not slots["contract_id"].startswith("CT-"):
        return slots["contract_id"]

    parts: list[str] = []
    if frame.filters.date_hint == "yesterday":
        parts.append("вчера")
    elif frame.filters.date_hint == "last_week":
        parts.append("за последнюю неделю")
    elif frame.filters.date_hint == "last_month":
        parts.append("за последний месяц")
    if "unsigned" in frame.filters.status:
        parts.append("неподписанные")
    if any(field in {"latest", "last", "status"} for field in frame.fields):
        parts.append("последний")
    parts.append("договор")
    return " ".join(parts)


def _is_dependent(todo_id: str, card: ProtocolCard) -> bool:
    return bool(_depends_on(todo_id, card))


def _depends_on(todo_id: str, card: ProtocolCard) -> list[int]:
    if todo_id == "resolve_party_role" and "get_contract_parties" in card.todo_template:
        return [_todo_order(card, "get_contract_parties")]
    if todo_id == "get_counterparty" and "get_contract_parties" in card.todo_template:
        return [_todo_order(card, "get_contract_parties")]
    if todo_id == "get_contract" and "get_creative_status" in card.todo_template:
        return [_todo_order(card, "get_creative_status")]
    return []


def _todo_order(card: ProtocolCard, todo_id: str) -> int:
    try:
        return _todo_template(card).index(todo_id) + 1
    except ValueError:
        return 0


def _needs_fan_out(todo_id: str, frame: PlanningFrame) -> bool:
    return todo_id in {"get_counterparty", "list_placements", "list_documents"} and frame.cardinality in {
        "all",
        "n",
    }


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


def _clarify_plan(
    frame: PlanningFrame,
    reason: str,
    *,
    code: ClarifyCode = "missing_selector",
    missing_slot: str = "",
) -> TodoPlan:
    question = _clarify_question(frame, reason, code, missing_slot)
    return TodoPlan(
        goal=_goal(frame) or "clarify",
        protocol_id="clarify_first",
        strategy="compiled:clarify",
        status="awaiting_clarification",
        confidence=min(frame.confidence, 0.4),
        clarify_question=question,
        clarify_code=code,
        items=[
            TodoItem(id="parse_goal", order=1),
            TodoItem(id="clarify", order=2, blockers=[reason]),
        ],
        trace=["compiler clarify", code, reason],
    )


def _clarify_question(
    frame: PlanningFrame,
    reason: str,
    code: ClarifyCode,
    missing_slot: str,
) -> str:
    del frame
    if code == "unknown_target":
        return "Уточните, по какой сущности нужны данные."
    if code == "ambiguous_role":
        return "Уточните роль стороны: заказчик или исполнитель."
    if code == "ambiguous_protocol":
        return "Уточните действие или нужную сущность."
    if code == "write_confirm":
        label = _slot_label(missing_slot)
        return f"Уточните {label} для операции."
    if code == "low_confidence":
        return "Уточните цель запроса и нужную сущность."
    if missing_slot:
        return f"Уточните {_slot_label(missing_slot)}."
    return reason if reason.endswith("?") else f"Уточните входные данные: {reason}."


def _slot_label(slot: str) -> str:
    return _SLOT_LABELS.get(slot, slot or "входные данные")


def _rank_trace(ranked: list[ProtocolRank], selected: ProtocolRank) -> list[str]:
    events = [f"compiler rank {item.card.id}={item.score}" for item in ranked]
    events.append(
        "compiler slots "
        f"covered={','.join(selected.covered_slots) or '-'} "
        f"missing={','.join(selected.missing_slots) or '-'}"
    )
    return events


__all__ = [
    "TAU_CLARIFY",
    "ClarifyCode",
    "ProtocolRank",
    "compile_frame",
    "compile_plan",
    "rank_protocol_cards",
]
