"""Strict binding pass for todo plans."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from eva_agent.domain.plan import PlanStep, TodoItem, TodoPlan
from eva_agent.domain.relations import RELATIONS
from eva_agent.domain.slice import RelationSpec
from eva_agent.planner.catalog import CATALOG


class RemappedRef(BaseModel):
    """Legacy step reference converted to stable todo reference."""

    target_todo: str
    target_tool: str
    target_arg: str
    step: int
    todo: str
    path: str


class AutoWire(BaseModel):
    """Inserted relation-backed input binding."""

    target_todo: str
    target_tool: str
    target_arg: str
    producer_id: str
    source_path: str
    selector: str | None = None
    selector_value: Any | None = None
    cardinality: str = "one"
    fan_out: bool = False


class BindIssue(BaseModel):
    """Strict binding issue that must not be guessed around."""

    target_todo: str
    target_tool: str
    target_arg: str
    source_tool: str
    producer_ids: list[str] = Field(default_factory=list)
    reason: str


class BindReport(BaseModel):
    """Traceable result of the binding pass."""

    remapped: list[RemappedRef] = Field(default_factory=list)
    auto_wired: list[AutoWire] = Field(default_factory=list)
    ambiguous: list[BindIssue] = Field(default_factory=list)
    missing_producer: list[BindIssue] = Field(default_factory=list)


def bind_plan(plan: TodoPlan, relations: list[RelationSpec] | None = None) -> BindReport:
    """Rewrite legacy refs and insert deterministic relation bindings in-place."""

    active_relations = list(relations) if relations is not None else list(RELATIONS)
    report = BindReport()

    order_to_todo = _step_order_to_todo(plan)
    _remap_legacy_refs(plan, order_to_todo, report)
    _auto_wire_missing_inputs(plan, active_relations, report)
    return report


def _step_order_to_todo(plan: TodoPlan) -> dict[int, str]:
    owners: dict[int, set[str]] = {}
    for todo in plan.items:
        for step in todo.tool_calls:
            owners.setdefault(step.order, set()).add(todo.id)
    return {order: next(iter(ids)) for order, ids in owners.items() if len(ids) == 1}


def _remap_legacy_refs(
    plan: TodoPlan,
    order_to_todo: dict[int, str],
    report: BindReport,
) -> None:
    for todo in plan.items:
        for step in todo.tool_calls:
            for arg, value in list(step.args.items()):
                _remap_value(
                    value,
                    target_todo=todo,
                    target_step=step,
                    target_arg=str(arg),
                    order_to_todo=order_to_todo,
                    report=report,
                    plan=plan,
                )


def _remap_value(
    value: Any,
    *,
    target_todo: TodoItem,
    target_step: PlanStep,
    target_arg: str,
    order_to_todo: dict[int, str],
    report: BindReport,
    plan: TodoPlan,
) -> None:
    if isinstance(value, dict):
        ref = value.get("$from")
        if isinstance(ref, dict) and "todo" not in ref:
            step_order = ref.get("step")
            path = ref.get("path", "")
            if isinstance(step_order, int) and isinstance(path, str):
                producer_todo = order_to_todo.get(step_order)
                if producer_todo:
                    new_ref: dict[str, Any] = {"todo": producer_todo, "path": path}
                    for key in ("selector", "selector_value", "cardinality", "fan_out"):
                        if key in ref:
                            new_ref[key] = ref[key]
                    value["$from"] = new_ref
                    remap = RemappedRef(
                        target_todo=target_todo.id,
                        target_tool=str(target_step.tool),
                        target_arg=target_arg,
                        step=step_order,
                        todo=producer_todo,
                        path=path,
                    )
                    report.remapped.append(remap)
                    _append_trace(
                        plan,
                        f"remap {remap.target_tool}.{remap.target_arg} "
                        f"<- step {remap.step} as {remap.todo}.{remap.path}",
                    )
                return
        for child in value.values():
            _remap_value(
                child,
                target_todo=target_todo,
                target_step=target_step,
                target_arg=target_arg,
                order_to_todo=order_to_todo,
                report=report,
                plan=plan,
            )
    elif isinstance(value, list):
        for child in value:
            _remap_value(
                child,
                target_todo=target_todo,
                target_step=target_step,
                target_arg=target_arg,
                order_to_todo=order_to_todo,
                report=report,
                plan=plan,
            )


def _auto_wire_missing_inputs(
    plan: TodoPlan,
    relations: list[RelationSpec],
    report: BindReport,
) -> None:
    producers_by_tool = _producer_index(plan)
    for todo in plan.ordered():
        for step in sorted(todo.tool_calls, key=lambda current: current.order):
            target_tool = str(step.tool)
            for relation in relations:
                if relation.target_tool != target_tool:
                    continue
                if not _missing_value(step.args.get(relation.target_arg)):
                    continue
                if _has_value(todo.inputs.get(relation.target_arg)):
                    continue
                producers = [
                    producer
                    for producer in producers_by_tool.get(relation.source_tool, [])
                    if producer.id != todo.id
                ]
                if len(producers) != 1:
                    _record_producer_count_issue(plan, todo, step, relation, producers, report)
                    continue
                producer = producers[0]
                if not _can_add_dependency(plan, consumer=todo, producer=producer):
                    issue = BindIssue(
                        target_todo=todo.id,
                        target_tool=target_tool,
                        target_arg=relation.target_arg,
                        source_tool=relation.source_tool,
                        producer_ids=[producer.id],
                        reason="producer cannot precede consumer",
                    )
                    report.ambiguous.append(issue)
                    _append_trace(plan, _issue_trace("ambiguous", issue))
                    continue
                _insert_auto_wire(plan, todo, step, relation, producer, report)


def _producer_index(plan: TodoPlan) -> dict[str, list[TodoItem]]:
    out: dict[str, list[TodoItem]] = {}
    for todo in plan.items:
        tools = {str(step.tool) for step in todo.tool_calls}
        for tool in tools:
            out.setdefault(tool, []).append(todo)
    return out


def _record_producer_count_issue(
    plan: TodoPlan,
    todo: TodoItem,
    step: PlanStep,
    relation: RelationSpec,
    producers: list[TodoItem],
    report: BindReport,
) -> None:
    target_tool = str(step.tool)
    if producers:
        issue = BindIssue(
            target_todo=todo.id,
            target_tool=target_tool,
            target_arg=relation.target_arg,
            source_tool=relation.source_tool,
            producer_ids=[producer.id for producer in producers],
            reason="multiple producers",
        )
        report.ambiguous.append(issue)
        _append_trace(plan, _issue_trace("ambiguous", issue))
        return
    if _required_for_todo(todo, relation.target_arg):
        issue = BindIssue(
            target_todo=todo.id,
            target_tool=target_tool,
            target_arg=relation.target_arg,
            source_tool=relation.source_tool,
            producer_ids=[],
            reason="missing producer",
        )
        report.missing_producer.append(issue)
        _append_trace(plan, _issue_trace("missing producer", issue))


def _insert_auto_wire(
    plan: TodoPlan,
    todo: TodoItem,
    step: PlanStep,
    relation: RelationSpec,
    producer: TodoItem,
    report: BindReport,
) -> None:
    selector_value = (
        _todo_selector_value(producer, relation.selector) if relation.selector else None
    )
    fan_out = relation.cardinality == "many" and _explicit_fan_out_requested(todo, step)
    ref: dict[str, Any] = {
        "todo": producer.id,
        "path": relation.source_path,
        "selector": relation.selector,
        "cardinality": relation.cardinality,
    }
    if selector_value is not None:
        ref["selector_value"] = selector_value
    if fan_out:
        ref["fan_out"] = True
    step.args[relation.target_arg] = {"$from": ref}
    if producer.order not in todo.depends_on:
        todo.depends_on.append(producer.order)
    wire = AutoWire(
        target_todo=todo.id,
        target_tool=str(step.tool),
        target_arg=relation.target_arg,
        producer_id=producer.id,
        source_path=relation.source_path,
        selector=relation.selector,
        selector_value=selector_value,
        cardinality=relation.cardinality,
        fan_out=fan_out,
    )
    report.auto_wired.append(wire)
    _append_trace(plan, _auto_wire_trace(wire))


def _auto_wire_trace(wire: AutoWire) -> str:
    selector = f"[selector={wire.selector}]" if wire.selector else ""
    return (
        f"auto-wire {wire.target_tool}.{wire.target_arg} "
        f"<- {wire.producer_id}.{wire.source_path}{selector}"
    )


def _todo_selector_value(todo: TodoItem, selector: str) -> Any | None:
    values: list[Any] = []
    _append_selector_value(values, todo.inputs.get(selector))
    for step in todo.tool_calls:
        _append_selector_value(values, step.args.get(selector))
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values[1:]):
        return first
    return None


def _append_selector_value(values: list[Any], value: Any) -> None:
    if value in (None, ""):
        return
    if isinstance(value, (Mapping, list)):
        return
    values.append(value)


def _explicit_fan_out_requested(todo: TodoItem, step: PlanStep) -> bool:
    return (
        _has_true(todo.inputs, "fan_out")
        or _has_true(todo.inputs, "select_all")
        or _has_true(step.args, "fan_out")
        or _has_true(step.args, "select_all")
    )


def _has_true(values: Mapping[str, Any], key: str) -> bool:
    return values.get(key) is True


def _issue_trace(kind: str, issue: BindIssue) -> str:
    producers = ",".join(issue.producer_ids) if issue.producer_ids else "-"
    return (
        f"{kind} {issue.target_tool}.{issue.target_arg} "
        f"<- {issue.source_tool} producers={producers}"
    )


def _can_add_dependency(plan: TodoPlan, *, consumer: TodoItem, producer: TodoItem) -> bool:
    if consumer.order == producer.order:
        return False
    graph = {todo.order: list(todo.depends_on) for todo in plan.items}
    return not _has_dependency_path(graph, start=producer.order, target=consumer.order)


def _has_dependency_path(graph: Mapping[int, list[int]], *, start: int, target: int) -> bool:
    stack = [start]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, []))
    return False


def _required_for_todo(todo: TodoItem, arg: str) -> bool:
    spec = CATALOG.get(todo.id)
    return spec is not None and arg in spec.inputs_required


def _missing_value(value: Any) -> bool:
    return value in (None, "", [], {})


def _has_value(value: Any) -> bool:
    return not _missing_value(value)


def _append_trace(plan: TodoPlan, event: str) -> None:
    if event not in plan.trace:
        plan.trace.append(event)


__all__ = ["AutoWire", "BindIssue", "BindReport", "RemappedRef", "bind_plan"]
