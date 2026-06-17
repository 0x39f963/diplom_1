"""Deterministic validation for planner todo plans."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from eva_agent.domain.plan import TodoItem, TodoPlan
from eva_agent.planner.bind import BindReport, bind_plan
from eva_agent.planner.catalog import AVAILABLE_TOOLS_DEFAULT, CATALOG, TodoSpec, todo_is_available
from eva_agent.planner.protocols import PROTOCOLS
from eva_agent.tools.selector import EXECUTION_REGISTRY

PLANNER_MIN_CONFIDENCE = 0.45
_INTERNAL_TODOS = frozenset({"parse_goal", "summarize_answer", "clarify"})


def validate_plan(plan: TodoPlan) -> TodoPlan:
    """Validate a planner plan without raising on model mistakes."""

    bind_report = bind_plan(plan)
    violations: list[str] = []
    _validate_bind_report(plan, bind_report, violations)
    _validate_todos(plan, violations)
    _validate_step_order(plan, violations)
    _validate_from_refs(plan, violations)
    _validate_depends_on(plan, violations)
    _validate_required_inputs(plan)
    _validate_mandatory(plan, violations)

    if plan.confidence < PLANNER_MIN_CONFIDENCE:
        violations.append("low confidence")
    if plan.is_empty:
        violations.append("empty plan")

    if violations:
        _await_clarification(plan, violations)
    return plan


def _validate_bind_report(
    plan: TodoPlan,
    report: BindReport,
    violations: list[str],
) -> None:
    by_id = {todo.id: todo for todo in plan.items}
    for issue in report.ambiguous:
        todo = by_id.get(issue.target_todo)
        if todo is None:
            continue
        todo.status = "blocked"
        blocker = f"ambiguous auto-wire: {issue.target_arg}"
        _add_blocker(todo, blocker)
        violations.append(blocker)


def _validate_todos(plan: TodoPlan, violations: list[str]) -> None:
    for todo in plan.items:
        spec = CATALOG.get(todo.id)
        if spec is None:
            todo.status = "skipped"
            _add_blocker(todo, f"unknown todo: {todo.id}")
            violations.append(f"unknown todo: {todo.id}")
            continue
        if not todo_is_available(todo.id, AVAILABLE_TOOLS_DEFAULT):
            todo.status = "skipped"
            _add_blocker(todo, f"unavailable todo: {todo.id}")
            violations.append(f"unavailable todo: {todo.id}")
            continue
        _validate_tool_calls(todo, spec, violations)


def _validate_tool_calls(todo: TodoItem, spec: TodoSpec, violations: list[str]) -> None:
    allowed_tools = set(spec.tools)
    for step in todo.tool_calls:
        tool = str(step.tool)
        if tool not in EXECUTION_REGISTRY:
            todo.status = "blocked"
            _add_blocker(todo, f"unknown tool: {tool}")
            violations.append(f"unknown tool: {tool}")
        if allowed_tools and tool not in allowed_tools:
            todo.status = "blocked"
            _add_blocker(todo, f"tool is not allowed for todo: {tool}")
            violations.append(f"tool is not allowed for todo: {tool}")
        if tool not in AVAILABLE_TOOLS_DEFAULT:
            todo.status = "blocked"
            _add_blocker(todo, f"unavailable tool: {tool}")
            violations.append(f"unavailable tool: {tool}")


def _validate_step_order(plan: TodoPlan, violations: list[str]) -> None:
    orders = [step.order for todo in plan.ordered() for step in sorted(todo.tool_calls, key=lambda s: s.order)]
    if len(orders) != len(set(orders)):
        violations.append("duplicate step order")
        return
    previous = 0
    for order in orders:
        if order <= previous:
            violations.append("step order is not strictly increasing")
            return
        previous = order


def _validate_from_refs(plan: TodoPlan, violations: list[str]) -> None:
    existing = {step.order for todo in plan.items for step in todo.tool_calls}
    todos = {todo.id: todo for todo in plan.items}
    for todo in plan.items:
        for step in todo.tool_calls:
            for ref_todo in _iter_from_todos(step.args):
                producer = todos.get(ref_todo)
                if producer is None:
                    todo.status = "blocked"
                    _add_blocker(todo, f"unknown $from todo: {ref_todo}")
                    violations.append(f"unknown $from todo: {ref_todo}")
                elif producer.order >= todo.order and producer.order not in todo.depends_on:
                    todo.status = "blocked"
                    _add_blocker(todo, f"forward $from todo: {ref_todo}")
                    violations.append(f"forward $from todo: {ref_todo}")
                    legacy_step = _single_step_order(producer)
                    if legacy_step is not None:
                        _add_blocker(todo, f"forward $from step: {legacy_step}")
                        violations.append(f"forward $from step: {legacy_step}")
            for ref_step in _iter_from_steps(step.args):
                if ref_step not in existing:
                    todo.status = "blocked"
                    _add_blocker(todo, f"unknown $from step: {ref_step}")
                    violations.append(f"unknown $from step: {ref_step}")
                elif ref_step >= step.order:
                    todo.status = "blocked"
                    _add_blocker(todo, f"forward $from step: {ref_step}")
                    violations.append(f"forward $from step: {ref_step}")


def _validate_depends_on(plan: TodoPlan, violations: list[str]) -> None:
    orders = [todo.order for todo in plan.items]
    order_set = set(orders)
    if len(orders) != len(order_set):
        violations.append("duplicate todo order")
    graph: dict[int, list[int]] = {}
    for todo in plan.items:
        if todo.type == "dependent" and not todo.depends_on:
            todo.status = "blocked"
            _add_blocker(todo, "dependent todo has no dependencies")
            violations.append("dependent todo has no dependencies")
        graph[todo.order] = list(todo.depends_on)
        for dependency in todo.depends_on:
            if dependency not in order_set:
                todo.status = "blocked"
                _add_blocker(todo, f"unknown dependency: {dependency}")
                violations.append(f"unknown dependency: {dependency}")
    if _has_cycle(graph):
        violations.append("dependency cycle")
        for todo in plan.items:
            if todo.depends_on:
                todo.status = "blocked"
                _add_blocker(todo, "dependency cycle")


def _validate_required_inputs(plan: TodoPlan) -> None:
    for todo in plan.items:
        spec = CATALOG.get(todo.id)
        if spec is None or todo.status in ("blocked", "skipped"):
            continue
        for required in spec.inputs_required:
            if _has_required_input(todo, required):
                continue
            todo.status = "blocked"
            blocker = spec.blockers[0] if spec.blockers else f"missing input: {required}"
            _add_blocker(todo, blocker)


def _validate_mandatory(plan: TodoPlan, violations: list[str]) -> None:
    if plan.protocol_id not in PROTOCOLS:
        violations.append(f"unknown protocol: {plan.protocol_id}")
        return
    spec = PROTOCOLS[plan.protocol_id]
    by_id = {todo.id: todo for todo in plan.items}
    for todo_id in spec.mandatory:
        todo = by_id.get(todo_id)
        if todo is None and todo_id in _INTERNAL_TODOS:
            continue
        if todo is None:
            violations.append(f"missing mandatory todo: {todo_id}")
            continue
        if todo.id == "clarify":
            continue
        if todo.status in ("blocked", "skipped"):
            violations.append(f"blocked mandatory todo: {todo_id}")


def _iter_from_steps(value: Any) -> Iterable[int]:
    if isinstance(value, Mapping):
        ref = value.get("$from")
        if isinstance(ref, Mapping):
            step = ref.get("step")
            if isinstance(step, int):
                yield step
        for child in value.values():
            yield from _iter_from_steps(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_from_steps(child)


def _iter_from_todos(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        ref = value.get("$from")
        if isinstance(ref, Mapping):
            todo = ref.get("todo")
            if isinstance(todo, str):
                yield todo
        for child in value.values():
            yield from _iter_from_todos(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_from_todos(child)


def _single_step_order(todo: TodoItem) -> int | None:
    orders = {step.order for step in todo.tool_calls}
    if len(orders) != 1:
        return None
    return next(iter(orders))


def _has_required_input(todo: TodoItem, name: str) -> bool:
    if name in todo.inputs and todo.inputs[name] not in (None, ""):
        return True
    return any(name in step.args and step.args[name] not in (None, "") for step in todo.tool_calls)


def _has_cycle(graph: dict[int, list[int]]) -> bool:
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: int) -> bool:
        if node in visited:
            return False
        if node in visiting:
            return True
        visiting.add(node)
        for dependency in graph.get(node, []):
            if dependency in graph and visit(dependency):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def _await_clarification(plan: TodoPlan, violations: list[str]) -> None:
    plan.status = "awaiting_clarification"
    if plan.clarify_question:
        return
    blocker = _first_blocker(plan)
    if blocker:
        plan.clarify_question = _question_from_blocker(blocker)
    else:
        plan.clarify_question = _question_from_blocker(violations[0])


def _first_blocker(plan: TodoPlan) -> str:
    for todo in plan.ordered():
        if todo.blockers:
            return todo.blockers[0]
    return ""


def _question_from_blocker(blocker: str) -> str:
    return f"Уточните входные данные: {blocker}."


def _add_blocker(todo: TodoItem, blocker: str) -> None:
    if blocker not in todo.blockers:
        todo.blockers.append(blocker)
