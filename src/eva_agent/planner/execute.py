"""Deterministic executor for planner todo lists."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from typing import Any

from eva_agent.domain.plan import StepResult, TodoItem, TodoPlan
from eva_agent.domain.slice import RelationSpec
from eva_agent.planner.bind import BindReport, bind_plan
from eva_agent.planner.filters import apply_filters
from eva_agent.planner.protocols import PROTOCOLS
from eva_agent.state import ApiFinding
from eva_agent.tools.selector import EXECUTION_REGISTRY

_INTERNAL_TODOS = frozenset({"parse_goal", "summarize_answer", "clarify"})
_MIXED_DATA_TODOS = frozenset(
    {
        "get_contract",
        "get_creative_status",
        "get_contract_parties",
        "get_counterparty",
        "list_placements",
    }
)
_FAN_OUT_CAP = 20


class _UnresolvedRef(Exception):
    """Raised when a $from reference cannot be resolved."""


class _AmbiguousRef(Exception):
    """Raised when a relation-backed reference has no strict single value."""

    def __init__(self, reason: str, *, arg_name: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.arg_name = arg_name


class _FanOutValues(list[Any]):
    """Marker for relation-backed many values that require fan-out execution."""


def execute_plan(
    plan: TodoPlan,
    relations: list[RelationSpec] | None = None,
) -> tuple[list[ApiFinding], TodoPlan]:
    """Execute todo items in order and update their statuses."""

    bind_report = bind_plan(plan, relations)
    _apply_bind_blockers(plan, bind_report)

    findings: list[ApiFinding] = []
    results: dict[int, StepResult] = {}
    results_by_todo: dict[str, StepResult] = {}
    seen: set[tuple[str, str]] = set()
    done_todos: set[int] = set()

    pending = plan.ordered()
    while pending:
        progressed = False
        deferred: list[TodoItem] = []
        for todo in pending:
            if todo.status in ("blocked", "skipped"):
                progressed = True
                continue
            if any(dependency not in done_todos for dependency in todo.depends_on):
                deferred.append(todo)
                continue
            _execute_todo(plan, todo, findings, results, results_by_todo, seen)
            if todo.status == "done":
                done_todos.add(todo.order)
            progressed = True
        if not deferred:
            break
        if not progressed:
            for todo in deferred:
                if todo.status not in ("blocked", "skipped"):
                    todo.status = "blocked"
                    _add_blocker(todo, "dependency is not done")
            break
        pending = deferred

    plan = _apply_checklist(plan)
    return findings, plan


def _execute_todo(
    plan: TodoPlan,
    todo: TodoItem,
    findings: list[ApiFinding],
    results: dict[int, StepResult],
    results_by_todo: dict[str, StepResult],
    seen: set[tuple[str, str]],
) -> None:
    if todo.status in ("blocked", "skipped"):
        return
    if not todo.tool_calls and todo.id in _INTERNAL_TODOS:
        todo.status = "done"
        return
    if not todo.tool_calls:
        todo.status = "blocked"
        _add_blocker(todo, "no tool calls")
        return

    todo_ok = False
    for step in sorted(todo.tool_calls, key=lambda current: current.order):
        tool = str(step.tool)
        fn = EXECUTION_REGISTRY.get(tool)
        if fn is None:
            _add_blocker(todo, f"unknown tool: {tool}")
            continue
        try:
            args = _resolve_args(step.args, results, results_by_todo, todo)
        except _UnresolvedRef as exc:
            _add_blocker(todo, str(exc))
            continue
        except _AmbiguousRef as exc:
            _add_blocker(todo, _ambiguous_blocker(exc))
            continue
        args = apply_filters(step, args)
        if _has_fan_out(args):
            todo_ok = _execute_fan_out(
                plan,
                todo,
                step_order=step.order,
                tool=tool,
                fn=fn,
                args=args,
                findings=findings,
                results=results,
                results_by_todo=results_by_todo,
                seen=seen,
            ) or todo_ok
            continue
        finding, skipped_duplicate = _execute_call(todo, tool, fn, args, seen)
        if skipped_duplicate:
            todo_ok = True
            continue
        if finding is None:
            continue
        findings.append(finding)
        result = _step_result(step.order, tool, finding, args)
        results[step.order] = result
        results_by_todo[todo.id] = result
        todo.result_ref = tool
        todo_ok = True

    todo.status = "done" if todo_ok else "blocked"


def _execute_call(
    todo: TodoItem,
    tool: str,
    fn: Callable[..., ApiFinding],
    args: dict[str, Any],
    seen: set[tuple[str, str]],
) -> tuple[ApiFinding | None, bool]:
    key = (tool, _dedup_args(args))
    if key in seen:
        return None, True
    seen.add(key)
    try:
        return _call_tool(fn, args), False
    except Exception as exc:
        _add_blocker(todo, f"tool failed: {tool}: {exc.__class__.__name__}")
        return None, False


def _execute_fan_out(
    plan: TodoPlan,
    todo: TodoItem,
    *,
    step_order: int,
    tool: str,
    fn: Callable[..., ApiFinding],
    args: dict[str, Any],
    findings: list[ApiFinding],
    results: dict[int, StepResult],
    results_by_todo: dict[str, StepResult],
    seen: set[tuple[str, str]],
) -> bool:
    fan_args = [key for key, value in args.items() if isinstance(value, _FanOutValues)]
    if len(fan_args) != 1:
        _add_blocker(todo, "ambiguous auto-wire: multiple fan-out args")
        return False
    fan_arg = fan_args[0]
    values = list(args[fan_arg])
    if not values:
        _add_blocker(todo, "empty producer")
        _append_trace(plan, "empty producer")
        return False
    if len(values) > _FAN_OUT_CAP:
        _add_blocker(todo, f"fan-out capped at {_FAN_OUT_CAP} of {len(values)}")
        _append_trace(plan, f"fan-out capped at {_FAN_OUT_CAP} of {len(values)}")
        values = values[:_FAN_OUT_CAP]

    todo_ok = False
    for index, value in enumerate(values, start=1):
        call_args = dict(args)
        call_args[fan_arg] = value
        _append_trace(plan, f"fan-out {tool}.{fan_arg}[{index}]")
        finding, skipped_duplicate = _execute_call(todo, tool, fn, call_args, seen)
        if skipped_duplicate:
            todo_ok = True
            continue
        if finding is None:
            continue
        findings.append(finding)
        result = _step_result(step_order, tool, finding, call_args)
        results[step_order] = result
        results_by_todo[todo.id] = result
        todo.result_ref = tool
        todo_ok = True
    return todo_ok


def _step_result(
    order: int,
    tool: str,
    finding: ApiFinding,
    fallback_args: dict[str, Any],
) -> StepResult:
    return StepResult(
        order=order,
        tool=tool,
        args=dict(finding.args) if finding.args else fallback_args,
        data=dict(finding.data),
    )


def _apply_bind_blockers(plan: TodoPlan, report: BindReport) -> None:
    by_id = {todo.id: todo for todo in plan.items}
    for issue in report.ambiguous:
        todo = by_id.get(issue.target_todo)
        if todo is None:
            continue
        todo.status = "blocked"
        _add_blocker(todo, f"ambiguous auto-wire: {issue.target_arg}")
    for issue in report.missing_producer:
        todo = by_id.get(issue.target_todo)
        if todo is None:
            continue
        todo.status = "blocked"
        _add_blocker(todo, f"missing producer: {issue.target_arg}")


def _append_trace(plan: TodoPlan, event: str) -> None:
    if event not in plan.trace:
        plan.trace.append(event)


def _resolve_args(
    args: dict[str, Any],
    results: dict[int, StepResult],
    results_by_todo: dict[str, StepResult],
    todo: TodoItem,
) -> dict[str, Any]:
    selector_values = _selector_values(args, todo)
    resolved: dict[str, Any] = {}
    for key, value in args.items():
        try:
            resolved[str(key)] = _resolve_value(value, results, results_by_todo, selector_values)
        except _AmbiguousRef as exc:
            if not exc.arg_name:
                exc.arg_name = str(key)
            raise
    return resolved


def _resolve_value(
    value: Any,
    results: dict[int, StepResult],
    results_by_todo: dict[str, StepResult],
    selector_values: dict[str, Any],
) -> Any:
    if isinstance(value, Mapping):
        ref = value.get("$from")
        if isinstance(ref, Mapping):
            return _resolve_ref(ref, results, results_by_todo, selector_values)
        return {
            str(key): _resolve_value(child, results, results_by_todo, selector_values)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_resolve_value(child, results, results_by_todo, selector_values) for child in value]
    return value


def _resolve_ref(
    ref: Mapping[str, Any],
    results: dict[int, StepResult],
    results_by_todo: dict[str, StepResult],
    selector_values: dict[str, Any],
) -> Any:
    path = ref.get("path", "")
    if not isinstance(path, str):
        raise _UnresolvedRef("invalid $from reference")

    todo_id = ref.get("todo")
    if isinstance(todo_id, str):
        result = results_by_todo.get(todo_id)
        if result is None:
            raise _UnresolvedRef(f"unresolved $from todo: {todo_id}")
        selector = ref.get("selector")
        if selector is not None and not isinstance(selector, str):
            raise _UnresolvedRef("invalid $from selector")
        cardinality = ref.get("cardinality", "one")
        if cardinality not in ("one", "many"):
            raise _UnresolvedRef("invalid $from cardinality")
        selector_value = ref.get("selector_value")
        if selector_value in (None, "") and selector:
            selector_value = selector_values.get(selector)
        return _dig_select(
            result.data,
            path,
            selector=selector,
            cardinality=cardinality,
            selector_value=selector_value,
        )

    step = ref.get("step")
    if not isinstance(step, int):
        raise _UnresolvedRef("invalid $from reference")
    result = results.get(step)
    if result is None:
        raise _UnresolvedRef(f"unresolved $from step: {step}")
    return _dig(result.data, path)


def _selector_values(args: dict[str, Any], todo: TodoItem) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for source in (todo.inputs, args):
        for key, value in source.items():
            selector_value = _simple_selector_value(value)
            if selector_value is not None:
                values[str(key)] = selector_value
    return values


def _simple_selector_value(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (Mapping, list)):
        return None
    return value


def _dig_select(
    data: Any,
    path: str,
    *,
    selector: str | None,
    cardinality: str,
    selector_value: Any,
) -> Any:
    values = _dig_many(data, path, selector=selector, selector_value=selector_value)
    if cardinality == "many" and selector_value in (None, ""):
        return _FanOutValues(values)
    if not values:
        raise _AmbiguousRef("empty producer")
    if len(values) != 1:
        raise _AmbiguousRef("ambiguous")
    return values[0]


def _dig_many(data: Any, path: str, *, selector: str | None, selector_value: Any) -> list[Any]:
    current = [data]
    if not path:
        return current
    for part in path.split("."):
        next_values: list[Any] = []
        for item in current:
            next_values.extend(_dig_part(item, part, path, selector, selector_value))
        current = next_values
    return current


def _dig_part(
    data: Any,
    part: str,
    full_path: str,
    selector: str | None,
    selector_value: Any,
) -> list[Any]:
    if part.endswith("[]"):
        key = part[:-2]
        list_value = _mapping_value(data, key, full_path)
        if not isinstance(list_value, list):
            raise _UnresolvedRef(f"missing list path: {full_path}")
        items = list_value
        if selector and selector_value not in (None, ""):
            items = [
                item
                for item in items
                if isinstance(item, Mapping) and item.get(selector) == selector_value
            ]
        return list(items)
    if isinstance(data, Mapping):
        if part not in data:
            raise _UnresolvedRef(f"missing path: {full_path}")
        return [data[part]]
    if isinstance(data, list):
        try:
            index = int(part)
        except ValueError as exc:
            raise _UnresolvedRef(f"invalid list index: {part}") from exc
        try:
            return [data[index]]
        except IndexError as exc:
            raise _UnresolvedRef(f"missing list index: {part}") from exc
    raise _UnresolvedRef(f"missing path: {full_path}")


def _mapping_value(data: Any, key: str, full_path: str) -> Any:
    if not isinstance(data, Mapping) or key not in data:
        raise _UnresolvedRef(f"missing path: {full_path}")
    return data[key]


def _has_fan_out(args: dict[str, Any]) -> bool:
    return any(isinstance(value, _FanOutValues) for value in args.values())


def _ambiguous_blocker(exc: _AmbiguousRef) -> str:
    if exc.reason == "empty producer":
        return "empty producer"
    arg = exc.arg_name or "reference"
    return f"ambiguous auto-wire: {arg}"


def _dig(data: Any, path: str) -> Any:
    current = data
    if not path:
        return current
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise _UnresolvedRef(f"missing path: {path}")
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
            except ValueError as exc:
                raise _UnresolvedRef(f"invalid list index: {part}") from exc
            try:
                current = current[index]
            except IndexError as exc:
                raise _UnresolvedRef(f"missing list index: {part}") from exc
        else:
            raise _UnresolvedRef(f"missing path: {path}")
    return current


def _call_tool(fn: Callable[..., ApiFinding], args: dict[str, Any]) -> ApiFinding:
    signature = inspect.signature(fn)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return fn(**args)
    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    call_args = {key: value for key, value in args.items() if key in allowed}
    return fn(**call_args)


def _apply_checklist(plan: TodoPlan) -> TodoPlan:
    """Apply mandatory protocol checklist semantics."""

    if plan.protocol_id not in PROTOCOLS:
        plan.status = "awaiting_clarification"
        if not plan.clarify_question:
            plan.clarify_question = "Уточните цель запроса."
        return plan
    spec = PROTOCOLS[plan.protocol_id]

    by_id = {todo.id: todo for todo in plan.items}
    unmet: list[str] = []
    blockers: list[str] = []
    for todo_id in spec.mandatory:
        todo = by_id.get(todo_id)
        if todo is None and todo_id in _INTERNAL_TODOS:
            continue
        if todo is None:
            unmet.append(todo_id)
            blockers.append(f"missing mandatory todo: {todo_id}")
            continue
        if _mandatory_ok(todo):
            continue
        unmet.append(todo_id)
        blockers.extend(todo.blockers or [f"mandatory todo is not done: {todo_id}"])

    if plan.protocol_id == "mixed_legal_data" and not _has_done_mixed_data_todo(plan.items):
        unmet.append("mixed_legal_data:data_todo")
        blockers.append("нет выполненного data-todo для сопоставления с нормой")

    if unmet:
        plan.status = "awaiting_clarification"
        if not plan.clarify_question:
            plan.clarify_question = _default_clarify(blockers or unmet)
    else:
        plan.status = "answered"
    return plan


def _mandatory_ok(todo: TodoItem) -> bool:
    if todo.status == "done":
        return True
    return todo.status == "blocked" and any("terminal:" in blocker for blocker in todo.blockers)


def _has_done_mixed_data_todo(items: list[TodoItem]) -> bool:
    return any(todo.id in _MIXED_DATA_TODOS and todo.status == "done" for todo in items)


def _default_clarify(blockers: list[str]) -> str:
    blocker = blockers[0] if blockers else "недостаточно данных"
    return f"Уточните входные данные: {blocker}."


def _dedup_args(args: dict[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def _add_blocker(todo: TodoItem, blocker: str) -> None:
    if blocker not in todo.blockers:
        todo.blockers.append(blocker)
