"""Deterministic executor for planner todo lists."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from typing import Any

from eva_agent.domain.plan import StepResult, TodoItem, TodoPlan
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


class _UnresolvedRef(Exception):
    """Raised when a $from reference cannot be resolved."""


def execute_plan(plan: TodoPlan) -> tuple[list[ApiFinding], TodoPlan]:
    """Execute todo items in order and update their statuses."""

    findings: list[ApiFinding] = []
    results: dict[int, StepResult] = {}
    seen: set[tuple[str, str]] = set()
    done_todos: set[int] = set()

    for todo in plan.ordered():
        if todo.status in ("blocked", "skipped"):
            continue
        if any(dependency not in done_todos for dependency in todo.depends_on):
            todo.status = "blocked"
            _add_blocker(todo, "dependency is not done")
            continue
        if not todo.tool_calls and todo.id in _INTERNAL_TODOS:
            todo.status = "done"
            done_todos.add(todo.order)
            continue
        if not todo.tool_calls:
            todo.status = "blocked"
            _add_blocker(todo, "no tool calls")
            continue

        todo_ok = False
        for step in sorted(todo.tool_calls, key=lambda current: current.order):
            tool = str(step.tool)
            fn = EXECUTION_REGISTRY.get(tool)
            if fn is None:
                _add_blocker(todo, f"unknown tool: {tool}")
                continue
            try:
                args = _resolve_args(step.args, results)
            except _UnresolvedRef as exc:
                _add_blocker(todo, str(exc))
                continue
            args = apply_filters(step, args)
            key = (tool, _dedup_args(args))
            if key in seen:
                todo_ok = True
                continue
            seen.add(key)
            try:
                finding = _call_tool(fn, args)
            except Exception as exc:
                _add_blocker(todo, f"tool failed: {tool}: {exc.__class__.__name__}")
                continue
            findings.append(finding)
            results[step.order] = StepResult(
                order=step.order,
                tool=tool,
                args=dict(finding.args) if finding.args else args,
                data=dict(finding.data),
            )
            todo.result_ref = tool
            todo_ok = True

        todo.status = "done" if todo_ok else "blocked"
        if todo_ok:
            done_todos.add(todo.order)

    plan = _apply_checklist(plan)
    return findings, plan


def _resolve_args(args: dict[str, Any], results: dict[int, StepResult]) -> dict[str, Any]:
    resolved = _resolve_value(args, results)
    if not isinstance(resolved, dict):
        raise _UnresolvedRef("resolved args are not an object")
    return resolved


def _resolve_value(value: Any, results: dict[int, StepResult]) -> Any:
    if isinstance(value, Mapping):
        ref = value.get("$from")
        if isinstance(ref, Mapping):
            step = ref.get("step")
            path = ref.get("path", "")
            if not isinstance(step, int) or not isinstance(path, str):
                raise _UnresolvedRef("invalid $from reference")
            result = results.get(step)
            if result is None:
                raise _UnresolvedRef(f"unresolved $from step: {step}")
            return _dig(result.data, path)
        return {str(key): _resolve_value(child, results) for key, child in value.items()}
    if isinstance(value, list):
        return [_resolve_value(child, results) for child in value]
    return value


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
