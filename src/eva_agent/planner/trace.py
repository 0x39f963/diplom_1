"""Planner trace payload for audit and LangFuse metadata."""

from __future__ import annotations

from typing import Any

from eva_agent.domain.plan import TodoPlan
from eva_agent.llm.observability import langfuse_enabled
from eva_agent.state import ApiFinding


def trace_plan(
    plan: TodoPlan,
    findings: list[ApiFinding],
    *,
    plan_reused: bool = False,
    plan_attempts: int = 0,
    rebuild_reason: str = "",
) -> dict[str, Any]:
    """Build and optionally attach a structured planner trace payload."""

    payload: dict[str, Any] = {
        "protocol_id": plan.protocol_id,
        "strategy": plan.strategy,
        "dialog_status": plan.status,
        "confidence": plan.confidence,
        "goal": plan.goal,
        "todos": [
            {
                "id": todo.id,
                "type": todo.type,
                "order": todo.order,
                "status": todo.status,
                "success": todo.status == "done",
                "failure": todo.status in ("blocked", "skipped"),
                "tool": todo.result_ref or None,
                "error": "; ".join(todo.blockers) if todo.blockers else "",
                "blockers": list(todo.blockers),
                "tools_planned": [str(call.tool) for call in todo.tool_calls],
            }
            for todo in plan.ordered()
        ],
        "findings_tools": [finding.tool for finding in findings],
        "trace": list(plan.trace),
        "coverage": dict(plan.coverage),
        "clarify_question": plan.clarify_question,
        "clarify_code": plan.clarify_code,
        "plan_reused": plan_reused,
        "plan_attempts": plan_attempts,
        "rebuild_reason": rebuild_reason,
    }
    _attach_to_langfuse(payload)
    return payload


def _attach_to_langfuse(payload: dict[str, Any]) -> None:
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().update_current_span(metadata={"planner": payload})
    except Exception:
        return
