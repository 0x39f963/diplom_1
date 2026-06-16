"""Deterministic planner filters."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from eva_agent.domain.plan import DateHint, PlanStep, StatusHint

_LIST_FILTER_TOOLS = frozenset({"eva_search_contracts", "eva_list_unsigned_contracts"})


def date_range(hint: DateHint, *, today: date | None = None) -> tuple[str, str] | None:
    """Return an inclusive ISO date range for a planner date hint."""

    if hint == "none":
        return None
    base = today or date.today()
    if hint == "yesterday":
        target = base - timedelta(days=1)
        return target.isoformat(), target.isoformat()
    if hint == "last_week":
        start = base - timedelta(days=7)
        return start.isoformat(), base.isoformat()
    if hint == "last_month":
        start = base - timedelta(days=30)
        return start.isoformat(), base.isoformat()
    return None


def status_set(hint: StatusHint) -> list[str]:
    """Return status values for a planner status hint."""

    if hint == "unsigned":
        return ["draft", "pending", "sent"]
    if hint == "draft":
        return ["draft"]
    if hint == "registered":
        return ["registered"]
    return []


def apply_filters(
    step: PlanStep,
    args: dict[str, Any],
    *,
    today: date | None = None,
) -> dict[str, Any]:
    """Apply date and status hints to list-style tool arguments."""

    filtered = dict(args)
    if step.tool not in _LIST_FILTER_TOOLS:
        return filtered

    dates = date_range(step.date_hint, today=today)
    if dates is not None:
        date_from, date_to = dates
        filtered.setdefault("date_from", date_from)
        filtered.setdefault("date_to", date_to)
        if step.tool == "eva_search_contracts":
            filtered.setdefault("date_hint", date_from if date_from == date_to else f"{date_from}..{date_to}")

    statuses = status_set(step.status_hint)
    if statuses:
        filtered.setdefault("statuses", statuses)
        if step.tool == "eva_search_contracts":
            filtered.setdefault("status_hint", step.status_hint)

    return filtered
