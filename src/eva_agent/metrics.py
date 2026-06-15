"""Учет стоимости/вызовов LLM за один прогон графа (через contextvar).

`start_run()` перед graph.invoke, затем `run_cost_usd()`/`run_calls()` после - дает
cost-per-run для метрик eval без сквозного протаскивания usage через состояние.
"""

from __future__ import annotations

import contextvars
from typing import Any

_usage: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "eva_usage", default=None
)


def start_run() -> None:
    _usage.set([])


def record(usage: dict[str, Any]) -> None:
    log = _usage.get()
    if log is not None:
        log.append(usage)


def run_cost_usd() -> float:
    return sum(float(u.get("cost_usd") or 0.0) for u in (_usage.get() or []))


def run_calls() -> int:
    return len(_usage.get() or [])


def run_tokens() -> dict[str, int]:
    """Сумма токенов за прогон (prompt/completion/total) - для отчетности."""
    log = _usage.get() or []
    prompt = sum(int(u.get("prompt_tokens") or 0) for u in log)
    completion = sum(int(u.get("completion_tokens") or 0) for u in log)
    return {"prompt": prompt, "completion": completion, "total": prompt + completion}
