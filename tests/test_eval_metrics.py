from __future__ import annotations

from typing import Any

from evals.run_evals import compute_quality_metrics

from eva_agent.domain.plan import TodoPlan
from eva_agent.state import Intent


def test_compute_quality_metrics_counts_routes_and_clarifications() -> None:
    evaluated: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (
            {"id": "answerable-data", "gold_route": "data", "clarify_warranted": False},
            {"intent": Intent(kind="mixed_diagnostic", confidence=0.9)},
        ),
        (
            {"id": "warranted-clarify", "gold_route": "clarify", "clarify_warranted": True},
            {
                "intent": Intent(kind="mixed_diagnostic", confidence=0.9),
                "todo_plan": TodoPlan(status="awaiting_clarification"),
            },
        ),
        (
            {"id": "avoidable-clarify", "gold_route": "data", "clarify_warranted": False},
            {"intent": Intent(kind="need_clarification", confidence=0.5)},
        ),
        (
            {"id": "legal", "gold_route": "legal", "clarify_warranted": False},
            {"intent": Intent(kind="legal_consult", confidence=0.9)},
        ),
    ]

    metrics = compute_quality_metrics(evaluated)

    assert metrics["route_ok"] == 3
    assert metrics["route_total"] == 4
    assert metrics["route_accuracy"] == 0.75
    assert metrics["clarify_precision"] == 0.5
    assert metrics["clarify_recall"] == 1.0
    assert metrics["avoidable_clarification_rate"] == 0.25
