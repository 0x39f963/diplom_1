from __future__ import annotations

from eva_agent.domain.plan import PlanStep, TodoItem, TodoPlan
from eva_agent.planner.bind import bind_plan


def _plan(items: list[TodoItem]) -> TodoPlan:
    return TodoPlan(
        goal="test",
        protocol_id="contract_card",
        items=items,
        status="in_progress",
        confidence=0.9,
    )


def test_bind_auto_wires_creative_to_contract() -> None:
    plan = _plan(
        [
            TodoItem(
                id="get_creative_status",
                order=1,
                inputs={"creative_id": "CR-1"},
                tool_calls=[
                    PlanStep(
                        order=1,
                        tool="eva_get_creative_status",
                        args={"creative_id": "CR-1"},
                    )
                ],
            ),
            TodoItem(
                id="get_contract",
                type="dependent",
                order=2,
                tool_calls=[PlanStep(order=2, tool="eva_get_contract", args={})],
            ),
        ]
    )

    report = bind_plan(plan)

    assert len(report.auto_wired) == 1
    ref = plan.items[1].tool_calls[0].args["contract_id"]["$from"]
    assert ref == {
        "todo": "get_creative_status",
        "path": "contract_id",
        "selector": None,
        "cardinality": "one",
    }
    assert plan.items[1].depends_on == [1]
    assert plan.trace == [
        "auto-wire eva_get_contract.contract_id <- get_creative_status.contract_id"
    ]


def test_bind_remaps_legacy_step_ref_to_todo_ref() -> None:
    plan = _plan(
        [
            TodoItem(
                id="get_creative_status",
                order=1,
                inputs={"creative_id": "CR-1"},
                tool_calls=[
                    PlanStep(
                        order=1,
                        tool="eva_get_creative_status",
                        args={"creative_id": "CR-1"},
                    )
                ],
            ),
            TodoItem(
                id="get_contract",
                type="dependent",
                order=2,
                depends_on=[1],
                tool_calls=[
                    PlanStep(
                        order=2,
                        tool="eva_get_contract",
                        args={"contract_id": {"$from": {"step": 1, "path": "contract_id"}}},
                    )
                ],
            ),
        ]
    )

    report = bind_plan(plan)

    assert len(report.remapped) == 1
    assert plan.items[1].tool_calls[0].args["contract_id"] == {
        "$from": {"todo": "get_creative_status", "path": "contract_id"}
    }


def test_bind_auto_wire_carries_producer_selector_value() -> None:
    plan = TodoPlan(
        goal="test",
        protocol_id="counterparty_card",
        status="in_progress",
        confidence=0.9,
        items=[
            TodoItem(
                id="resolve_party_role",
                order=1,
                inputs={"contract_id": "CT-1", "role": "customer"},
                tool_calls=[
                    PlanStep(
                        order=1,
                        tool="eva_get_contract_parties",
                        args={"contract_id": "CT-1"},
                    )
                ],
            ),
            TodoItem(
                id="get_counterparty",
                type="dependent",
                order=2,
                tool_calls=[PlanStep(order=2, tool="eva_get_counterparty", args={})],
            ),
        ],
    )

    report = bind_plan(plan)

    assert len(report.auto_wired) == 1
    assert report.auto_wired[0].selector_value == "customer"
    assert plan.items[1].tool_calls[0].args["counterparty_id"]["$from"] == {
        "todo": "resolve_party_role",
        "path": "parties[].counterparty_id",
        "selector": "role",
        "cardinality": "many",
        "selector_value": "customer",
    }


def test_bind_reports_multiple_producers_without_choosing_first() -> None:
    plan = TodoPlan(
        goal="test",
        protocol_id="counterparty_card",
        status="in_progress",
        confidence=0.9,
        items=[
            TodoItem(
                id="get_contract_parties",
                order=1,
                inputs={"contract_id": "CT-1"},
                tool_calls=[
                    PlanStep(
                        order=1,
                        tool="eva_get_contract_parties",
                        args={"contract_id": "CT-1"},
                    )
                ],
            ),
            TodoItem(
                id="resolve_party_role",
                order=2,
                inputs={"contract_id": "CT-2"},
                tool_calls=[
                    PlanStep(
                        order=2,
                        tool="eva_get_contract_parties",
                        args={"contract_id": "CT-2"},
                    )
                ],
            ),
            TodoItem(
                id="get_counterparty",
                type="dependent",
                order=3,
                tool_calls=[PlanStep(order=3, tool="eva_get_counterparty", args={})],
            ),
        ],
    )

    report = bind_plan(plan)

    assert report.auto_wired == []
    assert len(report.ambiguous) == 1
    assert set(report.ambiguous[0].producer_ids) == {"get_contract_parties", "resolve_party_role"}
    assert "counterparty_id" not in plan.items[2].tool_calls[0].args
    assert any("ambiguous eva_get_counterparty.counterparty_id" in event for event in plan.trace)


def test_bind_is_idempotent_after_auto_wire() -> None:
    plan = _plan(
        [
            TodoItem(
                id="get_creative_status",
                order=1,
                inputs={"creative_id": "CR-1"},
                tool_calls=[
                    PlanStep(
                        order=1,
                        tool="eva_get_creative_status",
                        args={"creative_id": "CR-1"},
                    )
                ],
            ),
            TodoItem(
                id="get_contract",
                type="dependent",
                order=2,
                tool_calls=[PlanStep(order=2, tool="eva_get_contract", args={})],
            ),
        ]
    )

    bind_plan(plan)
    snapshot = plan.model_dump()
    second = bind_plan(plan)

    assert second.remapped == []
    assert second.auto_wired == []
    assert plan.model_dump() == snapshot
