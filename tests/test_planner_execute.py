from __future__ import annotations

from datetime import date

from eva_agent.domain.plan import PlanStep, TodoItem, TodoPlan
from eva_agent.planner.execute import _apply_checklist, execute_plan
from eva_agent.planner.filters import apply_filters
from eva_agent.planner.validate import validate_plan


def _plan(protocol_id: str, items: list[TodoItem]) -> TodoPlan:
    return TodoPlan(
        goal="test",
        protocol_id=protocol_id,
        items=items,
        status="in_progress",
        confidence=0.9,
    )


def test_execute_resolves_from_chain_to_counterparty() -> None:
    plan = _plan(
        "party_lookup",
        [
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
                id="get_counterparty",
                type="dependent",
                order=2,
                depends_on=[1],
                inputs={"counterparty_id": {"$from": {"step": 1, "path": "parties.0.counterparty_id"}}},
                tool_calls=[
                    PlanStep(
                        order=2,
                        tool="eva_get_counterparty",
                        args={
                            "counterparty_id": {
                                "$from": {"step": 1, "path": "parties.0.counterparty_id"}
                            }
                        },
                    )
                ],
            ),
            TodoItem(id="resolve_party_role", order=3),
        ],
    )

    findings, executed = execute_plan(plan)

    assert [finding.tool for finding in findings[:2]] == [
        "eva_get_contract_parties",
        "eva_get_counterparty",
    ]
    first_counterparty = findings[0].data["parties"][0]["counterparty_id"]
    assert findings[1].args["counterparty_id"] == first_counterparty
    assert executed.items[1].status == "done"


def test_execute_blocks_unknown_tool_without_crashing() -> None:
    step = PlanStep.model_construct(
        order=1,
        tool="eva_unknown",
        args={},
        date_hint="none",
        status_hint="none",
        reason="",
    )
    todo = TodoItem(id="get_contract", order=1, tool_calls=[step])
    plan = _plan("contract_card", [todo])

    findings, executed = execute_plan(plan)

    assert findings == []
    assert executed.items[0].status == "blocked"
    assert "unknown tool: eva_unknown" in executed.items[0].blockers


def test_checklist_blocked_mandatory_awaits_clarification() -> None:
    plan = _plan(
        "party_lookup",
        [
            TodoItem(
                id="resolve_party_role",
                order=1,
                status="blocked",
                blockers=["нет contract_id"],
            )
        ],
    )

    checked = _apply_checklist(plan)

    assert checked.status == "awaiting_clarification"
    assert "нет contract_id" in checked.clarify_question


def test_checklist_mixed_legal_data_requires_done_data_todo() -> None:
    plan = _plan(
        "mixed_legal_data",
        [
            TodoItem(id="legal_lookup", order=1, status="done"),
            TodoItem(id="legal_check_against_data", order=2, status="done"),
        ],
    )

    checked = _apply_checklist(plan)

    assert checked.status == "awaiting_clarification"
    assert "data-todo" in checked.clarify_question


def test_validate_plan_duplicate_step_order_awaits_clarification() -> None:
    plan = _plan(
        "contract_card",
        [
            TodoItem(
                id="get_contract",
                order=1,
                inputs={"contract_id": "CT-1"},
                tool_calls=[
                    PlanStep(order=1, tool="eva_get_contract", args={"contract_id": "CT-1"}),
                    PlanStep(order=1, tool="eva_get_contract", args={"contract_id": "CT-1"}),
                ],
            )
        ],
    )

    validated = validate_plan(plan)

    assert validated.status == "awaiting_clarification"
    assert "step order" in validated.clarify_question


def test_validate_plan_dependency_cycle_awaits_clarification() -> None:
    plan = _plan(
        "contract_card",
        [
            TodoItem(id="get_contract", type="dependent", order=1, depends_on=[2]),
            TodoItem(id="search_contracts", type="dependent", order=2, depends_on=[1]),
        ],
    )

    validated = validate_plan(plan)

    assert validated.status == "awaiting_clarification"
    assert any("dependency cycle" in blocker for item in validated.items for blocker in item.blockers)


def test_validate_plan_forward_from_awaits_clarification() -> None:
    plan = _plan(
        "party_lookup",
        [
            TodoItem(
                id="get_counterparty",
                order=1,
                inputs={"counterparty_id": {"$from": {"step": 2, "path": "id"}}},
                tool_calls=[
                    PlanStep(
                        order=1,
                        tool="eva_get_counterparty",
                        args={"counterparty_id": {"$from": {"step": 2, "path": "id"}}},
                    )
                ],
            ),
            TodoItem(
                id="get_contract_parties",
                order=2,
                inputs={"contract_id": "CT-1"},
                tool_calls=[
                    PlanStep(
                        order=2,
                        tool="eva_get_contract_parties",
                        args={"contract_id": "CT-1"},
                    )
                ],
            ),
            TodoItem(id="resolve_party_role", order=3),
        ],
    )

    validated = validate_plan(plan)

    assert validated.status == "awaiting_clarification"
    assert any("forward $from step" in blocker for item in validated.items for blocker in item.blockers)


def test_apply_filters_changes_args_deterministically() -> None:
    step = PlanStep(
        order=1,
        tool="eva_search_contracts",
        args={"q": "последние договоры"},
        date_hint="yesterday",
        status_hint="unsigned",
    )

    args = apply_filters(step, dict(step.args), today=date(2026, 6, 16))

    assert args["date_from"] == "2026-06-15"
    assert args["date_to"] == "2026-06-15"
    assert args["date_hint"] == "2026-06-15"
    assert args["statuses"] == ["draft", "pending", "sent"]
    assert args["status_hint"] == "unsigned"
