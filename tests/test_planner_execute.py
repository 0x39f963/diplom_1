from __future__ import annotations

from datetime import date
from typing import Any

from eva_agent.domain.plan import PlanStep, TodoItem, TodoPlan
from eva_agent.planner import execute as execute_module
from eva_agent.planner.execute import _apply_checklist, execute_plan
from eva_agent.planner.filters import apply_filters
from eva_agent.planner.validate import validate_plan
from eva_agent.state import ApiFinding


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


def test_execute_auto_wires_creative_to_contract() -> None:
    plan = _plan(
        "contract_card",
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
        ],
    )

    findings, executed = execute_plan(plan)

    assert [finding.tool for finding in findings] == [
        "eva_get_creative_status",
        "eva_get_contract",
    ]
    assert findings[1].args["contract_id"] == "CT-1"
    assert executed.items[1].status == "done"
    assert executed.status == "answered"
    assert any("auto-wire eva_get_contract.contract_id" in event for event in executed.trace)


def test_execute_selector_value_picks_one_counterparty() -> None:
    plan = _plan(
        "counterparty_card",
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
                inputs={"role": "customer"},
                tool_calls=[
                    PlanStep(
                        order=2,
                        tool="eva_get_counterparty",
                        args={"role": "customer"},
                    )
                ],
            ),
        ],
    )

    findings, executed = execute_plan(plan)

    counterparty_findings = [
        finding for finding in findings if finding.tool == "eva_get_counterparty"
    ]
    assert len(counterparty_findings) == 1
    assert counterparty_findings[0].args["counterparty_id"] == "CP-1"
    assert executed.items[1].status == "done"
    assert executed.status == "answered"


def test_execute_fan_out_counterparties_without_selector_value() -> None:
    plan = _plan(
        "counterparty_card",
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
                tool_calls=[PlanStep(order=2, tool="eva_get_counterparty", args={})],
            ),
        ],
    )

    findings, executed = execute_plan(plan)

    counterparty_ids = [
        finding.args["counterparty_id"]
        for finding in findings
        if finding.tool == "eva_get_counterparty"
    ]
    assert counterparty_ids == ["CP-1", "CP-2"]
    assert executed.items[1].status == "done"
    assert "fan-out eva_get_counterparty.counterparty_id[1]" in executed.trace
    assert "fan-out eva_get_counterparty.counterparty_id[2]" in executed.trace


def test_execute_blocks_empty_producer_without_calling_consumer(monkeypatch: Any) -> None:
    counterparty_calls: list[str] = []

    def fake_parties(contract_id: str) -> ApiFinding:
        return ApiFinding(
            tool="eva_get_contract_parties",
            args={"contract_id": contract_id},
            data={"id": contract_id, "parties": []},
        )

    def fake_counterparty(counterparty_id: str) -> ApiFinding:
        counterparty_calls.append(counterparty_id)
        return ApiFinding(
            tool="eva_get_counterparty",
            args={"counterparty_id": counterparty_id},
            data={"id": counterparty_id},
        )

    monkeypatch.setitem(execute_module.EXECUTION_REGISTRY, "eva_get_contract_parties", fake_parties)
    monkeypatch.setitem(execute_module.EXECUTION_REGISTRY, "eva_get_counterparty", fake_counterparty)
    plan = _plan(
        "counterparty_card",
        [
            TodoItem(
                id="get_contract_parties",
                order=1,
                inputs={"contract_id": "CT-empty"},
                tool_calls=[
                    PlanStep(
                        order=1,
                        tool="eva_get_contract_parties",
                        args={"contract_id": "CT-empty"},
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

    findings, executed = execute_plan(plan)

    assert [finding.tool for finding in findings] == ["eva_get_contract_parties"]
    assert counterparty_calls == []
    assert executed.status == "awaiting_clarification"
    assert "empty producer" in executed.items[1].blockers


def test_execute_blocks_ambiguous_producers_without_choosing_first() -> None:
    plan = _plan(
        "counterparty_card",
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

    findings, executed = execute_plan(plan)

    assert all(finding.tool != "eva_get_counterparty" for finding in findings)
    assert executed.items[2].status == "blocked"
    assert "ambiguous auto-wire: counterparty_id" in executed.items[2].blockers
    assert executed.status == "awaiting_clarification"


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


def test_validate_plan_todo_from_refs_are_strict() -> None:
    unknown = _plan(
        "counterparty_card",
        [
            TodoItem(
                id="get_counterparty",
                order=1,
                inputs={"counterparty_id": {"$from": {"todo": "missing", "path": "id"}}},
                tool_calls=[
                    PlanStep(
                        order=1,
                        tool="eva_get_counterparty",
                        args={"counterparty_id": {"$from": {"todo": "missing", "path": "id"}}},
                    )
                ],
            )
        ],
    )
    forward = _plan(
        "counterparty_card",
        [
            TodoItem(
                id="get_counterparty",
                order=1,
                tool_calls=[
                    PlanStep(
                        order=1,
                        tool="eva_get_counterparty",
                        args={
                            "counterparty_id": {
                                "$from": {"todo": "get_contract_parties", "path": "parties.0.counterparty_id"}
                            }
                        },
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
        ],
    )

    validated_unknown = validate_plan(unknown)
    validated_forward = validate_plan(forward)

    assert validated_unknown.status == "awaiting_clarification"
    assert any(
        "unknown $from todo" in blocker
        for item in validated_unknown.items
        for blocker in item.blockers
    )
    assert validated_forward.status == "awaiting_clarification"
    assert any(
        "forward $from todo" in blocker
        for item in validated_forward.items
        for blocker in item.blockers
    )


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
