from __future__ import annotations

from eva_agent.domain.frame import PlanningFrame
from eva_agent.planner.compile import compile_plan, rank_protocol_cards


def _frame(**updates) -> PlanningFrame:
    payload = {
        "operation": "read",
        "target": "Contract",
        "relation": None,
        "fields": [],
        "filters": {"date_hint": "none", "status": []},
        "cardinality": "one",
        "selector": {},
        "output": "summary",
        "subtasks": [],
        "needs_clarification": False,
        "clarify_reason": "",
        "confidence": 0.9,
    }
    payload.update(updates)
    return PlanningFrame.model_validate(payload)


def test_compile_party_customer_builds_chain_with_role_selector() -> None:
    plan = compile_plan(
        _frame(
            target="ContractParty",
            relation="parties",
            selector={"contract_id": "CT-1", "role": "customer"},
            output="card",
        )
    )

    by_id = {item.id: item for item in plan.items}
    assert plan.protocol_id == "party_lookup"
    assert by_id["get_contract_parties"].tool_calls[0].tool == "eva_get_contract_parties"
    assert by_id["resolve_party_role"].inputs["role"] == "customer"
    assert by_id["get_counterparty"].tool_calls[0].tool == "eva_get_counterparty"
    ref = by_id["get_counterparty"].tool_calls[0].args["counterparty_id"]["$from"]
    assert ref["todo"] == "get_contract_parties"
    assert ref["selector"] == "role"
    assert ref["selector_value"] == "customer"
    assert by_id["get_counterparty"].depends_on == [2]


def test_compile_all_parties_marks_fan_out() -> None:
    plan = compile_plan(
        _frame(
            operation="list",
            target="ContractParty",
            relation="parties",
            cardinality="all",
            selector={"contract_id": "CT-1"},
            output="list",
        )
    )

    get_counterparty = next(item for item in plan.items if item.id == "get_counterparty")
    assert get_counterparty.inputs["fan_out"] is True
    ref = get_counterparty.tool_calls[0].args["counterparty_id"]["$from"]
    assert ref["fan_out"] is True
    assert "selector_value" not in ref


def test_compile_unsigned_overview_uses_list_unsigned_tool() -> None:
    plan = compile_plan(
        _frame(
            operation="list",
            target="Contract",
            filters={"date_hint": "none", "status": ["unsigned"]},
            cardinality="all",
            output="list",
        )
    )

    assert plan.protocol_id == "overview"
    build_overview = next(item for item in plan.items if item.id == "build_overview")
    assert build_overview.tool_calls[0].tool == "eva_list_unsigned_contracts"
    assert build_overview.tool_calls[0].status_hint == "unsigned"


def test_compile_unsupported_frame_awaits_clarification() -> None:
    plan = compile_plan(_frame(target="UnknownThing"))

    assert plan.status == "awaiting_clarification"
    assert plan.protocol_id == "clarify_first"
    assert "протокола" in plan.clarify_question or "сущность" in plan.clarify_question


def test_scoring_creatives_by_contract_prefers_placements_over_contract_card() -> None:
    frame = _frame(
        operation="list",
        target="Creative",
        relation="placements",
        cardinality="all",
        selector={"contract_id": "CT-2"},
        output="list",
    )

    ranked = rank_protocol_cards(frame)
    plan = compile_plan(frame)

    assert ranked[0].card.id == "creatives_by_contract"
    assert plan.protocol_id == "placement_list"
    assert [step.tool for item in plan.items for step in item.tool_calls] == ["eva_list_placements"]


def test_compile_roleless_single_party_asks_to_clarify() -> None:
    plan = compile_plan(
        _frame(
            target="ContractParty",
            relation="parties",
            selector={"contract_id": "CT-1"},
        )
    )

    assert plan.status == "awaiting_clarification"
    assert "роль" in plan.clarify_question


def test_high_confidence_compiled_plan_does_not_clarify() -> None:
    plan = compile_plan(_frame(selector={"contract_id": "CT-1"}, confidence=0.9))

    assert plan.status == "in_progress"
    assert plan.clarify_question == ""
    assert plan.clarify_code == ""


def test_low_confidence_frame_asks_targeted_clarify() -> None:
    plan = compile_plan(_frame(selector={"contract_id": "CT-1"}, confidence=0.2))

    assert plan.status == "awaiting_clarification"
    assert plan.clarify_code == "low_confidence"
    assert "сущность" in plan.clarify_question


def test_write_like_missing_slot_uses_write_confirm_code() -> None:
    plan = compile_plan(
        _frame(
            operation="download",
            target="Document",
            relation="documents",
            selector={"contract_id": "CT-1"},
            output="value",
            confidence=0.9,
        )
    )

    assert plan.status == "awaiting_clarification"
    assert plan.clarify_code == "write_confirm"
    assert "документ" in plan.clarify_question


def test_compile_composite_plan_merges_subtasks_in_order() -> None:
    plan = compile_plan(
        _frame(
            target="Creative",
            selector={"creative_id": "CR-1"},
            output="card",
            subtasks=[
                _frame(target="Creative", selector={"creative_id": "CR-1"}, output="card"),
                _frame(
                    operation="list",
                    target="Document",
                    relation="documents",
                    cardinality="all",
                    selector={"contract_id": "CT-1"},
                    output="list",
                ),
            ],
            confidence=0.9,
        )
    )

    tools = [step.tool for item in plan.items for step in item.tool_calls]
    assert plan.strategy == "compiled:composite"
    assert tools == ["eva_get_creative_status", "eva_list_contract_documents"]
    list_documents = next(item for item in plan.items if item.id == "list_documents")
    assert list_documents.depends_on
