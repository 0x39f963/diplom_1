from __future__ import annotations

from eva_agent.domain.frame import PlanningFrame
from eva_agent.nlu.preprocess import preprocess
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


def test_compile_roleless_single_party_fetches_all_parties() -> None:
    plan = compile_plan(
        _frame(
            target="ContractParty",
            relation="parties",
            selector={"contract_id": "CT-1"},
        )
    )

    tools = [step.tool for item in plan.items for step in item.tool_calls]
    assert plan.status == "in_progress"
    assert plan.protocol_id == "party_lookup"
    assert "eva_get_contract_parties" in tools
    assert "resolve_party_role" not in [item.id for item in plan.items]


def test_compile_counterparty_status_by_contract_fetches_parties_chain() -> None:
    plan = compile_plan(
        _frame(
            target="ContractParty",
            relation="parties",
            fields=["status"],
            selector={"contract_id": "CT-2"},
        )
    )

    tools = [step.tool for item in plan.items for step in item.tool_calls]
    assert plan.status == "in_progress"
    assert tools == ["eva_get_contract_parties", "eva_get_counterparty"]


def test_high_confidence_compiled_plan_does_not_clarify() -> None:
    plan = compile_plan(_frame(selector={"contract_id": "CT-1"}, confidence=0.9))

    assert plan.status == "in_progress"
    assert plan.clarify_question == ""
    assert plan.clarify_code == ""


def test_contract_number_card_uses_search_as_resolver() -> None:
    plan = compile_plan(
        _frame(
            target="Contract",
            selector={"contract_id": "Д-2025/249"},
            output="card",
        )
    )

    by_id = {item.id: item for item in plan.items}
    tools = [step.tool for item in plan.items for step in item.tool_calls]

    assert plan.status == "in_progress"
    assert plan.protocol_id == "contract_card"
    assert plan.strategy == "compiled:contract_card_via_search"
    assert tools == ["eva_search_contracts", "eva_get_contract"]
    assert by_id["get_contract"].depends_on == [2]
    ref = by_id["get_contract"].tool_calls[0].args["contract_id"]["$from"]
    assert ref == {
        "todo": "search_contracts",
        "path": "contracts[].id",
        "cardinality": "one",
    }


def test_search_party_role_uses_resolver_then_parties() -> None:
    plan = compile_plan(
        _frame(
            operation="list",
            target="ContractParty",
            relation="parties",
            fields=["search"],
            cardinality="one",
            selector={"role": "executor", "search_query": "найди договор с площадкой"},
            output="list",
        )
    )

    by_id = {item.id: item for item in plan.items}
    tools = [step.tool for item in plan.items for step in item.tool_calls]

    assert plan.status == "in_progress"
    assert plan.protocol_id == "party_lookup"
    assert tools[:2] == ["eva_search_contracts", "eva_get_contract_parties"]
    assert by_id["get_contract_parties"].depends_on == [2]
    assert by_id["resolve_party_role"].inputs["role"] == "executor"
    ref = by_id["get_contract_parties"].tool_calls[0].args["contract_id"]["$from"]
    assert ref["todo"] == "search_contracts"
    assert ref["cardinality"] == "one"


def test_known_contract_id_does_not_use_search_resolver() -> None:
    plan = compile_plan(_frame(selector={"contract_id": "CT-1"}, output="card"))

    tools = [step.tool for item in plan.items for step in item.tool_calls]
    assert tools == ["eva_get_contract"]
    assert plan.strategy == "compiled:contract_card"


def test_mixed_legal_placements_adds_retrieve_legal_to_data_plan() -> None:
    plan = compile_plan(
        _frame(
            operation="list",
            target="Placement",
            relation="placements",
            fields=["legal_signal"],
            cardinality="all",
            selector={"contract_id": "CT-1", "legal_query": "маркировка рекламы erid"},
            output="list",
        )
    )

    by_id = {item.id: item for item in plan.items}
    tools = [step.tool for item in plan.items for step in item.tool_calls]

    assert plan.status == "in_progress"
    assert plan.protocol_id == "placement_list"
    assert "eva_list_placements" in tools
    assert "retrieve_legal" in tools
    assert by_id["legal_lookup"].depends_on == []
    assert by_id["legal_lookup"].tool_calls[0].args["query"] == "маркировка рекламы erid"


def test_coverage_adds_contract_card_parties_and_documents() -> None:
    plan = compile_plan(
        _frame(
            target="Contract",
            selector={"contract_id": "CT-2"},
            output="card",
        ),
        nlu=preprocess("карточка договора CT-2, стороны и документы"),
    )

    tools = [step.tool for item in plan.items for step in item.tool_calls]
    assert plan.status == "in_progress"
    assert tools == [
        "eva_get_contract",
        "eva_get_contract_parties",
        "eva_list_contract_documents",
    ]
    assert plan.coverage["missing"] == []
    assert set(plan.coverage["covered"]) >= {
        "Contract.card",
        "Contract.parties",
        "Contract.documents",
    }


def test_coverage_adds_creative_status_after_placements() -> None:
    plan = compile_plan(
        _frame(
            operation="list",
            target="Placement",
            relation="placements",
            cardinality="all",
            selector={"contract_id": "CT-1"},
            output="list",
        ),
        nlu=preprocess("какие размещения по CT-1 и какие креативы"),
    )

    by_id = {item.id: item for item in plan.items}
    tools = [step.tool for item in plan.items for step in item.tool_calls]

    assert plan.status == "in_progress"
    assert tools == ["eva_list_placements", "eva_get_creative_status"]
    ref = by_id["get_creative_status"].tool_calls[0].args["creative_id"]["$from"]
    assert ref == {
        "todo": "list_placements",
        "path": "placements[].creative_id",
        "cardinality": "many",
        "fan_out": True,
    }
    assert by_id["get_creative_status"].depends_on == [2]
    assert plan.coverage["missing"] == []


def test_coverage_adds_legal_rules_from_nlu_signal() -> None:
    plan = compile_plan(
        _frame(
            operation="list",
            target="Placement",
            relation="placements",
            cardinality="all",
            selector={"contract_id": "CT-1"},
            output="list",
        ),
        nlu=preprocess("нужна ли маркировка для размещений CT-1, покажи размещения"),
    )

    tools = [step.tool for item in plan.items for step in item.tool_calls]
    assert plan.status == "in_progress"
    assert tools == ["eva_list_placements", "retrieve_legal"]
    assert plan.coverage["missing"] == []
    assert "Legal.rules" in plan.coverage["covered"]


def test_coverage_keeps_single_atom_contract_card_unchanged() -> None:
    plan = compile_plan(
        _frame(selector={"contract_id": "CT-1"}, output="card"),
        nlu=preprocess("покажи карточку CT-1"),
    )

    tools = [step.tool for item in plan.items for step in item.tool_calls]
    assert plan.status == "in_progress"
    assert tools == ["eva_get_contract"]
    assert plan.coverage["missing"] == []


def test_coverage_keeps_direct_counterparty_card_unchanged() -> None:
    plan = compile_plan(
        _frame(
            target="Counterparty",
            selector={"counterparty_id": "CP-1"},
            output="card",
        ),
        nlu=preprocess("покажи карточку контрагента CP-1"),
    )

    tools = [step.tool for item in plan.items for step in item.tool_calls]
    assert plan.status == "in_progress"
    assert tools == ["eva_get_counterparty"]
    assert plan.coverage["missing"] == []


def test_coverage_keeps_direct_creative_status_unchanged() -> None:
    plan = compile_plan(
        _frame(
            operation="diagnose",
            target="Creative",
            fields=["status"],
            selector={"creative_id": "CR-1"},
            output="value",
        ),
        nlu=preprocess("проверь статус креатива CR-1"),
    )

    tools = [step.tool for item in plan.items for step in item.tool_calls]
    assert plan.status == "in_progress"
    assert tools == ["eva_get_creative_status"]
    assert plan.coverage["missing"] == []


def test_coverage_keeps_direct_document_read_unchanged() -> None:
    plan = compile_plan(
        _frame(
            target="Document",
            relation="documents",
            selector={"doc_id": "DOC-1"},
            output="card",
        ),
        nlu=preprocess("прочитай документ DOC-1"),
    )

    tools = [step.tool for item in plan.items for step in item.tool_calls]
    assert plan.status == "in_progress"
    assert tools == ["eva_doc_read"]
    assert plan.coverage["missing"] == []


def test_low_confidence_with_explicit_id_compiles() -> None:
    plan = compile_plan(_frame(selector={"contract_id": "CT-1"}, confidence=0.2))

    assert plan.status == "in_progress"
    assert plan.clarify_code == ""
    assert "low confidence" in " ".join(plan.trace)


def test_low_confidence_without_domain_signal_asks_targeted_clarify() -> None:
    plan = compile_plan(_frame(confidence=0.2))

    assert plan.status == "awaiting_clarification"
    assert plan.clarify_code == "low_confidence"
    assert "сущность" in plan.clarify_question


def test_needs_clarification_is_overridden_when_safe_plan_exists() -> None:
    plan = compile_plan(
        _frame(
            target="Document",
            relation="documents",
            fields=["missing"],
            selector={"contract_id": "CT-2"},
            needs_clarification=True,
            clarify_reason="низкая уверенность разбора",
            confidence=0.2,
        )
    )

    tools = [step.tool for item in plan.items for step in item.tool_calls]
    assert plan.status == "in_progress"
    assert "eva_list_contract_documents" in tools
    assert plan.clarify_code == ""
    assert "clarify overridden: safe read-only plan exists" in plan.trace


def test_needs_clarification_uses_fallback_read_plan() -> None:
    plan = compile_plan(
        _frame(
            target="Document",
            relation="documents",
            selector={"contract_id": "CT-2"},
            needs_clarification=True,
            clarify_reason="нет doc_id",
            confidence=0.2,
        )
    )

    tools = [step.tool for item in plan.items for step in item.tool_calls]
    assert plan.status == "in_progress"
    assert tools == ["eva_list_contract_documents"]
    assert "clarify overridden: safe read-only plan exists" in plan.trace


def test_needs_clarification_without_domain_signal_stays_clarify() -> None:
    plan = compile_plan(
        _frame(
            needs_clarification=True,
            clarify_reason="уточните договор",
        )
    )

    assert plan.status == "awaiting_clarification"
    assert plan.protocol_id == "clarify_first"


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
