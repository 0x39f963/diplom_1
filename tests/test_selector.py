from __future__ import annotations

from eva_agent.state import RetrievalResult
from eva_agent.tools import selector as selector_module
from eva_agent.tools.entity_ref import extract_refs
from eva_agent.tools.selector import EXECUTION_REGISTRY, TOOL_REGISTRY, select_tools

_TOOL_ARGS = {
    "eva_get_contract": {"contract_id": "CT-1"},
    "eva_get_contract_parties": {"contract_id": "CT-1"},
    "eva_get_counterparty": {"counterparty_id": "CP-1"},
    "eva_get_creative_status": {"creative_id": "CR-1"},
    "eva_list_contract_documents": {"contract_id": "CT-2"},
    "eva_list_placements": {"contract_id": "CT-1"},
    "eva_list_unsigned_contracts": {},
    "eva_search_contracts": {"q": "последний договор"},
    "eva_kktu_suggest": {"description": "реклама услуг"},
}


def test_tool_registry_names_match_api_finding_tool() -> None:
    for name, fn in TOOL_REGISTRY.items():
        finding = fn(**_TOOL_ARGS[name])
        assert finding.tool == name


def test_execution_registry_contains_rag_wrapper() -> None:
    assert EXECUTION_REGISTRY["retrieve_legal"] is selector_module.eva_retrieve_legal
    assert TOOL_REGISTRY["eva_kktu_suggest"] is selector_module.eva_kktu_suggest


def test_select_tools_for_contract_parties_and_card() -> None:
    query = "Под каким номером проходит договор CT-1 и кто его стороны?"
    refs = extract_refs(query)

    selected = select_tools(query, refs)

    assert "eva_get_contract" in selected
    assert "eva_get_contract_parties" in selected


def test_select_tools_for_counterparty_creative_docs_and_search() -> None:
    assert select_tools("статус контрагента CP-1", extract_refs("CP-1")) == [
        "eva_get_counterparty"
    ]
    assert select_tools("что мешает креативу CR-2", extract_refs("CR-2")) == [
        "eva_get_creative_status"
    ]
    assert "eva_list_contract_documents" in select_tools(
        "каких документов не хватает по договору CT-2", extract_refs("CT-2")
    )
    assert select_tools("покажи последний договор", extract_refs("покажи последний договор")) == [
        "eva_search_contracts"
    ]


def test_retrieve_legal_wrapper_returns_empty_chunks_when_rag_is_down(monkeypatch) -> None:
    def fail(query: str) -> RetrievalResult:
        raise RuntimeError(query)

    monkeypatch.setattr(selector_module, "_retrieve_legal", fail)

    finding = selector_module.eva_retrieve_legal("маркировка рекламы")

    assert finding.tool == "retrieve_legal"
    assert finding.data["chunks"] == []
    assert finding.data["citations"] == []
