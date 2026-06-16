"""Unit-тесты read-only мок-оберток (fixtures-режим детерминирован)."""

from __future__ import annotations

from eva_agent.mock.data import (
    OWNER_SCOPE,
    eva_get_contract,
    eva_get_contract_parties,
    eva_get_counterparty,
    eva_get_creative_status,
    eva_kktu_suggest,
    eva_list_contract_documents,
    eva_list_placements,
    eva_list_unsigned_contracts,
    eva_search_contracts,
)
from eva_agent.state import ApiFinding


def test_creative_ready_for_erid() -> None:
    finding = eva_get_creative_status("CR-1")
    assert finding.data["erid_token"]
    assert finding.data["blocking_reasons"] == []
    assert finding.owner_ref == OWNER_SCOPE


def test_creative_blocked_by_contract() -> None:
    finding = eva_get_creative_status("CR-2")
    assert finding.data["erid_token"] is None
    assert finding.data["blocking_reasons"]
    assert "ОРД" in finding.data["blocking_reasons"][0]


def test_unsigned_contracts_lists_ct2() -> None:
    finding = eva_list_unsigned_contracts()
    ids = [c["id"] for c in finding.data["contracts"]]
    assert "CT-2" in ids
    assert "CT-1" not in ids


def test_contract_parties() -> None:
    finding = eva_get_contract_parties("CT-2")
    roles = {p["role"] for p in finding.data["parties"]}
    assert {"customer", "executor"} <= roles


def test_get_contract_returns_api_finding() -> None:
    finding = eva_get_contract("CT-1")
    assert isinstance(finding, ApiFinding)
    assert finding.tool == "eva_get_contract"
    assert finding.data["number"] == "Д-2025/01"
    assert finding.data["source"] == "mock"


def test_get_counterparty_returns_api_finding() -> None:
    finding = eva_get_counterparty("CP-3")
    assert isinstance(finding, ApiFinding)
    assert finding.tool == "eva_get_counterparty"
    assert finding.data["name"] == "ООО Площадка"
    assert finding.data["ord_status"] == "pending"


def test_list_placements_returns_api_finding() -> None:
    finding = eva_list_placements("CT-2")
    assert isinstance(finding, ApiFinding)
    assert finding.tool == "eva_list_placements"
    assert finding.data["count"] == 1
    assert finding.data["placements"][0]["id"] == "PL-2"


def test_list_contract_documents_returns_missing_types() -> None:
    finding = eva_list_contract_documents("CT-2")
    assert isinstance(finding, ApiFinding)
    assert finding.tool == "eva_list_contract_documents"
    assert finding.data["missing"] == ["act"]


def test_search_contracts_finds_latest_fixture() -> None:
    finding = eva_search_contracts(q="последний договор")
    assert isinstance(finding, ApiFinding)
    assert finding.tool == "eva_search_contracts"
    assert finding.data["count"] == 1
    assert finding.data["contracts"][0]["number"] == "Д-2025/249"


def test_kktu_suggest_top3() -> None:
    finding = eva_kktu_suggest("реклама онлайн-курсов")
    assert len(finding.data["suggestions"]) == 3
