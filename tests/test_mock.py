"""Unit-тесты read-only мок-оберток (fixtures-режим детерминирован)."""

from __future__ import annotations

from eva_agent.mock.data import (
    OWNER_SCOPE,
    eva_get_contract_parties,
    eva_get_creative_status,
    eva_kktu_suggest,
    eva_list_unsigned_contracts,
)


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


def test_kktu_suggest_top3() -> None:
    finding = eva_kktu_suggest("реклама онлайн-курсов")
    assert len(finding.data["suggestions"]) == 3
