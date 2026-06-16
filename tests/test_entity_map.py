from __future__ import annotations

from datetime import date

from eva_agent.domain.entities import Contract, ContractType, OrdStatus
from eva_agent.domain.entity_map import (
    CONTRACT_NUMBER_EXAMPLE,
    SYNTHETIC_ID_CONVENTIONS,
    render_entity_map,
)


def test_entities_import_and_validate() -> None:
    contract = Contract(
        id="CT-1",
        contract_number=CONTRACT_NUMBER_EXAMPLE,
        contract_date=date(2025, 1, 1),
        contract_type=ContractType.service,
        chain_role="initial",
        ord_status=OrdStatus.draft,
    )

    assert contract.id == "CT-1"
    assert contract.contract_number == CONTRACT_NUMBER_EXAMPLE


def test_render_entity_map_is_compact() -> None:
    text = render_entity_map()

    assert text
    assert len(text) <= 1500
    for entity_name in (
        "Contract",
        "ContractParty",
        "Counterparty",
        "Creative",
        "CreativeMedia",
        "Placement",
        "Document",
    ):
        assert entity_name in text
    assert SYNTHETIC_ID_CONVENTIONS["counterparty"] == "CP-N"
    assert SYNTHETIC_ID_CONVENTIONS["document"] == "DOC-N"
    assert SYNTHETIC_ID_CONVENTIONS["placement"] == "PL-N"
