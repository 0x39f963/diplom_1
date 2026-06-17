from __future__ import annotations

from typing import get_args

from eva_agent.domain.checklist import EntityCount, PlanningChecklist
from eva_agent.domain.plan import PlanTool
from eva_agent.domain.slice import DomainSlice
from eva_agent.tools.build_domain_map import (
    build_domain_map,
    make_slice,
    render_domain_slice,
    write_domain_map,
)


def test_build_domain_map_has_entities_with_access() -> None:
    domain_map = build_domain_map()
    entities = domain_map["entities"]

    assert len(entities) == 7
    for entity in entities.values():
        assert entity["fields"]
        assert entity["endpoints"]
        assert entity["tools"]


def test_relations_are_machine_edges_with_plan_tools() -> None:
    domain_map = build_domain_map()
    plan_tools = set(get_args(PlanTool))
    required_edges = {
        ("ContractParty", "Counterparty"),
        ("Creative", "Contract"),
        ("Contract", "Placement"),
        ("Contract", "Document"),
    }
    actual_edges = {
        (relation["source_entity"], relation["target_entity"])
        for relation in domain_map["relations"]
    }

    assert required_edges <= actual_edges
    for relation in domain_map["relations"]:
        assert relation["source_tool"] in plan_tools
        assert relation["source_path"]
        assert relation["target_tool"] in plan_tools
        assert relation["target_arg"]
        assert relation["cardinality"] in {"one", "many"}


def test_render_domain_slice_is_compact_and_filtered() -> None:
    text = render_domain_slice(["ContractParty", "Counterparty"], build_domain_map())

    assert "ContractParty" in text
    assert "Counterparty" in text
    assert "ContractParty->Counterparty" in text
    assert "Placement" not in text


def test_make_slice_contains_selected_relations() -> None:
    domain_slice = make_slice(["Contract", "ContractParty", "Counterparty"], build_domain_map())

    assert domain_slice.relations
    assert any(
        relation.source_entity == "ContractParty" and relation.target_entity == "Counterparty"
        for relation in domain_slice.relations
    )


def test_write_domain_map_is_idempotent(tmp_path) -> None:
    path = tmp_path / "domain_map.json"

    write_domain_map(path)
    first = path.read_bytes()
    write_domain_map(path)

    assert path.read_bytes() == first


def test_domain_slice_and_planning_checklist_round_trip() -> None:
    domain_slice = make_slice(["ContractParty", "Counterparty"], build_domain_map())
    checklist = PlanningChecklist(
        intent="parties",
        entities=["ContractParty", "Counterparty"],
        cardinality=[EntityCount(entity="Counterparty", ref_count=1)],
        access=["eva_get_contract_parties", "eva_get_counterparty"],
        needs_chain=True,
    )

    assert DomainSlice.model_validate(domain_slice.model_dump()).model_dump() == domain_slice.model_dump()
    assert PlanningChecklist.model_validate(checklist.model_dump()).model_dump() == checklist.model_dump()
