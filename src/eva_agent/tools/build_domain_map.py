"""Генератор машинной карты домена."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast, get_args

from eva_agent.domain.entity_map import CHAIN_HINTS, ENTITY_MAP, EntitySpec
from eva_agent.domain.plan import PlanTool
from eva_agent.domain.relations import RELATIONS
from eva_agent.domain.slice import DomainSlice, RelationSpec
from eva_agent.planner.catalog import AVAILABLE_TOOLS_DEFAULT

DOMAIN_MAP_PATH = Path(__file__).resolve().parents[1] / "domain" / "domain_map.json"

PLAN_TOOL_NAMES: frozenset[str] = frozenset(cast(tuple[str, ...], get_args(PlanTool)))
AVAILABLE_PLAN_TOOLS: frozenset[str] = PLAN_TOOL_NAMES & AVAILABLE_TOOLS_DEFAULT

_ENDPOINT_TOOLS: dict[str, tuple[str, ...]] = {
    "GET /api/contracts": ("eva_list_unsigned_contracts", "eva_search_contracts"),
    "GET /api/contracts/{id}": ("eva_get_contract",),
    "GET /api/contracts/{id}/parties": ("eva_get_contract_parties",),
    "GET /api/counterparties/{id}": ("eva_get_counterparty",),
    "GET /api/creatives/{id}": ("eva_get_creative_status",),
    "GET /api/creatives/{id}/media": ("eva_get_creative_status",),
    "GET /api/contracts/{id}/placements": ("eva_list_placements",),
    "GET /api/placements?contract_id=": ("eva_list_placements",),
    "GET /api/contracts/{id}/documents": (
        "eva_list_contract_documents",
        "eva_doc_read",
        "eva_doc_download",
    ),
}


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _tools_for_entity(entity: EntitySpec) -> list[str]:
    tools: list[str] = []
    for endpoint in entity.endpoints:
        for tool in _ENDPOINT_TOOLS.get(endpoint, ()):
            if tool in AVAILABLE_PLAN_TOOLS:
                _append_unique(tools, tool)
    return tools


def build_domain_map() -> dict[str, Any]:
    """Собрать машинную карту домена из локальных описаний."""

    entities: dict[str, dict[str, Any]] = {}
    for entity in ENTITY_MAP:
        entities[entity.name] = {
            "description": entity.description,
            "fields": list(entity.key_fields),
            "endpoints": list(entity.endpoints),
            "operations": list(entity.operations),
            "statuses": list(entity.statuses),
            "roles": list(entity.roles),
            "tools": _tools_for_entity(entity),
        }

    return {
        "version": 1,
        "entities": entities,
        "relations": [relation.model_dump() for relation in RELATIONS],
        "chains": list(CHAIN_HINTS),
    }


def write_domain_map(path: Path = DOMAIN_MAP_PATH) -> dict[str, Any]:
    """Записать детерминированный JSON карты домена."""

    domain_map = build_domain_map()
    path.write_text(
        json.dumps(domain_map, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return domain_map


def load_domain_map() -> dict[str, Any]:
    """Прочитать сохраненную карту домена или собрать ее из локальных описаний."""

    if not DOMAIN_MAP_PATH.exists():
        return build_domain_map()
    data = json.loads(DOMAIN_MAP_PATH.read_text(encoding="utf-8"))
    return cast(dict[str, Any], data)


def _selected_entities(entities: list[str], domain_map: dict[str, Any]) -> list[str]:
    all_entities = cast(dict[str, Any], domain_map["entities"])
    selected: list[str] = []
    for entity in entities:
        if entity in all_entities:
            _append_unique(selected, entity)
    return selected


def _slice_relations(entities: list[str], domain_map: dict[str, Any]) -> list[RelationSpec]:
    selected = set(entities)
    relations = []
    for raw_relation in cast(list[dict[str, Any]], domain_map["relations"]):
        relation = RelationSpec.model_validate(raw_relation)
        if relation.source_entity in selected and relation.target_entity in selected:
            relations.append(relation)
    return relations


def render_domain_slice(entities: list[str], domain_map: dict[str, Any] | None = None) -> str:
    """Сформировать компактный текст по выбранным сущностям."""

    domain_map = domain_map or load_domain_map()
    selected = _selected_entities(entities, domain_map)
    all_entities = cast(dict[str, dict[str, Any]], domain_map["entities"])

    lines: list[str] = []
    for entity_name in selected:
        entity = all_entities[entity_name]
        fields = ",".join(cast(list[str], entity["fields"]))
        operations = ",".join(cast(list[str], entity.get("operations", []))) or "-"
        roles = ",".join(cast(list[str], entity.get("roles", []))) or "-"
        statuses = ",".join(cast(list[str], entity.get("statuses", []))) or "-"
        tools = ",".join(cast(list[str], entity["tools"]))
        lines.append(
            f"{entity_name}: f {fields}; ops {operations}; roles {roles}; "
            f"statuses {statuses}; tools {tools}."
        )

    relation_text = []
    for relation in _slice_relations(selected, domain_map):
        relation_text.append(
            f"{relation.source_entity}->{relation.target_entity} "
            f"{relation.source_path}->{relation.target_tool}.{relation.target_arg}"
        )
    lines.append("Relations: " + ("; ".join(relation_text) if relation_text else "-") + ".")
    return "\n".join(lines)


def make_slice(entities: list[str], domain_map: dict[str, Any] | None = None) -> DomainSlice:
    """Собрать типизированный доменный срез."""

    domain_map = domain_map or load_domain_map()
    selected = _selected_entities(entities, domain_map)
    all_entities = cast(dict[str, dict[str, Any]], domain_map["entities"])

    tools: list[str] = []
    for entity_name in selected:
        for tool in cast(list[str], all_entities[entity_name]["tools"]):
            _append_unique(tools, tool)

    return DomainSlice(
        entities=selected,
        relations=_slice_relations(selected, domain_map),
        tools=tools,
    )


def main() -> None:
    write_domain_map()


if __name__ == "__main__":
    main()
