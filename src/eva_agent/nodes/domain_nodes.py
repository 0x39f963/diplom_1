"""Domain selector node for mixed diagnostic requests."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, cast

from eva_agent.domain.checklist import EntityCount, PlanningChecklist
from eva_agent.llm.config import get_client
from eva_agent.state import AgentState
from eva_agent.tools.build_domain_map import load_domain_map, make_slice
from eva_agent.tools.entity_ref import EntityRefs, extract_refs

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

_DOMAIN_SELECTOR_SYS = (
    "Ты выбираешь релевантные сущности внутренней системы для планировщика. "
    "Верни СТРОГО JSON без пояснений: {\"entities\":[\"EntityName\"]}. "
    "Используй только имена сущностей из списка. Если сущность не нужна, не включай ее."
)


def domain_selector(state: AgentState) -> dict:
    """Select a compact domain slice and initialize the planning checklist."""

    query = state.user_input_clean or state.user_input_raw
    intent_kind = state.intent.kind if state.intent else ""
    domain_map = load_domain_map()
    entities = cast(dict[str, dict[str, Any]], domain_map.get("entities", {}))
    refs = extract_refs(query)

    frame_selected = _select_from_frame(state.frame, all_entities=list(entities))
    llm_selected = _select_with_llm(query, intent_kind, entities)
    fallback_selected = _select_from_refs(refs, all_entities=list(entities))
    nlu_selected = state.nlu.entities if state.nlu is not None else []
    selected = _unique([*frame_selected, *llm_selected, *fallback_selected, *nlu_selected])
    if not selected:
        selected = list(entities)

    domain_slice = make_slice(selected, domain_map).model_copy(
        update={
            "hint": _hint(selected, intent_kind),
            "scope": "mixed_diagnostic",
        }
    )
    intent_counts = _intent_counts(state.checklist)
    resolution = "clarify" if state.frame is not None and state.frame.needs_clarification else "proceed"
    clarify_reason = state.frame.clarify_reason if state.frame is not None else ""
    checklist = PlanningChecklist(
        intent=intent_kind,
        entities=domain_slice.entities,
        cardinality=[
            EntityCount(
                entity=entity,
                intent_count=intent_counts.get(entity),
                ref_count=_ref_count(entity, refs),
            )
            for entity in domain_slice.entities
        ],
        access=domain_slice.tools,
        needs_chain=state.checklist.needs_chain if state.checklist is not None else False,
        resolution=resolution,
        clarify_reason=clarify_reason,
    )
    return {"domain_slice": domain_slice, "checklist": checklist}


def _select_with_llm(
    query: str,
    intent_kind: str,
    entities: dict[str, dict[str, Any]],
) -> list[str]:
    user = (
        f"ЗАПРОС:\n{query}\n\n"
        f"INTENT:\n{intent_kind or '-'}\n\n"
        f"СУЩНОСТИ:\n{_entity_index(entities)}"
    )
    try:
        response = get_client("domain").invoke(
            _DOMAIN_SELECTOR_SYS,
            user,
            temperature=0.0,
            json_mode=True,
        )
    except Exception:
        return []

    data = _safe_json(response.text)
    names = data.get("entities")
    if not isinstance(names, list):
        return []
    return _unique(str(name) for name in names if str(name) in entities)


def _safe_json(text: str) -> dict[str, Any]:
    clean = _strip_fences(text)
    try:
        parsed = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        parsed = _load_json_object(clean)
    return parsed if isinstance(parsed, dict) else {}


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _load_json_object(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return {}


def _entity_index(entities: dict[str, dict[str, Any]]) -> str:
    lines = []
    for name, entity in entities.items():
        description = str(entity.get("description", "")).strip()
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


def _select_from_refs(refs: EntityRefs, *, all_entities: list[str]) -> list[str]:
    selected: list[str] = []
    if refs.contract_ids or refs.contract_numbers:
        selected.append("Contract")
    if refs.creative_ids:
        selected.append("Creative")
    if refs.counterparty_ids or refs.counterparty_hints:
        selected.append("Counterparty")
    if refs.placement_ids:
        selected.append("Placement")
    if refs.document_ids:
        selected.append("Document")
    return [entity for entity in selected if entity in all_entities]


def _select_from_frame(frame: Any | None, *, all_entities: list[str]) -> list[str]:
    if frame is None:
        return []
    selected: list[str] = []
    _append_if_known(selected, frame.target, all_entities)
    if frame.relation == "parties":
        for entity in ("Contract", "ContractParty", "Counterparty"):
            _append_if_known(selected, entity, all_entities)
    elif frame.relation == "placements":
        for entity in ("Contract", "Placement", "Creative"):
            _append_if_known(selected, entity, all_entities)
    elif frame.relation == "documents":
        for entity in ("Contract", "Document"):
            _append_if_known(selected, entity, all_entities)
    elif frame.relation == "creative":
        for entity in ("Creative", "Contract"):
            _append_if_known(selected, entity, all_entities)
    for subtask in frame.subtasks:
        for entity in _select_from_frame(subtask, all_entities=all_entities):
            _append_if_known(selected, entity, all_entities)
    return selected


def _append_if_known(out: list[str], entity: str, all_entities: list[str]) -> None:
    if entity in all_entities and entity not in out:
        out.append(entity)


def _intent_counts(checklist: PlanningChecklist | None) -> dict[str, int]:
    if checklist is None:
        return {}
    return {
        count.entity: count.intent_count
        for count in checklist.cardinality
        if count.intent_count is not None
    }


def _ref_count(entity: str, refs: EntityRefs) -> int:
    counts = {
        "Contract": len(refs.contract_ids) + len(refs.contract_numbers),
        "Creative": len(refs.creative_ids),
        "Counterparty": len(refs.counterparty_ids) + len(refs.counterparty_hints),
        "Placement": len(refs.placement_ids),
        "Document": len(refs.document_ids),
    }
    return counts.get(entity, 0)


def _hint(selected: list[str], intent_kind: str) -> str:
    focus = ", ".join(selected) if selected else "all"
    return f"intent={intent_kind or '-'}; entities={focus}"


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


__all__ = ["domain_selector"]
