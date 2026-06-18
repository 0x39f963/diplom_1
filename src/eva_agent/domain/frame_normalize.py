"""Deterministic semantic normalization for planning frames."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from eva_agent.domain.frame import PlanningFrame
from eva_agent.nlu.preprocess import NluFeatures

_RELATION_TARGETS: dict[str, str] = {
    "parties": "ContractParty",
    "documents": "Document",
    "placements": "Placement",
    "creative": "Creative",
}
_RELATION_SIGNALS = frozenset(_RELATION_TARGETS)
_PARTY_LEMMAS = frozenset(
    {
        "сторона",
        "заказчик",
        "исполнитель",
        "контрагент",
        "участник",
        "участвовать",
        "заключить",
    }
)
_PARTY_TEXT = (
    "сторон",
    "контрагент по договору",
    "кто участвует",
    "с кем заключ",
)
_DOCUMENT_LEMMAS = frozenset({"документ", "акт", "приложение", "файл", "хватать"})
_DOCUMENT_TEXT = ("документ", "акт", "приложен", "вложен", "не хватает", "чего не хватает")
_PLACEMENT_LEMMAS = frozenset({"размещение", "объявление", "площадка", "показ", "публикация"})
_PLACEMENT_TEXT = ("размещени", "объявлени", "площадк")
_TARGET_ID_SLOTS = {
    "Contract": "contract_id",
    "Creative": "creative_id",
    "Counterparty": "counterparty_id",
    "Document": "doc_id",
    "Placement": "placement_id",
}


def normalize_frame(
    frame: PlanningFrame,
    nlu: NluFeatures,
    domain_map: dict[str, Any],
) -> PlanningFrame:
    """Apply deterministic relation and selector rules after LLM parsing."""

    selector = dict(frame.selector)
    _preserve_contract_id(selector, nlu)

    relation = frame.relation
    target = frame.target
    trace = list(frame.trace)

    inferred_relation = _inferred_relation(nlu, selector)
    if inferred_relation and _would_override_explicit_target(frame, inferred_relation, selector):
        inferred_relation = None
    if relation not in _RELATION_SIGNALS and inferred_relation:
        relation = inferred_relation
        trace.append(f"frame normalized: relation={relation} from nlu")

    implied = relation_implied_target(relation)
    if implied and _should_apply_relation_target(frame, relation, selector):
        if target != implied:
            trace.append(f"frame normalized: target {target or '-'}->{implied} by relation")
        target = implied

    updates: dict[str, Any] = {
        "target": target,
        "relation": relation,
        "selector": selector,
        "trace": _unique(trace),
    }
    if relation == "documents" and "missing" in frame.fields and "doc_id" not in selector:
        updates.update({"operation": "diagnose", "cardinality": "all", "output": "list"})
    if target in _entity_names(domain_map) and _only_target_clarification(frame.clarify_reason):
        updates["needs_clarification"] = False
        updates["clarify_reason"] = ""
    return frame.model_copy(update=updates)


def relation_implied_target(relation: str | None) -> str:
    """Return the natural target for a relation, if the relation is known."""

    return _RELATION_TARGETS.get((relation or "").strip().lower(), "")


def relation_implies_target(relation: str | None, target: str) -> bool:
    """Whether target is the deterministic natural target of relation."""

    return bool(target) and relation_implied_target(relation) == target


def _preserve_contract_id(selector: dict[str, str], nlu: NluFeatures) -> None:
    if selector.get("contract_id"):
        return
    for key in ("contracts", "contract_numbers"):
        values = nlu.entity_ids.get(key) or []
        if values:
            selector["contract_id"] = str(values[0])
            return


def _inferred_relation(nlu: NluFeatures, selector: dict[str, str]) -> str | None:
    if "contract_id" not in selector:
        return None
    lowered = nlu.query.lower().replace("\u0451", "е")
    lemmas = set(nlu.lemmas)
    if nlu.roles or _has_any(lemmas, _PARTY_LEMMAS) or _contains_any(lowered, _PARTY_TEXT):
        return "parties"
    if _has_any(lemmas, _DOCUMENT_LEMMAS) or _contains_any(lowered, _DOCUMENT_TEXT):
        return "documents"
    if _has_any(lemmas, _PLACEMENT_LEMMAS) or _contains_any(lowered, _PLACEMENT_TEXT):
        return "placements"
    return None


def _should_apply_relation_target(
    frame: PlanningFrame,
    relation: str | None,
    selector: dict[str, str],
) -> bool:
    if relation != "creative":
        return True
    return not (frame.target == "Contract" and selector.get("creative_id") and not selector.get("contract_id"))


def _would_override_explicit_target(
    frame: PlanningFrame,
    relation: str,
    selector: dict[str, str],
) -> bool:
    if frame.target == "Contract":
        return False
    id_slot = _TARGET_ID_SLOTS.get(frame.target)
    if not id_slot or not selector.get(id_slot):
        return False
    implied = relation_implied_target(relation)
    return bool(implied and implied != frame.target)


def _entity_names(domain_map: dict[str, Any]) -> set[str]:
    raw = domain_map.get("entities", {})
    return set(cast(dict[str, Any], raw)) if isinstance(raw, dict) else set()


def _only_target_clarification(reason: str) -> bool:
    lowered = reason.lower()
    return not lowered or "сущ" in lowered or "target" in lowered


def _has_any(values: set[str], expected: Iterable[str]) -> bool:
    return any(value in values for value in expected)


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


__all__ = ["normalize_frame", "relation_implied_target", "relation_implies_target"]
