"""Lite semantic coverage for compiled todo plans."""

from __future__ import annotations

from dataclasses import dataclass

from eva_agent.domain.frame import PlanningFrame
from eva_agent.domain.plan import TodoPlan
from eva_agent.nlu.preprocess import NluFeatures, has_legal_signal
from eva_agent.planner.catalog import CATALOG

_ATOM_ORDER: tuple[str, ...] = (
    "Contract.resolution",
    "Contract.card",
    "Contract.parties",
    "Contract.documents",
    "Contract.documents.missing",
    "Contract.placements",
    "Placement.creative",
    "Creative.status",
    "ContractParty.counterparty",
    "Counterparty.card",
    "Legal.rules",
)
_ORDER_INDEX = {atom: index for index, atom in enumerate(_ATOM_ORDER)}

_RELATION_ATOMS: dict[str, str] = {
    "parties": "Contract.parties",
    "documents": "Contract.documents",
    "placements": "Contract.placements",
    "creative": "Placement.creative",
}
_TARGET_READ_ATOMS: dict[str, str] = {
    "Contract": "Contract.card",
    "Counterparty": "Counterparty.card",
}
_PARTY_ENTITIES = frozenset({"ContractParty"})
_DOCUMENT_ENTITIES = frozenset({"Document"})
_PLACEMENT_ENTITIES = frozenset({"Placement"})
_CREATIVE_ENTITIES = frozenset({"Creative"})
_CARD_TEXT = ("карточк", "номер договора", "дата договора", "статус договора")
_CREATIVE_STATUS_TEXT = ("статус креатив", "готов", "блокир", "мешает", "выпустить")


@dataclass(frozen=True)
class CoverageResult:
    requested: set[str]
    covered: set[str]
    missing: set[str]


def requested_atoms(frame: PlanningFrame, nlu: NluFeatures | None = None) -> set[str]:
    """Return mandatory semantic atoms explicitly requested by the user."""

    atoms: set[str] = set()
    _add_frame_atoms(atoms, frame)
    for subtask in frame.subtasks:
        atoms.update(requested_atoms(subtask, None))
    if nlu is not None:
        _add_nlu_atoms(atoms, frame, nlu)
    return atoms


def plan_coverage(plan: TodoPlan, requested: set[str]) -> CoverageResult:
    emitted: set[str] = set()
    for item in plan.items:
        spec = CATALOG.get(item.id)
        if spec is not None:
            emitted.update(spec.emits)
    covered = requested & emitted
    return CoverageResult(
        requested=set(requested),
        covered=covered,
        missing=requested - covered,
    )


def todos_for_atom(atom: str) -> list[str]:
    return [todo_id for todo_id, spec in CATALOG.items() if atom in spec.emits]


def coverage_payload(result: CoverageResult) -> dict[str, list[str]]:
    return {
        "requested": sort_atoms(result.requested),
        "covered": sort_atoms(result.covered),
        "missing": sort_atoms(result.missing),
    }


def sort_atoms(atoms: set[str]) -> list[str]:
    return sorted(atoms, key=lambda atom: (_ORDER_INDEX.get(atom, len(_ATOM_ORDER)), atom))


def _add_frame_atoms(atoms: set[str], frame: PlanningFrame) -> None:
    relation_atom = _RELATION_ATOMS.get(_norm(frame.relation))
    if relation_atom and not _direct_entity_read(frame):
        atoms.add(relation_atom)

    if frame.operation in {"read", "open"}:
        target_atom = _TARGET_READ_ATOMS.get(frame.target)
        if target_atom:
            atoms.add(target_atom)
    if frame.operation == "diagnose" and frame.target == "Creative":
        atoms.add("Creative.status")
    if frame.target == "Creative" and "status" in frame.fields:
        atoms.add("Creative.status")
    if "missing" in frame.fields:
        atoms.add("Contract.documents.missing")
    if "legal_signal" in frame.fields or "legal" in frame.fields:
        atoms.add("Legal.rules")


def _add_nlu_atoms(atoms: set[str], frame: PlanningFrame, nlu: NluFeatures) -> None:
    entities = set(nlu.entities)
    lowered = nlu.query.lower().replace("ё", "е")
    selector = frame.selector
    if "Contract" in entities and _contains_any(lowered, _CARD_TEXT):
        atoms.add("Contract.card")
    if (
        nlu.roles
        or entities & _PARTY_ENTITIES
        or _contains_any(lowered, ("сторон", "кто участвует", "контрагент по договору"))
    ):
        atoms.add("Contract.parties")
    if (entities & _DOCUMENT_ENTITIES or _contains_any(lowered, ("документ", "акт", "приложен"))) and (
        selector.get("contract_id") or not selector.get("doc_id")
    ):
        atoms.add("Contract.documents")
    if "missing" in frame.fields or _contains_any(lowered, ("не хватает", "чего не хватает")):
        atoms.add("Contract.documents.missing")
    if entities & _PLACEMENT_ENTITIES or _contains_any(lowered, ("размещени", "площадк")):
        atoms.add("Contract.placements")
    if entities & _CREATIVE_ENTITIES and not selector.get("creative_id"):
        atoms.add("Placement.creative")
        if _contains_any(lowered, _CREATIVE_STATUS_TEXT) or _norm(frame.relation) == "placements":
            atoms.add("Creative.status")
    elif entities & _CREATIVE_ENTITIES and _contains_any(lowered, _CREATIVE_STATUS_TEXT):
        atoms.add("Creative.status")
    if has_legal_signal(nlu):
        atoms.add("Legal.rules")


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _direct_entity_read(frame: PlanningFrame) -> bool:
    selector = frame.selector
    if frame.target == "Document" and selector.get("doc_id") and frame.operation in {"read", "open", "download"}:
        return True
    if frame.target == "Creative" and selector.get("creative_id"):
        return True
    return bool(frame.target == "Counterparty" and selector.get("counterparty_id"))


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


__all__ = [
    "CoverageResult",
    "coverage_payload",
    "plan_coverage",
    "requested_atoms",
    "sort_atoms",
    "todos_for_atom",
]
