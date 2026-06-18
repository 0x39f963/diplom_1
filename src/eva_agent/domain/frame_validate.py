"""Validation helpers for semantic frames."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal, cast

from eva_agent.domain.frame import PlanningFrame
from eva_agent.domain.relations import RELATIONS
from eva_agent.planner.compile import rank_protocol_cards
from eva_agent.planner.protocols import PROTOCOL_CARDS

_OPERATIONS = frozenset({"read", "list", "compare", "open", "download", "attach", "diagnose"})
_CARDINALITIES = frozenset({"one", "all", "n"})
_OUTPUTS = frozenset({"value", "card", "list", "summary"})
_SELECTOR_KEYS = frozenset(
    {
        "contract_id",
        "creative_id",
        "counterparty_id",
        "doc_id",
        "placement_id",
        "role",
        "counterparty_hint",
        "contract_number",
        "query",
        "search_query",
        "name",
        "title",
        "legal_query",
    }
)
_DATE_HINTS = frozenset({"none", "yesterday", "last_week", "last_month"})
_STATUS_ALIASES = {"unregistered": "unsigned", "not_registered": "unsigned"}
_SOFT_CODES = frozenset({"bad_field", "card_op_mismatch"})


@dataclass(frozen=True)
class FrameIssue:
    level: Literal["syntax", "semantics", "compile"]
    code: str
    message: str
    hint: str = ""


@dataclass(frozen=True)
class FrameValidation:
    ok: bool
    issues: tuple[FrameIssue, ...]

    @property
    def signature(self) -> str:
        pairs = sorted({(issue.level, issue.code) for issue in self.issues})
        raw = "|".join(f"{level}:{code}" for level, code in pairs)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def validate_frame(
    frame: PlanningFrame,
    *,
    domain_map: dict[str, Any],
    draft: PlanningFrame | None = None,
) -> FrameValidation:
    issues: list[FrameIssue] = []
    _validate_syntax(frame, issues)
    _validate_semantics(frame, domain_map, draft, issues)
    _validate_compile(frame, domain_map, issues)
    hard = [issue for issue in issues if issue.code not in _SOFT_CODES]
    return FrameValidation(ok=not hard, issues=tuple(issues))


def _validate_syntax(frame: PlanningFrame, issues: list[FrameIssue]) -> None:
    operation = str(getattr(frame, "operation", ""))
    cardinality = str(getattr(frame, "cardinality", ""))
    output = str(getattr(frame, "output", ""))
    if operation not in _OPERATIONS:
        issues.append(
            FrameIssue(
                "syntax",
                "bad_operation",
                "Операция вне списка.",
                "выберите read, list, compare, open, download, attach или diagnose",
            )
        )
    if cardinality not in _CARDINALITIES:
        issues.append(
            FrameIssue(
                "syntax",
                "bad_cardinality",
                "Количество вне списка.",
                "выберите one, all или n",
            )
        )
    if output not in _OUTPUTS:
        issues.append(
            FrameIssue(
                "syntax",
                "bad_output",
                "Формат ответа вне списка.",
                "выберите value, card, list или summary",
            )
        )
    confidence = getattr(frame, "confidence", 0.0)
    if not isinstance(confidence, int | float) or confidence < 0.0 or confidence > 1.0:
        issues.append(
            FrameIssue(
                "syntax",
                "bad_confidence",
                "Уверенность вне диапазона.",
                "передайте число от 0 до 1",
            )
        )


def _validate_semantics(
    frame: PlanningFrame,
    domain_map: dict[str, Any],
    draft: PlanningFrame | None,
    issues: list[FrameIssue],
) -> None:
    entities = _entities(domain_map)
    target = frame.target.strip()
    if not target or target not in entities:
        hint = _target_hint(draft)
        issues.append(
            FrameIssue(
                "semantics",
                "unknown_target",
                "Целевая сущность не найдена.",
                hint,
            )
        )
        return

    if frame.relation and frame.relation not in _relations_for_target(target):
        issues.append(
            FrameIssue(
                "semantics",
                "bad_relation",
                "Связь не подходит к сущности.",
                f"выберите связь для {target}: {', '.join(sorted(_relations_for_target(target))) or '-'}",
            )
        )

    for key in frame.selector:
        if key not in _SELECTOR_KEYS:
            issues.append(
                FrameIssue(
                    "semantics",
                    "bad_selector_key",
                    "Ключ селектора не поддержан.",
                    f"замените {key} на один из известных ключей",
                )
            )

    _validate_statuses(frame, entities[target], issues)
    if frame.filters.date_hint not in _DATE_HINTS:
        issues.append(
            FrameIssue(
                "semantics",
                "bad_date",
                "Дата указана в неподдержанном виде.",
                "укажите вчера, прошлую неделю или прошлый месяц",
            )
        )
    _validate_fields(frame, entities[target], issues)
    _validate_cardinality(frame, issues)
    _validate_operation(frame, entities[target], issues)


def _validate_statuses(
    frame: PlanningFrame,
    entity: dict[str, Any],
    issues: list[FrameIssue],
) -> None:
    allowed = {str(status).lower() for status in _str_list(entity.get("statuses"))}
    if not allowed:
        return
    for status in frame.filters.status:
        normalized = status.strip().lower()
        if normalized in allowed:
            continue
        alias = _STATUS_ALIASES.get(normalized, normalized)
        if alias in allowed:
            continue
        issues.append(
            FrameIssue(
                "semantics",
                "bad_status",
                "Статус не подходит к сущности.",
                f"выберите статус из списка: {', '.join(sorted(allowed))}",
            )
        )


def _validate_fields(
    frame: PlanningFrame,
    entity: dict[str, Any],
    issues: list[FrameIssue],
) -> None:
    fields = set(_str_list(entity.get("fields")))
    for field in frame.fields:
        if field in fields:
            continue
        issues.append(
            FrameIssue(
                "semantics",
                "bad_field",
                "Поле не найдено у сущности.",
                f"проверьте поле {field}",
            )
        )


def _validate_cardinality(frame: PlanningFrame, issues: list[FrameIssue]) -> None:
    if frame.operation == "list" and frame.cardinality == "one" and not frame.selector:
        issues.append(
            FrameIssue(
                "semantics",
                "card_op_mismatch",
                "Операция похожа на список.",
                "поставьте cardinality=all или добавьте селектор",
            )
        )
    if frame.operation == "read" and frame.cardinality in {"all", "n"}:
        issues.append(
            FrameIssue(
                "semantics",
                "card_op_mismatch",
                "Количество похоже на список.",
                "поставьте operation=list",
            )
        )


def _validate_operation(
    frame: PlanningFrame,
    entity: dict[str, Any],
    issues: list[FrameIssue],
) -> None:
    operations = set(_str_list(entity.get("operations")))
    if operations and frame.operation not in operations:
        issues.append(
            FrameIssue(
                "semantics",
                "card_op_mismatch",
                "Операция не указана у сущности.",
                f"выберите операцию из списка: {', '.join(sorted(operations))}",
            )
        )


def _validate_compile(
    frame: PlanningFrame,
    domain_map: dict[str, Any],
    issues: list[FrameIssue],
) -> None:
    if frame.target not in _entities(domain_map):
        return
    target_cards = [card for card in PROTOCOL_CARDS if card.target == frame.target]
    if not target_cards:
        issues.append(
            FrameIssue(
                "compile",
                "no_protocol",
                "Для сущности нет протокола.",
                "уточните другую сущность или действие",
            )
        )
        return
    ranked = rank_protocol_cards(frame, domain_slice=None)
    if not ranked or ranked[0].score <= 0:
        issues.append(
            FrameIssue(
                "compile",
                "empty_after_compile",
                "Запрос не попал в протокол.",
                "уточните действие или нужную сущность",
            )
        )


def _entities(domain_map: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = domain_map.get("entities", {})
    if not isinstance(raw, dict):
        return {}
    return cast(dict[str, dict[str, Any]], raw)


def _relations_for_target(target: str) -> set[str]:
    allowed = {card.relation for card in PROTOCOL_CARDS if card.target == target and card.relation}
    for relation in RELATIONS:
        if relation.source_entity == target or relation.target_entity == target:
            allowed.add(_relation_alias(relation.source_entity, relation.target_entity))
    return {value for value in allowed if value}


def _relation_alias(source: str, target: str) -> str:
    if target == "Document":
        return "documents"
    if target == "Placement":
        return "placements"
    if source == "ContractParty" or target == "Counterparty":
        return "parties"
    if source == "Creative" and target == "Contract":
        return "creative"
    return f"{source.lower()}_{target.lower()}"


def _target_hint(draft: PlanningFrame | None) -> str:
    if draft is not None and draft.target:
        return f"проверьте сущность, похоже нужен {draft.target}"
    return "укажите сущность: договор, контрагент, креатив, документ или размещение"


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


__all__ = ["FrameIssue", "FrameValidation", "validate_frame"]
