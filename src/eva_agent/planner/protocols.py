"""Каталог протоколов планировщика."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProtocolId = Literal[
    "legal_only",
    "party_lookup",
    "contract_card",
    "creative_status",
    "counterparty_card",
    "placement_list",
    "document_list",
    "overview",
    "mixed_legal_data",
    "clarify_first",
]


class ProtocolSpec(BaseModel):
    """Стратегия под тип цели: обязательные todo, опциональные todo и критерий закрытия."""

    id: ProtocolId
    when: str
    mandatory: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)
    done_when: str = ""


PROTOCOLS: dict[ProtocolId, ProtocolSpec] = {
    "legal_only": ProtocolSpec(
        id="legal_only",
        when="вопрос по норме закона без обращения к данным системы",
        mandatory=["parse_goal", "legal_lookup", "summarize_answer"],
        done_when="найдена релевантная норма и дан ответ с опорой на нее",
    ),
    "party_lookup": ProtocolSpec(
        id="party_lookup",
        when="нужно найти заказчика, исполнителя или стороны договора",
        mandatory=["parse_goal", "resolve_party_role", "summarize_answer"],
        optional=["get_contract_parties", "get_counterparty"],
        done_when="роль стороны определена или собран блокер для уточнения",
    ),
    "contract_card": ProtocolSpec(
        id="contract_card",
        when="нужна карточка, номер, дата или статус договора",
        mandatory=["parse_goal", "get_contract", "summarize_answer"],
        optional=["search_contracts", "get_contract_parties"],
        done_when="получена карточка договора или зафиксирован недостающий вход",
    ),
    "creative_status": ProtocolSpec(
        id="creative_status",
        when="нужен статус креатива или причины, мешающие выпуску",
        mandatory=["parse_goal", "get_creative_status", "summarize_answer"],
        optional=["get_creative_blockers", "creative_to_contract", "list_documents"],
        done_when="получен статус креатива и причины блокировки",
    ),
    "counterparty_card": ProtocolSpec(
        id="counterparty_card",
        when="нужна карточка, реквизиты или статус контрагента",
        mandatory=["parse_goal", "get_counterparty", "summarize_answer"],
        optional=["check_counterparty_status"],
        done_when="получены данные контрагента или собран блокер для уточнения",
    ),
    "placement_list": ProtocolSpec(
        id="placement_list",
        when="нужны размещения по договору",
        mandatory=["parse_goal", "list_placements", "summarize_answer"],
        optional=["get_creative_status"],
        done_when="получен список размещений по договору",
    ),
    "document_list": ProtocolSpec(
        id="document_list",
        when="нужны документы договора или список недостающих документов",
        mandatory=["parse_goal", "list_documents", "summarize_answer"],
        optional=["check_missing_documents", "read_document", "download_document", "attach_document"],
        done_when="получен список документов и отмечены отсутствующие позиции",
    ),
    "overview": ProtocolSpec(
        id="overview",
        when="нужен общий обзор без конкретной сущности",
        mandatory=["parse_goal", "build_overview", "summarize_answer"],
        optional=["list_contracts", "list_unsigned_contracts", "search_contracts", "assess_readiness"],
        done_when="собрана обзорная сводка по доступным данным",
    ),
    "mixed_legal_data": ProtocolSpec(
        id="mixed_legal_data",
        when="нужно сопоставить норму закона с данными системы",
        mandatory=["parse_goal", "legal_lookup", "legal_check_against_data", "summarize_answer"],
        optional=[
            "get_contract",
            "get_contract_parties",
            "get_creative_status",
            "get_counterparty",
            "list_placements",
            "list_documents",
            "search_contracts",
        ],
        done_when="норма сопоставлена с одним фокусным data-todo и дан вывод",
    ),
    "clarify_first": ProtocolSpec(
        id="clarify_first",
        when="цель или целевая сущность не определены",
        mandatory=["parse_goal", "clarify"],
        done_when="пользователю задан конкретный уточняющий вопрос",
    ),
}

_LEGAL_HINTS = (
    "закон",
    "норм",
    "статья",
    "ст.",
    "38-фз",
    "обязан",
    "разреш",
    "запрещ",
    "требован",
)
_PARTY_HINTS = ("сторон", "заказчик", "исполнитель", "кто участвует", "с кем")
_CREATIVE_STATUS_HINTS = ("статус", "мешает", "выпустить", "блокир", "готов")
_PLACEMENT_HINTS = ("размещени", "площадк", "крутит", "период")
_DOCUMENT_HINTS = ("документ", "акт", "приложен", "не хватает", "файл", "скачать", "прочитать")
_CONTRACT_CARD_HINTS = ("карточк", "номер", "дата", "статус договора", "договор")
_COUNTERPARTY_HINTS = ("контрагент", "инн", "реквизит")
_OVERVIEW_HINTS = (
    "обзор",
    "сводк",
    "готовност",
    "неподпис",
    "незарегистр",
    "последн",
    "вчера",
    "оформл",
    "подписан",
    "не зарегистр",
)


def _has(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def select_protocol(
    query: str,
    has_entity: bool,
    intent_kind: str | None = None,
) -> ProtocolId:
    """Выбрать первичный протокол по текстовым сигналам и наличию сущности."""

    text = query.lower().strip()
    if not text:
        return "clarify_first"

    legal = _has(text, _LEGAL_HINTS) or intent_kind == "legal_consult"
    mixed_intent = intent_kind == "mixed_diagnostic"
    has_data_signal = has_entity or _has(
        text,
        _PARTY_HINTS
        + _CREATIVE_STATUS_HINTS
        + _PLACEMENT_HINTS
        + _DOCUMENT_HINTS
        + _CONTRACT_CARD_HINTS
        + _COUNTERPARTY_HINTS
        + _OVERVIEW_HINTS,
    )

    if mixed_intent or (legal and has_data_signal and has_entity):
        return "mixed_legal_data"
    if legal:
        return "legal_only"
    if _has(text, _OVERVIEW_HINTS):
        return "overview"
    if intent_kind == "need_clarification":
        return "clarify_first"
    if _has(text, _PARTY_HINTS):
        return "party_lookup"
    if "креатив" in text and _has(text, _CREATIVE_STATUS_HINTS):
        return "creative_status"
    if _has(text, _PLACEMENT_HINTS):
        return "placement_list"
    if _has(text, _DOCUMENT_HINTS):
        return "document_list"
    if _has(text, _COUNTERPARTY_HINTS):
        return "counterparty_card"
    if _has(text, _CONTRACT_CARD_HINTS):
        return "contract_card"
    if has_data_signal:
        return "overview"
    return "clarify_first"


def render_protocol(protocol_id: ProtocolId) -> str:
    """Сформировать текст протокола для промпта планировщика."""

    spec = PROTOCOLS[protocol_id]
    lines = [
        f"ПРОТОКОЛ: {spec.id}",
        f"КОГДА: {spec.when}",
        "ОБЯЗАТЕЛЬНЫЕ TODO: " + ", ".join(spec.mandatory),
    ]
    if spec.optional:
        lines.append("ОПЦИОНАЛЬНЫЕ TODO: " + ", ".join(spec.optional))
    lines.append(f"ГОТОВО КОГДА: {spec.done_when}")
    return "\n".join(lines)
