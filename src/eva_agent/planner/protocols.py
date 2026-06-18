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


class ProtocolCard(BaseModel):
    """Machine-readable protocol rule for deterministic compilation."""

    id: str
    operation: str
    target: str
    relation: str | None = None
    protocol_id: ProtocolId
    required_slots: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    emits: list[str] = Field(default_factory=list)
    todo_template: list[str] = Field(default_factory=list)
    priority: int = 0


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
        mandatory=["parse_goal", "summarize_answer"],
        optional=["get_contract_parties", "resolve_party_role", "get_counterparty"],
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
        mandatory=["parse_goal", "summarize_answer"],
        optional=["check_missing_documents", "read_document", "download_document", "attach_document"],
        done_when="получен список документов и отмечены отсутствующие позиции",
    ),
    "overview": ProtocolSpec(
        id="overview",
        when="нужен общий обзор без конкретной сущности",
        mandatory=["parse_goal", "summarize_answer"],
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

PROTOCOL_CARDS: tuple[ProtocolCard, ...] = (
    ProtocolCard(
        id="party_lookup_role",
        operation="read",
        target="ContractParty",
        relation="parties",
        protocol_id="party_lookup",
        required_slots=["contract_id"],
        preconditions=["contract_id is known"],
        emits=["parties", "counterparty"],
        todo_template=[
            "parse_goal",
            "get_contract_parties",
            "resolve_party_role",
            "get_counterparty",
            "summarize_answer",
        ],
        priority=45,
    ),
    ProtocolCard(
        id="party_lookup_all",
        operation="list",
        target="ContractParty",
        relation="parties",
        protocol_id="party_lookup",
        required_slots=["contract_id"],
        preconditions=["contract_id is known"],
        emits=["parties", "counterparties"],
        todo_template=["parse_goal", "get_contract_parties", "get_counterparty", "summarize_answer"],
        priority=40,
    ),
    ProtocolCard(
        id="counterparties_by_contract",
        operation="list",
        target="Counterparty",
        relation="parties",
        protocol_id="party_lookup",
        required_slots=["contract_id"],
        preconditions=["contract_id is known"],
        emits=["parties", "counterparties"],
        todo_template=["parse_goal", "get_contract_parties", "get_counterparty", "summarize_answer"],
        priority=42,
    ),
    ProtocolCard(
        id="contract_parties",
        operation="diagnose",
        target="Contract",
        relation="parties",
        protocol_id="party_lookup",
        required_slots=["contract_id"],
        preconditions=["contract_id is known"],
        emits=["parties"],
        todo_template=["parse_goal", "get_contract_parties", "summarize_answer"],
        priority=36,
    ),
    ProtocolCard(
        id="contract_search_read",
        operation="read",
        target="Contract",
        protocol_id="overview",
        emits=["contracts"],
        todo_template=["parse_goal", "search_contracts", "summarize_answer"],
        priority=18,
    ),
    ProtocolCard(
        id="contract_search_filtered",
        operation="list",
        target="Contract",
        protocol_id="overview",
        emits=["contracts"],
        todo_template=["parse_goal", "search_contracts", "summarize_answer"],
        priority=35,
    ),
    ProtocolCard(
        id="contract_card",
        operation="read",
        target="Contract",
        protocol_id="contract_card",
        required_slots=["contract_id"],
        preconditions=["contract_id is known"],
        emits=["contract"],
        todo_template=["parse_goal", "get_contract", "summarize_answer"],
        priority=20,
    ),
    ProtocolCard(
        id="contract_from_creative",
        operation="read",
        target="Contract",
        relation="creative",
        protocol_id="contract_card",
        required_slots=["creative_id"],
        preconditions=["creative_id is known"],
        emits=["creative", "contract"],
        todo_template=["parse_goal", "get_creative_status", "get_contract", "summarize_answer"],
        priority=30,
    ),
    ProtocolCard(
        id="creative_status",
        operation="read",
        target="Creative",
        protocol_id="creative_status",
        required_slots=["creative_id"],
        preconditions=["creative_id is known"],
        emits=["creative"],
        todo_template=["parse_goal", "get_creative_status", "summarize_answer"],
        priority=20,
    ),
    ProtocolCard(
        id="creative_diagnose",
        operation="diagnose",
        target="Creative",
        protocol_id="creative_status",
        required_slots=["creative_id"],
        preconditions=["creative_id is known"],
        emits=["creative"],
        todo_template=["parse_goal", "get_creative_status", "summarize_answer"],
        priority=25,
    ),
    ProtocolCard(
        id="creatives_by_contract",
        operation="list",
        target="Creative",
        relation="placements",
        protocol_id="placement_list",
        required_slots=["contract_id"],
        preconditions=["contract_id is known"],
        emits=["placements", "creatives"],
        todo_template=["parse_goal", "list_placements", "summarize_answer"],
        priority=45,
    ),
    ProtocolCard(
        id="placement_list",
        operation="list",
        target="Placement",
        relation="placements",
        protocol_id="placement_list",
        required_slots=["contract_id"],
        preconditions=["contract_id is known"],
        emits=["placements"],
        todo_template=["parse_goal", "list_placements", "summarize_answer"],
        priority=35,
    ),
    ProtocolCard(
        id="document_list",
        operation="list",
        target="Document",
        relation="documents",
        protocol_id="document_list",
        required_slots=["contract_id"],
        preconditions=["contract_id is known"],
        emits=["documents"],
        todo_template=["parse_goal", "list_documents", "summarize_answer"],
        priority=30,
    ),
    ProtocolCard(
        id="missing_documents",
        operation="diagnose",
        target="Document",
        relation="documents",
        protocol_id="document_list",
        required_slots=["contract_id"],
        preconditions=["contract_id is known"],
        emits=["documents", "missing"],
        todo_template=["parse_goal", "list_documents", "check_missing_documents", "summarize_answer"],
        priority=35,
    ),
    ProtocolCard(
        id="read_document_by_id",
        operation="read",
        target="Document",
        relation="documents",
        protocol_id="document_list",
        required_slots=["doc_id"],
        preconditions=["doc_id is known"],
        emits=["document"],
        todo_template=["parse_goal", "read_document", "summarize_answer"],
        priority=70,
    ),
    ProtocolCard(
        id="open_document_by_id",
        operation="open",
        target="Document",
        relation="documents",
        protocol_id="document_list",
        required_slots=["doc_id"],
        preconditions=["doc_id is known"],
        emits=["document"],
        todo_template=["parse_goal", "read_document", "summarize_answer"],
        priority=70,
    ),
    ProtocolCard(
        id="read_document",
        operation="open",
        target="Document",
        relation="documents",
        protocol_id="document_list",
        required_slots=["contract_id", "doc_id"],
        preconditions=["contract_id and doc_id are known"],
        emits=["document"],
        todo_template=["parse_goal", "list_documents", "read_document", "summarize_answer"],
        priority=25,
    ),
    ProtocolCard(
        id="download_document_by_id",
        operation="download",
        target="Document",
        relation="documents",
        protocol_id="document_list",
        required_slots=["doc_id"],
        preconditions=["doc_id is known"],
        emits=["download_link"],
        todo_template=["parse_goal", "download_document", "summarize_answer"],
        priority=75,
    ),
    ProtocolCard(
        id="download_document",
        operation="download",
        target="Document",
        relation="documents",
        protocol_id="document_list",
        required_slots=["contract_id", "doc_id"],
        preconditions=["contract_id and doc_id are known"],
        emits=["download_link"],
        todo_template=["parse_goal", "list_documents", "download_document", "summarize_answer"],
        priority=25,
    ),
    ProtocolCard(
        id="attach_document",
        operation="attach",
        target="Document",
        relation="documents",
        protocol_id="document_list",
        required_slots=["contract_id"],
        preconditions=["contract_id is known"],
        emits=["document"],
        todo_template=["parse_goal", "attach_document", "summarize_answer"],
        priority=75,
    ),
    ProtocolCard(
        id="counterparty_card",
        operation="read",
        target="Counterparty",
        protocol_id="counterparty_card",
        required_slots=["counterparty_id"],
        preconditions=["counterparty_id is known"],
        emits=["counterparty"],
        todo_template=["parse_goal", "get_counterparty", "summarize_answer"],
        priority=20,
    ),
    ProtocolCard(
        id="unsigned_overview",
        operation="list",
        target="Contract",
        protocol_id="overview",
        preconditions=["status filter is unsigned or no exact entity is selected"],
        emits=["contracts"],
        todo_template=["parse_goal", "build_overview", "summarize_answer"],
        priority=10,
    ),
    ProtocolCard(
        id="readiness_overview",
        operation="diagnose",
        target="Contract",
        protocol_id="overview",
        preconditions=["overview is requested"],
        emits=["contracts", "readiness"],
        todo_template=["parse_goal", "build_overview", "summarize_answer"],
        priority=10,
    ),
)

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
    has_data_signal = has_entity or mixed_intent or _has(
        text,
        _PARTY_HINTS
        + _CREATIVE_STATUS_HINTS
        + _PLACEMENT_HINTS
        + _DOCUMENT_HINTS
        + _CONTRACT_CARD_HINTS
        + _COUNTERPARTY_HINTS
        + _OVERVIEW_HINTS,
    )

    # mixed_diagnostic - это запрос данных, ему нужен data-протокол (overview/party_lookup/...),
    # а не строгий mixed_legal_data. Строгий берем только при реальном пересечении нормы и данных.
    if legal and has_data_signal and has_entity:
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
