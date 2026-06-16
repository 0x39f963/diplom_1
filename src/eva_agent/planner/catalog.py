"""Каталог стандартных todo планировщика."""

from __future__ import annotations

from pydantic import BaseModel, Field

from eva_agent.domain.plan import PlanTool


class TodoSpec(BaseModel):
    """Описание одного действия каталога."""

    id: str
    title: str
    when: str
    inputs_required: list[str] = Field(default_factory=list)
    inputs_optional: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    tools: list[PlanTool] = Field(default_factory=list)
    output: str = ""


CATALOG: dict[str, TodoSpec] = {
    "parse_goal": TodoSpec(
        id="parse_goal",
        title="Разобрать цель запроса",
        when="всегда первым шагом",
        output="цель запроса и нужные сущности",
    ),
    "clarify": TodoSpec(
        id="clarify",
        title="Запросить недостающий вход",
        when="цель или нужная сущность не определены",
        blockers=["неизвестна целевая сущность"],
        output="конкретный уточняющий вопрос",
    ),
    "list_contracts": TodoSpec(
        id="list_contracts",
        title="Список договоров",
        when="нужен обзор договоров без конкретного идентификатора",
        tools=["eva_list_unsigned_contracts"],
        output="список договоров со статусами",
    ),
    "search_contracts": TodoSpec(
        id="search_contracts",
        title="Поиск договоров",
        when="нужно найти договор без ID по тексту, дате или статусу",
        inputs_required=["query"],
        inputs_optional=["date_hint", "status_hint"],
        tools=["eva_search_contracts"],
        output="подходящие договоры",
    ),
    "get_contract": TodoSpec(
        id="get_contract",
        title="Карточка договора",
        when="нужны номер, дата, тип или статус конкретного договора",
        inputs_required=["contract_id"],
        blockers=["нет contract_id"],
        tools=["eva_get_contract"],
        output="карточка договора",
    ),
    "list_unsigned_contracts": TodoSpec(
        id="list_unsigned_contracts",
        title="Незарегистрированные договоры",
        when="нужны договоры без регистрации во внутренней системе",
        tools=["eva_list_unsigned_contracts"],
        output="договоры без статуса registered",
    ),
    "get_contract_parties": TodoSpec(
        id="get_contract_parties",
        title="Стороны договора",
        when="нужно понять, кто участвует в договоре",
        inputs_required=["contract_id"],
        blockers=["нет contract_id"],
        tools=["eva_get_contract_parties"],
        output="стороны договора с ролями и counterparty_id",
    ),
    "resolve_party_role": TodoSpec(
        id="resolve_party_role",
        title="Определить сторону по роли",
        when="спросили заказчика или исполнителя по договору",
        inputs_required=["contract_id"],
        inputs_optional=["role"],
        blockers=["нет contract_id"],
        tools=["eva_get_contract_parties"],
        output="counterparty_id стороны с ролью customer или executor",
    ),
    "get_counterparty": TodoSpec(
        id="get_counterparty",
        title="Карточка контрагента",
        when="нужны реквизиты или статус контрагента",
        inputs_required=["counterparty_id"],
        blockers=["нет counterparty_id"],
        tools=["eva_get_counterparty"],
        output="карточка контрагента",
    ),
    "list_user_counterparties": TodoSpec(
        id="list_user_counterparties",
        title="Контрагенты пользователя",
        when="нужно собрать контрагентов из договоров пользователя",
        tools=["eva_get_contract_parties"],
        output="контрагенты, собранные из сторон договоров",
    ),
    "check_counterparty_status": TodoSpec(
        id="check_counterparty_status",
        title="Статус контрагента",
        when="нужно проверить регистрацию контрагента",
        inputs_required=["counterparty_id"],
        blockers=["нет counterparty_id"],
        tools=["eva_get_counterparty"],
        output="статус регистрации контрагента",
    ),
    "get_creative_status": TodoSpec(
        id="get_creative_status",
        title="Статус креатива",
        when="нужно понять готовность креатива",
        inputs_required=["creative_id"],
        blockers=["нет creative_id"],
        tools=["eva_get_creative_status"],
        output="статус и причины блокировки креатива",
    ),
    "get_creative_blockers": TodoSpec(
        id="get_creative_blockers",
        title="Причины блокировки креатива",
        when="нужно узнать, что мешает выпустить креатив",
        inputs_required=["creative_id"],
        blockers=["нет creative_id"],
        tools=["eva_get_creative_status"],
        output="список причин блокировки",
    ),
    "get_creative_form": TodoSpec(
        id="get_creative_form",
        title="Форма распространения креатива",
        when="нужен формат или форма креатива",
        inputs_required=["creative_id"],
        blockers=["нет creative_id"],
        tools=["eva_get_creative_status"],
        output="форма распространения креатива",
    ),
    "compare_creatives": TodoSpec(
        id="compare_creatives",
        title="Сравнить креативы",
        when="нужно сравнить два или более креатива",
        inputs_required=["creative_id"],
        inputs_optional=["creative_id_2"],
        blockers=["нужно минимум два creative_id"],
        tools=["eva_get_creative_status"],
        output="сравнение статусов и форм креативов",
    ),
    "creative_to_contract": TodoSpec(
        id="creative_to_contract",
        title="Договор креатива",
        when="нужно понять, к какому договору привязан креатив",
        inputs_required=["creative_id"],
        blockers=["нет creative_id"],
        tools=["eva_get_creative_status"],
        output="contract_id из карточки креатива",
    ),
    "list_placements": TodoSpec(
        id="list_placements",
        title="Размещения по договору",
        when="нужно узнать, где и когда размещается креатив",
        inputs_required=["contract_id"],
        blockers=["нет contract_id"],
        tools=["eva_list_placements"],
        output="список размещений",
    ),
    "list_documents": TodoSpec(
        id="list_documents",
        title="Документы договора",
        when="нужно перечислить документы по договору",
        inputs_required=["contract_id"],
        blockers=["нет contract_id"],
        tools=["eva_list_contract_documents"],
        output="список документов со статусами",
    ),
    "check_missing_documents": TodoSpec(
        id="check_missing_documents",
        title="Недостающие документы",
        when="нужно понять, каких документов не хватает",
        inputs_required=["contract_id"],
        blockers=["нет contract_id"],
        tools=["eva_list_contract_documents"],
        output="список отсутствующих типов документов",
    ),
    "read_document": TodoSpec(
        id="read_document",
        title="Прочитать документ",
        when="нужно содержимое конкретного документа",
        inputs_required=["doc_id"],
        blockers=["нет doc_id"],
        tools=["eva_doc_read"],
        output="содержимое документа",
    ),
    "download_document": TodoSpec(
        id="download_document",
        title="Скачать документ",
        when="нужна ссылка или файл конкретного документа",
        inputs_required=["doc_id"],
        blockers=["нет doc_id"],
        tools=["eva_doc_download"],
        output="ссылка на файл документа",
    ),
    "attach_document": TodoSpec(
        id="attach_document",
        title="Приложить документ",
        when="нужно приложить документ к договору",
        inputs_required=["contract_id"],
        blockers=["операция записи закрыта гейтом"],
        tools=["eva_doc_attach"],
        output="результат прикрепления документа",
    ),
    "legal_lookup": TodoSpec(
        id="legal_lookup",
        title="Найти норму закона",
        when="нужна норма о рекламе с цитатой",
        inputs_required=["query"],
        tools=["retrieve_legal"],
        output="фрагменты норм с цитатами",
    ),
    "legal_check_against_data": TodoSpec(
        id="legal_check_against_data",
        title="Сопоставить норму с данными",
        when="нужно проверить состояние данных против требования нормы",
        inputs_required=["query"],
        inputs_optional=["contract_id", "creative_id", "counterparty_id"],
        tools=["retrieve_legal"],
        output="вывод о готовности или допустимости",
    ),
    "build_overview": TodoSpec(
        id="build_overview",
        title="Собрать общий обзор",
        when="нужен общий обзор готовности без конкретной сущности",
        tools=["eva_list_unsigned_contracts"],
        output="сводка по незарегистрированным договорам",
    ),
    "assess_readiness": TodoSpec(
        id="assess_readiness",
        title="Оценить готовность",
        when="нужно понять готовность к размещению",
        inputs_optional=["contract_id", "creative_id"],
        tools=["eva_list_unsigned_contracts", "eva_get_creative_status"],
        output="вывод о готовности и список блокеров",
    ),
    "summarize_answer": TodoSpec(
        id="summarize_answer",
        title="Сформулировать ответ",
        when="всегда последним шагом",
        output="итоговый ответ пользователю",
    ),
}

_AVAILABLE_TOOLS_DEFAULT_ITEMS: tuple[PlanTool, ...] = (
    "eva_get_contract",
    "eva_get_contract_parties",
    "eva_get_counterparty",
    "eva_get_creative_status",
    "eva_list_placements",
    "eva_list_contract_documents",
    "eva_list_unsigned_contracts",
    "eva_search_contracts",
    "retrieve_legal",
    "eva_doc_read",
    "eva_doc_download",
)
AVAILABLE_TOOLS_DEFAULT: frozenset[str] = frozenset(_AVAILABLE_TOOLS_DEFAULT_ITEMS)


def todo_tools(todo_id: str) -> list[PlanTool]:
    spec = CATALOG.get(todo_id)
    return list(spec.tools) if spec is not None else []


def todo_is_available(
    todo_id: str,
    available_tools: frozenset[str] = AVAILABLE_TOOLS_DEFAULT,
) -> bool:
    spec = CATALOG.get(todo_id)
    if spec is None:
        return False
    if not spec.tools:
        return True
    return any(tool in available_tools for tool in spec.tools)


def available_todo_ids(
    available_tools: frozenset[str] = AVAILABLE_TOOLS_DEFAULT,
) -> list[str]:
    return [todo_id for todo_id in CATALOG if todo_is_available(todo_id, available_tools)]


def render_catalog(available_tools: frozenset[str] = AVAILABLE_TOOLS_DEFAULT) -> str:
    """Сформировать компактный текст доступного каталога для промпта."""

    lines = ["КАТАЛОГ TODO:"]
    for todo_id in available_todo_ids(available_tools):
        spec = CATALOG[todo_id]
        required = ", ".join(spec.inputs_required) if spec.inputs_required else "-"
        tools = ", ".join(spec.tools) if spec.tools else "internal"
        lines.append(f"- {spec.id}: {spec.title}; inputs={required}; tools={tools}")
    return "\n".join(lines)
