"""Реестр read-only инструментов и простой селектор для data_gather."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from eva_agent.mock.data import (
    OWNER_SCOPE,
    eva_get_contract,
    eva_get_contract_parties,
    eva_get_counterparty,
    eva_get_creative_status,
    eva_kktu_suggest,
    eva_list_contract_documents,
    eva_list_placements,
    eva_list_unsigned_contracts,
    eva_search_contracts,
)
from eva_agent.state import ApiFinding, Chunk
from eva_agent.tools.doc_adapters import eva_doc_attach, eva_doc_download, eva_doc_read
from eva_agent.tools.entity_ref import EntityRefs
from eva_agent.tools.retrieve import retrieve_legal as _retrieve_legal

ToolFn = Callable[..., ApiFinding]

TOOL_REGISTRY: dict[str, ToolFn] = {
    "eva_get_contract": eva_get_contract,
    "eva_get_contract_parties": eva_get_contract_parties,
    "eva_get_counterparty": eva_get_counterparty,
    "eva_get_creative_status": eva_get_creative_status,
    "eva_list_contract_documents": eva_list_contract_documents,
    "eva_list_placements": eva_list_placements,
    "eva_list_unsigned_contracts": eva_list_unsigned_contracts,
    "eva_search_contracts": eva_search_contracts,
    "eva_kktu_suggest": eva_kktu_suggest,
}


def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "text": chunk.text,
        "citation": chunk.citation,
        "law_number": chunk.law_number,
        "article": chunk.article,
        "trust_level": chunk.trust_level,
        "score": chunk.score,
        "source_url": chunk.source_url,
    }


def eva_retrieve_legal(query: str) -> ApiFinding:
    """RAG-обертка в формате ApiFinding для общего реестра исполнения."""
    try:
        result = _retrieve_legal(query)
    except Exception as exc:
        return ApiFinding(
            tool="retrieve_legal",
            args={"query": query},
            data={"chunks": [], "citations": [], "error": exc.__class__.__name__},
            owner_ref=OWNER_SCOPE,
        )
    chunks = [_chunk_payload(chunk) for chunk in result.chunks]
    citations = [chunk.citation for chunk in result.chunks if chunk.citation]
    return ApiFinding(
        tool="retrieve_legal",
        args={"query": query},
        data={"chunks": chunks, "citations": citations},
        owner_ref=OWNER_SCOPE,
    )


EXECUTION_REGISTRY: dict[str, ToolFn] = {
    **TOOL_REGISTRY,
    "retrieve_legal": eva_retrieve_legal,
    "eva_doc_read": eva_doc_read,
    "eva_doc_download": eva_doc_download,
    "eva_doc_attach": eva_doc_attach,
}

_DOC_HINTS = ("документ", "приложен", "не хватает", "оформлен", "акт")
_PLACEMENT_HINTS = ("размещени", "креатив", "объявлен", "крутит", "площадк")
_PARTY_HINTS = (
    "сторон",
    "заказчик",
    "исполнитель",
    "контрагент",
    "реквизит",
    "подтвержд",
    "кто участвует",
    "с кем",
)
_CONTRACT_HINTS = ("номер договора", "статус договора", "карточк", "состояни", "договор")
_SEARCH_HINTS = ("последн", "вчера", "неподпис", "незарегистр")


def _has(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def _add(out: list[str], tool: str) -> None:
    if tool not in out:
        out.append(tool)


def select_tools(intent: str | None, refs: EntityRefs) -> list[str]:
    """Выбрать имена инструментов по текстовым сигналам intent и найденным сущностям."""
    text = (intent or "").lower()
    tools: list[str] = []
    has_contract = refs.primary_contract is not None

    if refs.document_ids:
        if any(hint in text for hint in ("скач", "выгруз")):
            _add(tools, "eva_doc_download")
        elif any(hint in text for hint in ("проч", "открой", "что внутри", "содерж")):
            _add(tools, "eva_doc_read")

    if refs.primary_counterparty is not None:
        _add(tools, "eva_get_counterparty")

    if has_contract and _has(text, _DOC_HINTS):
        _add(tools, "eva_list_contract_documents")
        if "не хватает" in text or "чего не хватает" in text:
            _add(tools, "eva_list_unsigned_contracts")
    if has_contract and _has(text, _PLACEMENT_HINTS):
        _add(tools, "eva_list_placements")
    if has_contract and _has(text, _PARTY_HINTS):
        _add(tools, "eva_get_contract_parties")
    if has_contract and (_has(text, _CONTRACT_HINTS) or not tools):
        _add(tools, "eva_get_contract")

    if refs.primary_creative is not None:
        _add(tools, "eva_get_creative_status")

    if not refs.has_any and ("с какими контрагентами" in text or "собери из договоров" in text):
        _add(tools, "eva_get_contract_parties")

    if not refs.has_any and _has(text, _SEARCH_HINTS):
        _add(tools, "eva_search_contracts")

    if not tools and "mixed_diagnostic" in text:
        _add(tools, "eva_list_unsigned_contracts")

    return tools


def _args_for_tool(tool: str, refs: EntityRefs, query: str) -> dict[str, Any] | None:
    if tool in {
        "eva_get_contract",
        "eva_get_contract_parties",
        "eva_list_contract_documents",
        "eva_list_placements",
    }:
        if refs.primary_contract is None:
            if tool == "eva_get_contract_parties" and (
                "с какими контрагентами" in query.lower() or "собери из договоров" in query.lower()
            ):
                return {"contract_id": "CT-1"}
            return None
        return {"contract_id": refs.primary_contract}
    if tool == "eva_get_counterparty":
        if refs.primary_counterparty is None:
            return None
        return {"counterparty_id": refs.primary_counterparty}
    if tool == "eva_get_creative_status":
        if refs.primary_creative is None:
            return None
        return {"creative_id": refs.primary_creative}
    if tool in {"eva_doc_read", "eva_doc_download"}:
        if not refs.document_ids:
            return None
        return {"doc_id": refs.document_ids[0]}
    if tool == "eva_search_contracts":
        return {"q": query}
    if tool == "eva_kktu_suggest":
        return {"description": query}
    if tool == "retrieve_legal":
        return {"query": query}
    return {}


def run_selected_tools(tool_names: list[str], refs: EntityRefs, query: str) -> list[ApiFinding]:
    """Исполнить выбранные инструменты через общий реестр."""
    findings: list[ApiFinding] = []
    seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
    for tool in tool_names:
        fn = EXECUTION_REGISTRY.get(tool)
        if fn is None:
            continue
        args = _args_for_tool(tool, refs, query)
        if args is None:
            continue
        key = (tool, tuple(sorted(args.items())))
        if key in seen:
            continue
        seen.add(key)
        findings.append(fn(**args))
    return findings
