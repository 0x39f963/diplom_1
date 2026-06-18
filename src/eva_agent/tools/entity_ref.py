"""Извлечение ссылок на сущности из текста запроса."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from eva_agent.nlu.gazetteer import DOMAIN_SIGNAL_LEMMAS
from eva_agent.nlu.ru import lemmatize_text
from eva_agent.planner.protocols import (
    _CONTRACT_CARD_HINTS,
    _COUNTERPARTY_HINTS,
    _CREATIVE_STATUS_HINTS,
    _DOCUMENT_HINTS,
    _OVERVIEW_HINTS,
    _PARTY_HINTS,
    _PLACEMENT_HINTS,
)

_CONTRACT_SYN = re.compile(r"\bCT-(\d+)\b", re.IGNORECASE)
_CREATIVE_SYN = re.compile(r"\bCR-(\d+)\b", re.IGNORECASE)
_COUNTERPARTY_SYN = re.compile(r"\bCP-(\d+)\b", re.IGNORECASE)
_DOCUMENT_SYN = re.compile(r"\bDOC-(\d+)\b", re.IGNORECASE)
_PLACEMENT_SYN = re.compile(r"\bPL-(\d+)\b", re.IGNORECASE)
_CONTRACT_NUM = re.compile(r"\b[ДD]-(\d{4})/(\d+)\b", re.IGNORECASE)

_ORG_FORM = r"(?:ООО|АО|ПАО|ЗАО|ОАО|ИП)"
_COUNTERPARTY_ORG_QUOTED = re.compile(
    rf"\b(?P<form>{_ORG_FORM})\s*[«\"'](?P<name>[^»\"']+)[»\"']",
    re.IGNORECASE,
)
_COUNTERPARTY_PHRASE_QUOTED = re.compile(
    rf"\bконтрагент\w*\s+(?:(?P<form>{_ORG_FORM})\s*)?[«\"'](?P<name>[^»\"']+)[»\"']",
    re.IGNORECASE,
)
_DOMAIN_HINTS = (
    _PARTY_HINTS
    + _CREATIVE_STATUS_HINTS
    + _PLACEMENT_HINTS
    + _DOCUMENT_HINTS
    + _CONTRACT_CARD_HINTS
    + _COUNTERPARTY_HINTS
    + _OVERVIEW_HINTS
)


@dataclass
class EntityRefs:
    contract_ids: list[str] = field(default_factory=list)
    creative_ids: list[str] = field(default_factory=list)
    counterparty_ids: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    placement_ids: list[str] = field(default_factory=list)
    contract_numbers: list[str] = field(default_factory=list)
    counterparty_hints: list[str] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return any(
            (
                self.contract_ids,
                self.creative_ids,
                self.counterparty_ids,
                self.document_ids,
                self.placement_ids,
                self.contract_numbers,
                self.counterparty_hints,
            )
        )

    @property
    def primary_contract(self) -> str | None:
        if self.contract_ids:
            return self.contract_ids[0]
        return self.contract_numbers[0] if self.contract_numbers else None

    @property
    def primary_creative(self) -> str | None:
        return self.creative_ids[0] if self.creative_ids else None

    @property
    def primary_counterparty(self) -> str | None:
        return self.counterparty_ids[0] if self.counterparty_ids else None

    @property
    def all_ids(self) -> list[str]:
        return _dedupe(
            [
                *self.contract_ids,
                *self.creative_ids,
                *self.counterparty_ids,
                *self.document_ids,
                *self.placement_ids,
                *self.contract_numbers,
                *self.counterparty_hints,
            ]
        )


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _normalize_contract_number(value: str) -> str | None:
    match = _CONTRACT_NUM.fullmatch(value.strip())
    if match is None:
        return None
    return f"Д-{match.group(1)}/{match.group(2)}"


def _hint(form: str | None, name: str) -> str:
    clean_name = name.strip()
    if not form:
        return clean_name
    return f"{form.upper()} {clean_name}"


def extract_refs(text: str | None) -> EntityRefs:
    """Достать ID и номера сущностей. Порядок первого упоминания сохраняется."""
    if not text:
        return EntityRefs()

    contract_ids = [f"CT-{match.group(1)}" for match in _CONTRACT_SYN.finditer(text)]
    creative_ids = [f"CR-{match.group(1)}" for match in _CREATIVE_SYN.finditer(text)]
    counterparty_ids = [f"CP-{match.group(1)}" for match in _COUNTERPARTY_SYN.finditer(text)]
    document_ids = [f"DOC-{match.group(1)}" for match in _DOCUMENT_SYN.finditer(text)]
    placement_ids = [f"PL-{match.group(1)}" for match in _PLACEMENT_SYN.finditer(text)]
    contract_numbers = [f"Д-{match.group(1)}/{match.group(2)}" for match in _CONTRACT_NUM.finditer(text)]

    counterparty_hints: list[str] = []
    for match in _COUNTERPARTY_PHRASE_QUOTED.finditer(text):
        counterparty_hints.append(_hint(match.group("form"), match.group("name")))
    for match in _COUNTERPARTY_ORG_QUOTED.finditer(text):
        counterparty_hints.append(_hint(match.group("form"), match.group("name")))

    return EntityRefs(
        contract_ids=_dedupe(contract_ids),
        creative_ids=_dedupe(creative_ids),
        counterparty_ids=_dedupe(counterparty_ids),
        document_ids=_dedupe(document_ids),
        placement_ids=_dedupe(placement_ids),
        contract_numbers=_dedupe(contract_numbers),
        counterparty_hints=_dedupe(counterparty_hints),
    )


def has_domain_signal(query: str) -> bool:
    """Есть ли в запросе хотя бы слабый сигнал системных данных."""
    refs = extract_refs(query)
    if refs.has_any:
        return True
    text = query.lower()
    if any(hint in text for hint in _DOMAIN_HINTS):
        return True
    try:
        lemmas = set(lemmatize_text(query))
    except Exception:
        return False
    return bool(lemmas & DOMAIN_SIGNAL_LEMMAS)


@lru_cache(maxsize=1)
def _fixtures() -> dict[str, Any]:
    path = Path(__file__).parents[1] / "mock" / "fixtures" / "scenarios.json"
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _contract_number_to_fixture_id(number: str) -> str | None:
    contracts = _fixtures().get("contracts")
    if not isinstance(contracts, dict):
        return None
    for contract_id, contract in contracts.items():
        if not isinstance(contract_id, str) or not isinstance(contract, dict):
            continue
        candidate = contract.get("contract_number") or contract.get("number")
        if isinstance(candidate, str) and _normalize_contract_number(candidate) == number:
            return contract_id
    return None


def resolve_contract_ref(ref: str | int) -> str | int:
    """Вернуть backend id для CT-N/numeric или fixture id для номера договора.

    У backend нет эндпоинта поиска договора по `contract_number`, поэтому номера
    вида `Д-2025/249` резолвятся мок-first по локальным фикстурам. Если номера
    нет в фикстурах, возвращается нормализованный номер.
    """
    if isinstance(ref, int):
        return ref

    clean = ref.strip()
    contract_match = _CONTRACT_SYN.fullmatch(clean)
    if contract_match is not None:
        return int(contract_match.group(1))
    if clean.isdigit():
        return int(clean)

    number = _normalize_contract_number(clean)
    if number is not None:
        fixture_id = _contract_number_to_fixture_id(number)
        return fixture_id if fixture_id is not None else number

    return clean
