"""Deterministic RU preprocessing for intent parsing."""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from typing import Literal, cast

import dateparser
from dateparser.search import search_dates
from pydantic import BaseModel, Field

from eva_agent.nlu.gazetteer import (
    ACTION_BY_LEMMA,
    DATE_HINT_BY_LEMMA,
    ENTITY_BY_LEMMA,
    MONTH_BY_LEMMA,
    ROLE_BY_LEMMA,
    STATUS_BY_LEMMA,
)
from eva_agent.nlu.ru import lemmatize_tokens, tokenize
from eva_agent.tools.entity_ref import EntityRefs, extract_refs

DateHint = Literal["none", "date", "yesterday", "last_week", "last_month", "month"]

_EXPLICIT_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b")
_READ_ACTIONS = frozenset({"read", "list", "search", "open", "download", "show"})
_WRITE_ACTIONS = frozenset({"attach", "delete", "update", "send"})
_WRITE_LEMMAS = frozenset(
    {
        "добавить",
        "изменить",
        "обновить",
        "отправить",
        "переслать",
        "передать",
        "прикрепить",
        "приложить",
        "удалить",
        "удаль",
        "убрать",
    }
)
_WRITE_TEXT_MARKERS = (
    "добавь",
    "добавить",
    "измени",
    "изменить",
    "обнови",
    "обновить",
    "отправь",
    "отправить",
    "передай",
    "передать",
    "перешли",
    "переслать",
    "прикрепи",
    "прикрепить",
    "приложи",
    "приложить",
    "удали",
    "удалить",
    "убери",
    "убрать",
)


class DateFeature(BaseModel):
    text: str = ""
    date_hint: DateHint = "none"
    start: str | None = None
    end: str | None = None


class NluFeatures(BaseModel):
    query: str
    tokens: list[str] = Field(default_factory=list)
    lemmas: list[str] = Field(default_factory=list)
    entity_ids: dict[str, list[str]] = Field(default_factory=dict)
    entities: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)
    dates: list[DateFeature] = Field(default_factory=list)
    date_hint: DateHint = "none"
    action_verbs: list[str] = Field(default_factory=list)


def preprocess(query: str) -> NluFeatures:
    tokens = tokenize(query)
    lemmas = lemmatize_tokens(tokens)
    refs = extract_refs(query)
    dates = _extract_dates(query, lemmas)
    return NluFeatures(
        query=query,
        tokens=tokens,
        lemmas=lemmas,
        entity_ids=_refs_to_dict(refs),
        entities=_entities(lemmas),
        roles=_roles(lemmas),
        statuses=_statuses(query, lemmas),
        dates=dates,
        date_hint=dates[0].date_hint if dates else "none",
        action_verbs=_actions(lemmas),
    )


def is_read_only_domain_command(query: str) -> bool:
    features = preprocess(query)
    actions = set(features.action_verbs)
    if actions & _WRITE_ACTIONS:
        return False
    if set(features.lemmas) & _WRITE_LEMMAS:
        return False
    lowered = query.lower().replace("ё", "е")
    if any(marker in lowered for marker in _WRITE_TEXT_MARKERS):
        return False
    if actions & _READ_ACTIONS:
        return True
    return bool(extract_refs(query).all_ids)


def _refs_to_dict(refs: EntityRefs) -> dict[str, list[str]]:
    return {
        "contracts": refs.contract_ids,
        "creatives": refs.creative_ids,
        "counterparties": refs.counterparty_ids,
        "documents": refs.document_ids,
        "placements": refs.placement_ids,
        "contract_numbers": refs.contract_numbers,
        "counterparty_hints": refs.counterparty_hints,
    }


def _entities(lemmas: list[str]) -> list[str]:
    values: list[str] = []
    for lemma in lemmas:
        values.extend(ENTITY_BY_LEMMA.get(lemma, ()))
    return _unique(values)


def _roles(lemmas: list[str]) -> list[str]:
    return _unique(ROLE_BY_LEMMA[lemma] for lemma in lemmas if lemma in ROLE_BY_LEMMA)


def _statuses(query: str, lemmas: list[str]) -> list[str]:
    lowered = _norm_text(query)
    values = [STATUS_BY_LEMMA[lemma] for lemma in lemmas if lemma in STATUS_BY_LEMMA]
    if "неподпис" in lowered or "не подпис" in lowered:
        values.append("unsigned")
    if "незарегистр" in lowered:
        values.append("unregistered")
    elif "зарегистр" in lowered:
        values.append("registered")
    if "чернов" in lowered:
        values.append("draft")
    return _unique(values)


def _actions(lemmas: list[str]) -> list[str]:
    return _unique(ACTION_BY_LEMMA[lemma] for lemma in lemmas if lemma in ACTION_BY_LEMMA)


def _extract_dates(query: str, lemmas: list[str]) -> list[DateFeature]:
    today = date.today()
    lemma_set = set(lemmas)
    lowered = _norm_text(query)
    features: list[DateFeature] = []

    for lemma, hint in DATE_HINT_BY_LEMMA.items():
        if lemma in lemma_set:
            target = today - timedelta(days=1)
            features.append(_range_feature(lemma, cast(DateHint, hint), target, target))

    if "прошлый" in lemma_set and "неделя" in lemma_set:
        current_week_start = today - timedelta(days=today.weekday())
        start = current_week_start - timedelta(days=7)
        features.append(_range_feature("прошлая неделя", "last_week", start, start + timedelta(days=6)))

    if "прошлый" in lemma_set and "месяц" in lemma_set:
        month = today.month - 1 or 12
        year = today.year - 1 if today.month == 1 else today.year
        features.append(_month_feature("прошлый месяц", year, month, "last_month"))

    month_features = [
        _month_feature(f"за {lemma}", today.year, month, "month")
        for lemma, month in MONTH_BY_LEMMA.items()
        if lemma in lemma_set and f"за {lemma}" in lowered
    ]
    features.extend(month_features)

    if not features and _looks_like_explicit_date(query, lemma_set):
        parsed = _search_date(query, today)
        if parsed is not None:
            text, parsed_date = parsed
            features.append(_range_feature(text, "date", parsed_date, parsed_date))

    return _dedupe_dates(features)


def _search_date(query: str, today: date) -> tuple[str, date] | None:
    settings = {
        "RELATIVE_BASE": datetime.combine(today, time.min),
        "PREFER_DATES_FROM": "past",
        "DATE_ORDER": "DMY",
    }
    parsed = search_dates(query, languages=["ru"], settings=settings)
    if parsed:
        text, value = parsed[0]
        return text, value.date()
    value = dateparser.parse(query, languages=["ru"], settings=settings)
    return (query, value.date()) if value is not None else None


def _looks_like_explicit_date(query: str, lemma_set: set[str]) -> bool:
    return bool(_EXPLICIT_DATE_RE.search(query)) or any(lemma in MONTH_BY_LEMMA for lemma in lemma_set)


def _range_feature(text: str, hint: DateHint, start: date, end: date) -> DateFeature:
    return DateFeature(text=text, date_hint=hint, start=start.isoformat(), end=end.isoformat())


def _month_feature(text: str, year: int, month: int, hint: DateHint) -> DateFeature:
    days = calendar.monthrange(year, month)[1]
    return _range_feature(text, hint, date(year, month, 1), date(year, month, days))


def _dedupe_dates(features: list[DateFeature]) -> list[DateFeature]:
    seen: set[tuple[str, str | None, str | None]] = set()
    out: list[DateFeature] = []
    for feature in features:
        key = (feature.date_hint, feature.start, feature.end)
        if key in seen:
            continue
        out.append(feature)
        seen.add(key)
    return out


def _norm_text(text: str) -> str:
    return text.lower().replace("ё", "е")


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
