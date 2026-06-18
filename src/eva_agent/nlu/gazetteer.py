"""Curated domain gazetteer matched by lemmas."""

from __future__ import annotations

ENTITY_BY_LEMMA: dict[str, tuple[str, ...]] = {
    "контрагент": ("Counterparty",),
    "клиент": ("Counterparty",),
    "партнер": ("Counterparty",),
    "договор": ("Contract",),
    "контракт": ("Contract",),
    "соглашение": ("Contract",),
    "креатив": ("Creative",),
    "баннер": ("Creative",),
    "объявление": ("Creative",),
    "материал": ("Creative",),
    "размещение": ("Placement",),
    "площадка": ("Placement",),
    "показ": ("Placement",),
    "публикация": ("Placement",),
    "документ": ("Document",),
    "файл": ("Document",),
    "акт": ("Document",),
    "приложение": ("Document",),
    "сторона": ("ContractParty",),
    "заказчик": ("ContractParty",),
    "исполнитель": ("ContractParty",),
    "участник": ("ContractParty",),
}

ROLE_BY_LEMMA: dict[str, str] = {
    "заказчик": "customer",
    "исполнитель": "executor",
}

STATUS_BY_LEMMA: dict[str, str] = {
    "неподписанный": "unsigned",
    "подписанный": "signed",
    "зарегистрированный": "registered",
    "незарегистрированный": "unregistered",
    "черновик": "draft",
    "черновой": "draft",
    "неоформленный": "unsigned",
    "незакрытый": "unsigned",
    "незавершенный": "unsigned",
}

ACTION_BY_LEMMA: dict[str, str] = {
    "открыть": "open",
    "прочитать": "read",
    "читать": "read",
    "скачать": "download",
    "загрузить": "download",
    "прикрепить": "attach",
    "добавить": "attach",
    "приложить": "attach",
    "показать": "show",
    "посмотреть": "show",
    "сравнить": "compare",
    "вывести": "show",
    "перечислить": "list",
    "найти": "search",
    "проверить": "check",
}

DATE_HINT_BY_LEMMA: dict[str, str] = {
    "вчера": "yesterday",
    "вчерашний": "yesterday",
}

MONTH_BY_LEMMA: dict[str, int] = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}

DOMAIN_SIGNAL_LEMMAS: frozenset[str] = frozenset(
    set(ENTITY_BY_LEMMA)
    | set(ROLE_BY_LEMMA)
    | set(STATUS_BY_LEMMA)
    | set(ACTION_BY_LEMMA)
    | {
        "закон",
        "норма",
        "статья",
        "обязан",
        "разрешить",
        "запретить",
        "требование",
        "статус",
        "блокировать",
        "готовность",
        "готовый",
        "выпустить",
        "период",
        "номер",
        "дата",
        "инн",
        "реквизит",
        "обзор",
        "сводка",
        "оформить",
        "хватать",
    }
)
