"""Кастомные RU-распознаватели PII (Presidio из коробки не знает РФ-сущности).

Детектируем ИНН (10/12 цифр + контрольная сумма), СНИЛС (контроль), паспорт РФ, телефон.
Контрольные суммы убирают ложные срабатывания на произвольных числах.
"""

from __future__ import annotations

import re

_INN12_RE = re.compile(r"\b\d{12}\b")
_INN10_RE = re.compile(r"\b\d{10}\b")
_SNILS_RE = re.compile(r"\b\d{3}-\d{3}-\d{3}[ -]\d{2}\b")
_PHONE_RE = re.compile(r"(?:\+7|8)[\s(]?\d{3}[\s)]?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b")
_PASSPORT_RE = re.compile(r"паспорт\D{0,15}(\d{2}\s?\d{2}\s?\d{6})\b", re.IGNORECASE)


def _inn10_valid(digits: str) -> bool:
    weights = [2, 4, 10, 3, 5, 9, 4, 6, 8]
    checksum = sum(int(digits[i]) * weights[i] for i in range(9)) % 11 % 10
    return checksum == int(digits[9])


def _inn12_valid(digits: str) -> bool:
    w1 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
    w2 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
    n1 = sum(int(digits[i]) * w1[i] for i in range(10)) % 11 % 10
    n2 = sum(int(digits[i]) * w2[i] for i in range(11)) % 11 % 10
    return n1 == int(digits[10]) and n2 == int(digits[11])


def _snils_valid(digits: str) -> bool:
    total = sum(int(digits[i]) * (9 - i) for i in range(9)) % 101
    if total == 100:
        total = 0
    return total == int(digits[9:11])


def find_pii(text: str) -> list[tuple[str, str]]:
    """Список (тип, найденная_подстрока). Только валидные по контролю ИНН/СНИЛС."""
    found: list[tuple[str, str]] = []
    for match in _INN12_RE.finditer(text):
        if _inn12_valid(match.group()):
            found.append(("ИНН", match.group()))
    for match in _INN10_RE.finditer(text):
        if _inn10_valid(match.group()):
            found.append(("ИНН", match.group()))
    for match in _SNILS_RE.finditer(text):
        if _snils_valid(re.sub(r"\D", "", match.group())):
            found.append(("СНИЛС", match.group()))
    for match in _PHONE_RE.finditer(text):
        found.append(("телефон", match.group()))
    for match in _PASSPORT_RE.finditer(text):
        found.append(("паспорт", match.group(1)))
    return found


def mask_pii(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Заменить найденные PII на [СКРЫТО:<тип>]. Возвращает (очищенный_текст, найденное)."""
    found = find_pii(text)
    masked = text
    for pii_type, value in found:
        masked = masked.replace(value, f"[СКРЫТО:{pii_type}]")
    return masked, found
