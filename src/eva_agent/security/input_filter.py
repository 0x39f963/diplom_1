"""Вход-фильтр - детерминированный первый guard (ТЗ-3 §2.1).

NFKC-нормализация, strip zero-width/bidi/Cf/tag-chars, decode-and-rescan (base64/hex),
флаг mixed-script (homoglyph-атаки), deny-фразы, лимит размера. Дешево и без LLM -
ловит явные атаки; тонкие отдает `injection_detector` (LLM-judge).
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata

from eva_agent.security.verdict import GuardVerdict

try:  # легкая зависимость; без нее mixed-script просто не проверяется
    from confusable_homoglyphs import confusables
except ImportError:  # pragma: no cover
    confusables = None

_MAX_LEN = 20_000

# Явные инъекционные фразы (RU + EN). Не основной гейт - дополнение к LLM-judge.
_DENY = re.compile(
    r"""
      ignore\s+(all\s+|the\s+|your\s+|previous\s+|prior\s+)*instructions
    | disregard\s+(all\s+|the\s+|previous\s+)*(instructions|rules)
    | (reveal|print|show|repeat|output)\s+(me\s+)?(your\s+|the\s+)?(system\s+)?(prompt|instructions)
    | you\s+are\s+now\s+(a\s+|an\s+)?\w
    | developer\s+mode
    | \bDAN\b
    | jailbreak
    | игнорир\w*\s+(все\s+|предыдущ\w+\s+|эти\s+)*(инструкц\w+|правил\w+|указани\w+)
    | забудь\s+(все\s+|свои\s+)?(инструкц\w+|правил\w+|указани\w+)
    | ты\s+теперь\s+\w
    | притворись,?\s+что
    | покажи\s+(мне\s+)?(свой\s+)?(систем\w+\s+)?промпт
    """,
    re.IGNORECASE | re.VERBOSE,
)

_B64 = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_HEX = re.compile(r"(?:[0-9a-fA-F]{2}){8,}")
_WORD = re.compile(r"\w+", re.UNICODE)


def _strip_invisible(text: str) -> str:
    """Убрать zero-width, bidi-controls, прочие format-символы и Unicode-tag-чары."""
    return "".join(
        ch
        for ch in text
        if not (unicodedata.category(ch) == "Cf" or 0xE0000 <= ord(ch) <= 0xE007F)
    )


def _normalize(text: str) -> str:
    return _strip_invisible(unicodedata.normalize("NFKC", text))


def _decoded_payloads(text: str) -> list[str]:
    """Раскодировать base64/hex-подстроки для пере-сканирования на инъекции."""
    out: list[str] = []
    for match in _B64.finditer(text):
        try:
            out.append(base64.b64decode(match.group(), validate=True).decode("utf-8", "ignore"))
        except (ValueError, binascii.Error):
            continue
    for match in _HEX.finditer(text):
        try:
            out.append(bytes.fromhex(match.group()).decode("utf-8", "ignore"))
        except ValueError:
            continue
    return out


def _has_mixed_script_word(text: str) -> bool:
    """True, если ОТДЕЛЬНОЕ слово мешает скрипты (homoglyph-атака).

    Проверяем по словам, а не по всей строке: латиница рядом с кириллицей разными
    словами (erid, ОРД) - норма для нашего домена, а не атака.
    """
    if confusables is None:
        return False
    for word in _WORD.findall(text):
        if len(word) < 2:
            continue
        try:
            if confusables.is_mixed_script(word):
                return True
        except Exception:
            continue
    return False


def filter_input(raw: str) -> GuardVerdict:
    """Проверить пользовательский ввод. Приоритет решений: block > sanitize > allow."""
    categories: list[str] = []
    risk = 0.0

    clean = _normalize(raw)
    sanitized = clean if clean != raw else None
    if sanitized is not None:
        categories.append("normalized")
        risk = max(risk, 0.15)

    if _DENY.search(clean):
        return GuardVerdict(
            decision="block",
            risk_score=0.9,
            categories=[*categories, "injection_phrase"],
            reason="Обнаружена явная инструкция-инъекция во вводе.",
        )

    for decoded in _decoded_payloads(clean):
        if _DENY.search(decoded):
            return GuardVerdict(
                decision="block",
                risk_score=0.9,
                categories=[*categories, "encoded_injection"],
                reason="Инъекция в закодированной (base64/hex) подстроке.",
            )

    if _has_mixed_script_word(clean):
        categories.append("mixed_script")
        risk = max(risk, 0.5)

    if len(clean) > _MAX_LEN:
        categories.append("oversize")
        risk = max(risk, 0.4)

    decision = "sanitize" if sanitized is not None else "allow"
    return GuardVerdict(
        decision=decision,
        risk_score=risk,
        categories=categories,
        reason="ok",
        sanitized_text=sanitized,
    )
