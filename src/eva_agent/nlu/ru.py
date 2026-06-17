"""Russian tokenization and lemmatization."""

from __future__ import annotations

import re
from functools import lru_cache

import pymorphy3
from razdel import tokenize as razdel_tokenize

_WORD_RE = re.compile(r"[\w-]+", re.UNICODE)
_RU_RE = re.compile(r"[а-яА-Я]")


@lru_cache(maxsize=1)
def _morph() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer()


def tokenize(text: str) -> list[str]:
    return [item.text for item in razdel_tokenize(text or "") if _WORD_RE.search(item.text)]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    lemmas: list[str] = []
    morph = _morph()
    for token in tokens:
        lowered = token.lower()
        if _RU_RE.search(lowered):
            parsed = morph.parse(lowered)
            lemmas.append(parsed[0].normal_form if parsed else lowered)
        else:
            lemmas.append(lowered)
    return lemmas


def lemmatize_text(text: str) -> list[str]:
    return lemmatize_tokens(tokenize(text))
