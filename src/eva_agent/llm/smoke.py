"""Smoke LLM-коннекторов (ТЗ-4 §8): OpenRouter + локальный Ollama.

Запуск: `python -m eva_agent.llm.smoke` (или `make smoke`). Делает по одному дешевому
вызову на каждый бэкенд и печатает текст + usage + walltime.
"""

from __future__ import annotations

import sys

from eva_agent.llm.base import LLMClient
from eva_agent.llm.config import get_client

_SYSTEM = "You are a terse assistant. Reply with exactly one word, no punctuation."
_USER = "Reply with the word: pong"


def _probe(label: str, client: LLMClient) -> bool:
    print(f"-- {label} ({client.backend}:{client.model}) --")
    try:
        resp = client.invoke(_SYSTEM, _USER, temperature=0.0)
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        return False
    print(f"  text={resp.text.strip()!r}  usage={resp.usage}  walltime={resp.walltime_sec:.2f}s")
    return bool(resp.text.strip())


def main() -> int:
    ok = True
    ok &= _probe("OpenRouter", get_client("reasoning"))
    ok &= _probe("Local Ollama", get_client("default"))
    print("OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
