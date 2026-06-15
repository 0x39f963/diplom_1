"""CLI-демо агента: `python -m eva_agent.cli "<запрос>"` (или `make cli Q="..."`)."""

from __future__ import annotations

import sys

from eva_agent.graph import build_graph
from eva_agent.tracing import run_request


def main() -> int:
    query = " ".join(sys.argv[1:]).strip() or "Нужен ли ERID для рекламного баннера на сайте?"
    graph = build_graph()
    result = run_request(graph, query)  # один трейс LangFuse на запрос (дерево вызовов)
    print(result.get("final") or "(пустой ответ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
