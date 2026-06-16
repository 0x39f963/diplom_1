"""CLI-демо агента: `python -m eva_agent.cli "<запрос>"` (или `make cli Q="..."`)."""

from __future__ import annotations

import sys

from eva_agent.graph import build_graph
from eva_agent.settings import settings
from eva_agent.tracing import run_request


def main() -> int:
    args = sys.argv[1:]
    session_id = settings.eva_dialog_session or None
    if "--session" in args:
        index = args.index("--session")
        if index + 1 >= len(args):
            raise SystemExit("--session requires a value")
        session_id = args[index + 1]
        args = args[:index] + args[index + 2 :]
    query = " ".join(args).strip() or "Нужен ли ERID для рекламного баннера на сайте?"
    graph = build_graph()
    result = run_request(graph, query, session_id=session_id)  # один трейс LangFuse на запрос
    print(result.get("final") or "(пустой ответ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
