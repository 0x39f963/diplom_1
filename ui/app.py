"""Демо web-чат поверх графа агента (Chainlit).

Запуск: PYTHONPATH=src RAG_API_BASE=http://localhost:8077 chainlit run ui/app.py
Показывает ответ + сворачиваемый шаг «Как агент отработал» (защита, интент, данные, цитаты).
"""

from __future__ import annotations

import asyncio

import chainlit as cl

from eva_agent.graph import build_graph
from eva_agent.tracing import run_request

_graph = build_graph()

_GREETING = (
    "Я - помощник маркетолога. Отвечаю на вопросы по закону о рекламе и помогаю разобраться "
    "с договорами, контрагентами и размещениями в вашей системе. Чем помочь?"
)


@cl.on_chat_start
async def start() -> None:
    await cl.Message(content=_GREETING).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    state = await asyncio.to_thread(run_request, _graph, message.content)

    guard_in = state.get("guard_in")
    intent = state.get("intent")
    guard_out = state.get("guard_out")
    findings = state.get("api_findings") or []
    citations = list(dict.fromkeys(state.get("citations") or []))

    async with cl.Step(name="Как агент отработал") as step:
        lines = []
        if guard_in is not None:
            cats = ", ".join(guard_in.categories) or "чисто"
            lines.append(f"Защита входа: {guard_in.decision} ({cats})")
        if intent is not None:
            lines.append(f"Интент: {intent.kind} (уверенность {intent.confidence:.0%})")
        if findings:
            lines.append("Данные системы (read-only): " + ", ".join(f.tool for f in findings))
        if citations:
            lines.append("Нормы: " + ", ".join(citations))
        if guard_out is not None:
            lines.append(f"Защита выхода: {guard_out.decision}")
        step.output = "\n".join(lines) or "-"

    await cl.Message(content=state.get("final") or "(пустой ответ)").send()
