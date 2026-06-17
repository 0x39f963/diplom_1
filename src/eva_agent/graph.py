"""Сборка графа LangGraph (ТЗ-2 §1): guards-in -> supervisor -> агенты -> critic -> guards-out -> finalize.

Три оси ветвления: маршрутизация supervisor, critic accept/rework (повтор решает loop-policy),
блокировки guard-узлов. Решение о повторе - у графа по loop-policy, критик лишь выносит вердикт.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from eva_agent.nodes.agents import (
    clarify,
    critic,
    data_gather,
    finalize,
    interface_agent,
    legal_agent,
    refuse,
    supervisor,
)
from eva_agent.nodes.dialog_nodes import load_context, save_turn
from eva_agent.nodes.domain_nodes import domain_selector
from eva_agent.nodes.guards import input_guard, output_guard
from eva_agent.settings import settings
from eva_agent.state import AgentState
from eva_agent.tracing import traced_node

_INTENT_ROUTE = {
    "legal_consult": "legal_agent",
    "interface_consult": "interface_agent",
    "mixed_diagnostic": "domain_selector",
    "need_clarification": "clarify",
    "out_of_scope": "refuse",
}


def _route_after_input(state: AgentState) -> str:
    return "refuse" if state.guard_in and state.guard_in.decision == "block" else "load_context"


def _route_after_supervisor(state: AgentState) -> str:
    kind = state.intent.kind if state.intent else "out_of_scope"
    return _INTENT_ROUTE.get(kind, "refuse")


def _route_after_data_gather(state: AgentState) -> str:
    """Если планировщик запросил уточнение, завершаем ход через clarify."""
    if state.todo_plan is not None and state.todo_plan.status == "awaiting_clarification":
        return "clarify"
    if state.intent is not None and state.intent.kind == "need_clarification":
        return "clarify"
    return "interface_agent"


def _route_after_critic(state: AgentState) -> str:
    """Loop-policy: accept -> finalize; rework и бюджет цикла не исчерпан -> target; иначе finalize."""
    verdict = state.critic
    if not verdict or verdict.decision == "accept":
        return "finalize"
    if verdict.target and state.loop_count <= settings.max_loops:
        return verdict.target
    return "finalize"


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)
    # traced_node - обертка в span LangFuse (узел дерева вызовов); no-op без ключей.
    graph.add_node("input_guard", traced_node("input_guard", input_guard))
    graph.add_node("load_context", traced_node("load_context", load_context))
    graph.add_node("supervisor", traced_node("supervisor", supervisor))
    graph.add_node("legal_agent", traced_node("legal_agent", legal_agent))
    graph.add_node("interface_agent", traced_node("interface_agent", interface_agent))
    graph.add_node("domain_selector", traced_node("domain_selector", domain_selector))
    graph.add_node("data_gather", traced_node("data_gather", data_gather))
    graph.add_node("critic", traced_node("critic", critic))
    graph.add_node("finalize", traced_node("finalize", finalize))
    graph.add_node("output_guard", traced_node("output_guard", output_guard))
    graph.add_node("clarify", traced_node("clarify", clarify))
    graph.add_node("refuse", traced_node("refuse", refuse))
    graph.add_node("save_turn", traced_node("save_turn", save_turn))

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard", _route_after_input, {"refuse": "refuse", "load_context": "load_context"}
    )
    graph.add_edge("load_context", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "legal_agent": "legal_agent",
            "interface_agent": "interface_agent",
            "domain_selector": "domain_selector",
            "data_gather": "data_gather",
            "clarify": "clarify",
            "refuse": "refuse",
        },
    )
    graph.add_edge("domain_selector", "data_gather")
    graph.add_conditional_edges(
        "data_gather",
        _route_after_data_gather,
        {"interface_agent": "interface_agent", "clarify": "clarify"},
    )
    graph.add_edge("legal_agent", "critic")
    graph.add_edge("interface_agent", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "finalize": "finalize",
            "legal_agent": "legal_agent",
            "interface_agent": "interface_agent",
            "data_gather": "data_gather",
        },
    )
    graph.add_edge("finalize", "output_guard")
    graph.add_edge("output_guard", "save_turn")
    graph.add_edge("clarify", "save_turn")
    graph.add_edge("refuse", "save_turn")
    graph.add_edge("save_turn", END)
    return graph.compile()
