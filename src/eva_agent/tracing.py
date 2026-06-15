"""LangFuse-группировка: один запрос пользователя = ОДИН трейс с деревом вызовов.

Зачем: `langfuse.openai` трейсит каждый вызов LLM по отдельности. Без общего корня
они разлетаются на отдельные трейсы, и не видно «как шел запрос через агентов».
Здесь мы оборачиваем весь прогон графа в корневой span (трейс), а каждый узел -
в дочерний span. Получается дерево: запрос -> узлы (supervisor/legal_agent/critic/...)
-> вызовы LLM (с моделью, токенами, стоимостью) -> финальный ответ.

Модель фиксируется: имя трейса содержит модель роли reasoning, в metadata - модели всех
ролей, а каждый вызов LLM (langfuse.openai) сам пишет свою модель в generation.

Без ключей LANGFUSE_* все становится no-op: обертки возвращают исходные функции,
накладных нет, тесты идут офлайн.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from eva_agent.llm.observability import langfuse_enabled
from eva_agent.settings import Role, settings

_ROLES: tuple[Role, ...] = ("reasoning", "default", "guard")


def _model_label() -> str:
    """Главная модель прогона (роль reasoning) - попадает в имя трейса."""
    return f"{settings.role_backend('reasoning')}:{settings.role_model('reasoning')}"


def _models_meta() -> dict[str, str]:
    """Модель на каждой роли - в metadata трейса (видно, через что шел прогон)."""
    return {role: f"{settings.role_backend(role)}:{settings.role_model(role)}" for role in _ROLES}


def traced_node[NodeF: Callable[..., dict]](name: str, fn: NodeF) -> NodeF:
    """Обернуть узел графа в span LangFuse (узел дерева). No-op без ключей LANGFUSE_*.

    Дженерик по типу функции-узла (PEP 695): сохраняет точную сигнатуру, чтобы
    LangGraph.add_node принял обертку (его перегрузка требует точный тип узла).
    """
    if not langfuse_enabled():
        return fn
    from langfuse import observe

    return observe(name=name, as_type="agent", capture_input=False, capture_output=False)(fn)


def run_request(graph: Any, user_input: str) -> dict:
    """Прогон графа как ОДИН трейс LangFuse: человек -> узлы-агенты -> ответ.

    Возвращает финальное состояние графа (как и обычный graph.invoke). Без ключей -
    обычный invoke без накладных.
    """
    payload = {"user_input_raw": user_input}
    if not langfuse_enabled():
        return graph.invoke(payload)

    from langfuse import get_client

    client = get_client()
    with client.start_as_current_observation(
        name=f"agent-request - {_model_label()}",
        as_type="chain",
        input=user_input,
        metadata={"models": _models_meta()},
    ):
        state = graph.invoke(payload)
        final = state.get("final") if isinstance(state, dict) else None
        client.update_current_span(output=final)
        client.set_current_trace_io(input=user_input, output=final)
    client.flush()
    return state
