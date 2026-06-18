"""Прогон бенчмарка + 3 типа eval + метрики (ТЗ-2 §8).

Запуск: PYTHONPATH=src python -m evals.run_evals  (нужен поднятый retrieval API + .env).

Три типа проверок:
  1) программный assert - детерминированные инварианты (интент, цитата, блокировка, tool);
  2) LLM-as-judge - отдельной моделью оцениваем релевантность/обоснованность ответа;
  3) tool-call-correctness - для диагностических кейсов правильный ли tool вызван.
Метрики: success-rate, latency p50/p95, cost per run.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from eva_agent import metrics
from eva_agent.graph import build_graph
from eva_agent.llm.config import get_client
from eva_agent.settings import Role, settings
from eva_agent.tracing import run_request

_DEFAULT_BENCH = Path(__file__).resolve().parents[1] / "bench" / "benchmark.jsonl"
# BENCH_FILE=bench/benchmark_big.jsonl - прогнать расширенный набор (122 кейса).
_BENCH = Path(os.environ["BENCH_FILE"]) if os.environ.get("BENCH_FILE") else _DEFAULT_BENCH
# EVAL_SLEEP_SEC - пауза между кейсами (бережем лимиты подписочных бэкендов).
_SLEEP_SEC = float(os.environ.get("EVAL_SLEEP_SEC", "0"))

_TOOL_ENTITIES: dict[str, set[str]] = {
    "eva_get_contract": {"contract"},
    "eva_get_contract_parties": {"party", "counterparty"},
    "eva_get_counterparty": {"counterparty"},
    "eva_get_creative_status": {"creative", "contract"},
    "eva_list_placements": {"placement", "creative", "contract"},
    "eva_list_contract_documents": {"document", "contract"},
    "eva_list_unsigned_contracts": {"contract"},
    "eva_search_contracts": {"contract"},
    "retrieve_legal": set(),
    "eva_doc_read": {"document"},
    "eva_doc_download": {"document"},
    "eva_doc_attach": {"document", "contract"},
}
_GOLD_ROUTES = {"data", "legal", "clarify"}


def load_cases() -> list[dict[str, Any]]:
    lines = _BENCH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _got_tools(state: dict[str, Any]) -> list[str]:
    findings = state.get("api_findings") or []
    return [str(f.tool) for f in findings]


def _ordered_subset(want: list[str], got: list[str]) -> bool:
    position = -1
    for tool in want:
        try:
            position = got.index(tool, position + 1)
        except ValueError:
            return False
    return True


def _as_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _check_expected_tools(assertion: dict[str, Any], got: list[str]) -> bool:
    want = _as_list(assertion.get("value"))
    if not want:
        return False
    if bool(assertion.get("soft")):
        return any(tool in got for tool in want)
    if not all(tool in got for tool in want):
        return False
    if bool(assertion.get("ordered")):
        return _ordered_subset(want, got)
    return True


def _got_entities(got_tools: list[str]) -> set[str]:
    entities: set[str] = set()
    for tool in got_tools:
        entities.update(_TOOL_ENTITIES.get(tool, set()))
    return entities


def _check_expected_entities(assertion: dict[str, Any], got_tools: list[str]) -> bool:
    want = set(_as_list(assertion.get("value")))
    if not want:
        return False
    got = _got_entities(got_tools)
    if bool(assertion.get("soft")):
        return bool(want & got)
    return want <= got


def _intent_kind(state: dict[str, Any]) -> str | None:
    intent = state.get("intent")
    value = getattr(intent, "kind", None)
    return str(value) if value is not None else None


def _todo_status(state: dict[str, Any]) -> str | None:
    todo_plan = state.get("todo_plan")
    value = getattr(todo_plan, "status", None)
    return str(value) if value is not None else None


def _plan_capture(state: dict[str, Any]) -> dict[str, Any] | None:
    """Снимок решения планировщика для аудита: протокол, статус, уверенность, todo с блокерами."""
    plan = state.get("todo_plan")
    if plan is None:
        return None
    items = [
        {
            "id": it.id,
            "type": it.type,
            "status": it.status,
            "blockers": list(it.blockers),
            "tools": [s.tool for s in it.tool_calls],
        }
        for it in getattr(plan, "items", [])
    ]
    return {
        "protocol_id": getattr(plan, "protocol_id", None),
        "status": getattr(plan, "status", None),
        "confidence": getattr(plan, "confidence", None),
        "clarify_question": getattr(plan, "clarify_question", ""),
        "clarify_code": getattr(plan, "clarify_code", ""),
        "empty_plan": getattr(plan, "is_empty", False),
        "items": items,
    }


def _frame_confidence(state: dict[str, Any]) -> float | None:
    frame = state.get("frame")
    value = getattr(frame, "confidence", None)
    return float(value) if isinstance(value, int | float) else None


def _decomposed(state: dict[str, Any]) -> bool:
    frame = state.get("frame")
    return bool(getattr(frame, "subtasks", []))


def _clarify_code(state: dict[str, Any]) -> str:
    plan = state.get("todo_plan")
    value = getattr(plan, "clarify_code", "")
    return str(value) if value else ""


def _empty_plan(state: dict[str, Any]) -> bool:
    plan = state.get("todo_plan")
    return bool(getattr(plan, "is_empty", False))


def _debug_capture(state: dict[str, Any]) -> dict[str, Any]:
    debug = state.get("debug")
    if isinstance(debug, dict) and debug:
        return debug

    guard = state.get("guard_in")
    frame = state.get("frame")
    plan = state.get("todo_plan")
    fallback: dict[str, Any] = {}
    if guard is not None:
        fallback["guard"] = {
            "decision": getattr(guard, "decision", None),
            "risk_type": getattr(guard, "risk_type", None),
        }
    if frame is not None:
        fallback["frame"] = {
            "target": getattr(frame, "target", None),
            "relation": getattr(frame, "relation", None),
        }
    if plan is not None:
        fallback["plan"] = {
            "protocol_id": getattr(plan, "protocol_id", None),
            "clarify_code": getattr(plan, "clarify_code", ""),
        }
    return fallback


def _is_clarification(state: dict[str, Any]) -> bool:
    return _intent_kind(state) == "need_clarification" or _todo_status(state) == "awaiting_clarification"


def _actual_route(state: dict[str, Any]) -> str:
    if _is_clarification(state):
        return "clarify"
    intent = _intent_kind(state)
    if intent == "mixed_diagnostic":
        return "data"
    if intent == "legal_consult":
        return "legal"
    return intent or "unknown"


def _assert_must_clarify(assertion: dict[str, Any]) -> bool:
    if assertion["type"] != "must_clarify":
        return False
    return bool(assertion.get("value", True))


def case_must_clarify(case: dict[str, Any]) -> bool:
    if bool(case.get("must_clarify")):
        return True
    return any(_assert_must_clarify(assertion) for assertion in case.get("asserts", []))


def case_clarify_warranted(case: dict[str, Any]) -> bool:
    value = case.get("clarify_warranted")
    if isinstance(value, bool):
        return value
    return case_must_clarify(case)


def _gold_route(case: dict[str, Any]) -> str | None:
    value = case.get("gold_route")
    return value if isinstance(value, str) and value in _GOLD_ROUTES else None


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_quality_metrics(
    evaluated: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, int | float]:
    route_total = route_ok = 0
    clarify_total = warranted_total = warranted_clarified = 0
    avoidable_clarifications = 0
    empty_plans = 0

    for case, state in evaluated:
        actual_route = _actual_route(state)
        expected_route = _gold_route(case)
        if expected_route is not None:
            route_total += 1
            route_ok += int(actual_route == expected_route)

        clarified = _is_clarification(state)
        warranted = case_clarify_warranted(case)
        clarify_total += int(clarified)
        warranted_total += int(warranted)
        warranted_clarified += int(clarified and warranted)
        avoidable_clarifications += int(clarified and not warranted)
        empty_plans += int(_empty_plan(state))

    total = len(evaluated)
    return {
        "route_ok": route_ok,
        "route_total": route_total,
        "route_accuracy": _rate(route_ok, route_total),
        "clarify_total": clarify_total,
        "clarify_warranted_total": warranted_total,
        "clarify_warranted_hit": warranted_clarified,
        "clarify_precision": _rate(warranted_clarified, clarify_total),
        "clarify_recall": _rate(warranted_clarified, warranted_total),
        "avoidable_clarifications": avoidable_clarifications,
        "avoidable_clarification_rate": _rate(avoidable_clarifications, total),
        "empty_plan": empty_plans,
        "empty_plan_rate": _rate(empty_plans, total),
    }


def known_fail_reason(case: dict[str, Any]) -> str | None:
    raw = case.get("known_fail")
    if isinstance(raw, dict):
        return str(raw.get("reason") or "known fail") if raw.get("value", True) else None
    if raw is True:
        return "known fail"
    return None


def check_assert(assertion: dict, state: dict[str, Any]) -> bool:
    kind = assertion["type"]
    guard_in = state.get("guard_in")
    final = state.get("final") or ""
    got_tools = _got_tools(state)
    citations = state.get("citations") or []
    blocked = guard_in is not None and guard_in.decision == "block"
    if kind == "intent":
        return _intent_kind(state) == assertion["value"]
    if kind == "blocked":
        return blocked
    if kind == "not_blocked":
        return not blocked
    if kind == "has_citation":
        return bool(citations) or "ст." in final
    if kind == "used_tool":
        return assertion["value"] in got_tools
    if kind == "expected_tools":
        return _check_expected_tools(assertion, got_tools)
    if kind == "expected_entities":
        return _check_expected_entities(assertion, got_tools)
    if kind == "mutating_tool":
        return assertion["value"] in got_tools
    if kind == "must_clarify":
        return _is_clarification(state) if bool(assertion.get("value", True)) else not _is_clarification(state)
    return False


_JUDGE_SYS = (
    "Ты - строгий оценщик ответов ИИ-помощника по рекламному праву (38-ФЗ). Верни СТРОГО JSON "
    '{"ok": true|false}: релевантен ли ОТВЕТ ВОПРОСУ и опирается ли на нормы/данные (а не вода).'
)


def llm_judge(question: str, answer: str) -> bool:
    try:
        response = get_client("guard").invoke(
            _JUDGE_SYS, f"ВОПРОС: {question}\n\nОТВЕТ: {answer}", temperature=0.0, json_mode=True
        )
        return bool(json.loads(response.text).get("ok"))
    except Exception:
        return False


def _models_under_test() -> dict[str, str]:
    """Снимок «какая модель на какой роли» - для шапки отчета."""
    roles: tuple[Role, ...] = ("reasoning", "default", "guard", "planner", "memory")
    return {
        role: f"{settings.role_backend(role)}:{settings.role_model(role)}"
        for role in roles
    }


def _tool_asserts(asserts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        assertion
        for assertion in asserts
        if assertion["type"] in {"used_tool", "expected_tools", "mutating_tool"}
    ]


def _expected_tools_for_row(asserts: list[dict[str, Any]]) -> list[str]:
    tools: list[str] = []
    for assertion in _tool_asserts(asserts):
        for tool in _as_list(assertion.get("value")):
            if tool not in tools:
                tools.append(tool)
    return tools


def _optional_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def main() -> int:
    graph = build_graph()
    cases = load_cases()
    latencies: list[float] = []
    costs: list[float] = []
    case_pass = 0
    assert_total = 0
    assert_pass = 0
    judge_runs = judge_ok = 0
    tool_runs = tool_ok = 0
    tool_soft_runs = tool_soft_ok = 0
    tool_hard_runs = tool_hard_ok = 0
    multistep_runs = multistep_ok = 0
    mutating_runs = mutating_ok = 0
    known_fail_runs = known_fail_pass = 0
    unexpected_fail = 0
    rows: list[dict[str, Any]] = []
    evaluated: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for case in cases:
        metrics.start_run()
        started = time.monotonic()
        try:
            state = run_request(graph, case["input"])  # один трейс LangFuse на кейс
        except Exception as exc:  # сетевой сбой на кейсе не должен ронять весь прогон
            print(f"  [ERR] {case['id']:24} {type(exc).__name__}: {str(exc)[:60]}")
            state = {}
        latency = time.monotonic() - started
        cost = metrics.run_cost_usd()
        calls = metrics.run_calls()
        tokens = metrics.run_tokens()
        latencies.append(latency)
        costs.append(cost)

        assertions = list(case["asserts"])
        results = [check_assert(a, state) for a in assertions]
        clarification_ok = case_must_clarify(case) and _is_clarification(state)
        if clarification_ok:
            results = [True for _ in assertions]
        assert_total += len(results)
        assert_pass += sum(results)
        ok = all(results)
        case_pass += int(ok)
        known_reason = known_fail_reason(case)
        if known_reason is not None:
            known_fail_runs += 1
            known_fail_pass += int(ok)
        elif not ok:
            unexpected_fail += 1

        judge_value: bool | None = None
        if case.get("expected_intent") in ("legal_consult", "interface_consult") and state.get("final"):
            judge_runs += 1
            judge_value = llm_judge(case["input"], state["final"])
            judge_ok += int(judge_value)

        tool_value: bool | None = None
        if case.get("expected_intent") == "mixed_diagnostic":
            tool_checks = _tool_asserts(assertions)
            got = _got_tools(state)
            if tool_checks:
                tool_runs += 1
                tool_value = all(check_assert(assertion, state) for assertion in tool_checks)
                tool_ok += int(tool_value)
            for assertion in tool_checks:
                if assertion["type"] == "expected_tools":
                    if bool(assertion.get("soft")):
                        tool_soft_runs += 1
                        tool_soft_ok += int(_check_expected_tools(assertion, got))
                    else:
                        tool_hard_runs += 1
                        tool_hard_ok += int(_check_expected_tools(assertion, got))
                    if len(_as_list(assertion.get("value"))) >= 2:
                        multistep_runs += 1
                        multistep_ok += int(_check_expected_tools(assertion, got))
                elif assertion["type"] == "used_tool":
                    tool_hard_runs += 1
                    tool_hard_ok += int(check_assert(assertion, state))
                elif assertion["type"] == "mutating_tool":
                    mutating_runs += 1
                    mutating_ok += int(check_assert(assertion, state))

        guard_in = state.get("guard_in")
        got_tools = _got_tools(state)
        actual_route = _actual_route(state)
        gold_route = _gold_route(case)
        clarify_warranted = case_clarify_warranted(case)
        evaluated.append((case, state))
        entities_ok = [
            check_assert(assertion, state)
            for assertion in assertions
            if assertion["type"] == "expected_entities"
        ]
        rows.append(
            {
                "id": case["id"],
                "expected_intent": case.get("expected_intent"),
                "intent": _intent_kind(state),
                "ok": ok,
                "blocked": guard_in is not None and guard_in.decision == "block",
                "must_clarify": case_must_clarify(case),
                "clarify_warranted": clarify_warranted,
                "clarification_ok": clarification_ok,
                "gold_route": gold_route,
                "actual_route": actual_route,
                "known_fail": known_reason,
                "composite_confidence": _frame_confidence(state),
                "clarify_code": _clarify_code(state),
                "decomposed": _decomposed(state),
                "empty_plan": _empty_plan(state),
                "latency_sec": round(latency, 3),
                "cost_usd": cost,
                "calls": calls,
                "tokens": tokens["total"],
                "expected_tools": _expected_tools_for_row(assertions),
                "got_tools": got_tools,
                "entities_ok": all(entities_ok) if entities_ok else None,
                "asserts": [
                    {"type": a["type"], "value": a.get("value"), "pass": r}
                    for a, r in zip(assertions, results, strict=True)
                ],
                "judge_ok": judge_value,
                "tool_ok": tool_value,
                "plan": _plan_capture(state),
                "debug": _debug_capture(state),
            }
        )
        print(f"  [{'OK' if ok else 'XX'}] {case['id']:24} lat={latency:5.1f}s cost=${cost:.5f}")
        if _SLEEP_SEC:
            time.sleep(_SLEEP_SEC)

    n = len(cases)
    ordered = sorted(latencies)
    p95 = ordered[min(n - 1, int(0.95 * n))] if n else 0.0
    p50 = statistics.median(latencies) if latencies else 0.0
    quality = compute_quality_metrics(evaluated)
    print("\n- EVAL -")
    print(f"  success-rate (кейсы):       {case_pass}/{n} = {case_pass / n:.0%}")
    print(f"  1) assert pass:             {assert_pass}/{assert_total}")
    print(f"  2) LLM-as-judge:            {judge_ok}/{judge_runs}")
    print(f"  3) tool-call-correctness:   {tool_ok}/{tool_runs}")
    print(f"  known-fail pass:            {known_fail_pass}/{known_fail_runs}")
    print(f"  unexpected fail:            {unexpected_fail}")
    print("- МЕТРИКИ -")
    print(f"  latency p50={p50:.1f}s  p95={p95:.1f}s")
    print(f"  cost/run avg=${statistics.mean(costs):.5f}  total=${sum(costs):.4f}")
    print(
        "  route_accuracy:            "
        f"{quality['route_ok']}/{quality['route_total']} = {quality['route_accuracy']:.0%}"
    )
    print(
        "  clarify_precision:         "
        f"{quality['clarify_warranted_hit']}/{quality['clarify_total']} = "
        f"{quality['clarify_precision']:.0%}"
    )
    print(
        "  clarify_recall:            "
        f"{quality['clarify_warranted_hit']}/{quality['clarify_warranted_total']} = "
        f"{quality['clarify_recall']:.0%}"
    )
    print(
        "  avoidable clarification:   "
        f"{quality['avoidable_clarifications']}/{n} = "
        f"{quality['avoidable_clarification_rate']:.0%}"
    )
    print(f"  empty_plan:                {quality['empty_plan']}/{n} = {quality['empty_plan_rate']:.0%}")

    out_path = os.environ.get("EVAL_OUT")
    if out_path:
        baseline_avoidable = _optional_float_env("EVAL_BASELINE_AVOIDABLE_CLARIFICATION_RATE")
        report = {
            "label": os.environ.get("EVAL_LABEL", "run"),
            "models": _models_under_test(),
            "bench_file": _BENCH.name,
            "n_cases": n,
            "summary": {
                "case_pass": case_pass,
                "success_rate": case_pass / n if n else 0.0,
                "assert_pass": assert_pass,
                "assert_total": assert_total,
                "judge_ok": judge_ok,
                "judge_runs": judge_runs,
                "tool_ok": tool_ok,
                "tool_runs": tool_runs,
                "tool_ok_soft": tool_soft_ok,
                "tool_runs_soft": tool_soft_runs,
                "tool_ok_hard": tool_hard_ok,
                "tool_runs_hard": tool_hard_runs,
                "multistep_ok": multistep_ok,
                "multistep_runs": multistep_runs,
                "mutating_ok": mutating_ok,
                "mutating_runs": mutating_runs,
                "known_fail_pass": known_fail_pass,
                "known_fail_runs": known_fail_runs,
                "unexpected_fail": unexpected_fail,
                "latency_p50_sec": round(p50, 3),
                "latency_p95_sec": round(p95, 3),
                "latency_total_sec": round(sum(latencies), 3),
                "cost_avg_usd": statistics.mean(costs) if costs else 0.0,
                "cost_total_usd": sum(costs),
                "avoidable_clarification_rate_before": baseline_avoidable,
                "avoidable_clarification_rate_after": quality["avoidable_clarification_rate"],
                **quality,
            },
            "cases": rows,
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> результаты сохранены: {out_path}")

    return 0 if unexpected_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
