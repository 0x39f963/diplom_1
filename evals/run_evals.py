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
from eva_agent.settings import settings
from eva_agent.tracing import run_request

_DEFAULT_BENCH = Path(__file__).resolve().parents[1] / "bench" / "benchmark.jsonl"
# BENCH_FILE=bench/benchmark_big.jsonl - прогнать расширенный набор (122 кейса).
_BENCH = Path(os.environ["BENCH_FILE"]) if os.environ.get("BENCH_FILE") else _DEFAULT_BENCH


def load_cases() -> list[dict]:
    lines = _BENCH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def check_assert(assertion: dict, state: dict[str, Any]) -> bool:
    kind = assertion["type"]
    intent = state.get("intent")
    guard_in = state.get("guard_in")
    final = state.get("final") or ""
    findings = state.get("api_findings") or []
    citations = state.get("citations") or []
    blocked = guard_in is not None and guard_in.decision == "block"
    if kind == "intent":
        return intent is not None and intent.kind == assertion["value"]
    if kind == "blocked":
        return blocked
    if kind == "not_blocked":
        return not blocked
    if kind == "has_citation":
        return bool(citations) or "ст." in final
    if kind == "used_tool":
        return assertion["value"] in [f.tool for f in findings]
    return False


_JUDGE_SYS = (
    "Ты - строгий оценщик ответов ИИ-помощника по рекламному праву (38-ФЗ). Верни СТРОГО JSON "
    '{"ok": true|false}: релевантен ли ОТВЕТ ВОПРОСУ и опирается ли на нормы/данные (а не вода).'
)


def llm_judge(question: str, answer: str) -> bool:
    response = get_client("guard").invoke(
        _JUDGE_SYS, f"ВОПРОС: {question}\n\nОТВЕТ: {answer}", temperature=0.0, json_mode=True
    )
    try:
        return bool(json.loads(response.text).get("ok"))
    except (json.JSONDecodeError, ValueError):
        return False


def _models_under_test() -> dict[str, str]:
    """Снимок «какая модель на какой роли» - для шапки отчета."""
    return {
        role: f"{settings.role_backend(role)}:{settings.role_model(role)}"  # type: ignore[arg-type]
        for role in ("reasoning", "default", "guard")
    }


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
    rows: list[dict[str, Any]] = []

    for case in cases:
        metrics.start_run()
        started = time.monotonic()
        state = run_request(graph, case["input"])  # один трейс LangFuse на кейс
        latency = time.monotonic() - started
        cost = metrics.run_cost_usd()
        calls = metrics.run_calls()
        tokens = metrics.run_tokens()
        latencies.append(latency)
        costs.append(cost)

        results = [check_assert(a, state) for a in case["asserts"]]
        assert_total += len(results)
        assert_pass += sum(results)
        ok = all(results)
        case_pass += int(ok)

        judge_value: bool | None = None
        if case.get("expected_intent") in ("legal_consult", "interface_consult") and state.get("final"):
            judge_runs += 1
            judge_value = llm_judge(case["input"], state["final"])
            judge_ok += int(judge_value)

        tool_value: bool | None = None
        if case.get("expected_intent") == "mixed_diagnostic":
            tool_runs += 1
            want = [a["value"] for a in case["asserts"] if a["type"] == "used_tool"]
            got = [f.tool for f in (state.get("api_findings") or [])]
            tool_value = all(w in got for w in want)
            tool_ok += int(tool_value)

        intent = state.get("intent")
        guard_in = state.get("guard_in")
        rows.append(
            {
                "id": case["id"],
                "expected_intent": case.get("expected_intent"),
                "intent": intent.kind if intent is not None else None,
                "ok": ok,
                "blocked": guard_in is not None and guard_in.decision == "block",
                "latency_sec": round(latency, 3),
                "cost_usd": cost,
                "calls": calls,
                "tokens": tokens["total"],
                "asserts": [
                    {"type": a["type"], "value": a.get("value"), "pass": r}
                    for a, r in zip(case["asserts"], results, strict=True)
                ],
                "judge_ok": judge_value,
                "tool_ok": tool_value,
            }
        )
        print(f"  [{'OK' if ok else 'XX'}] {case['id']:24} lat={latency:5.1f}s cost=${cost:.5f}")

    n = len(cases)
    ordered = sorted(latencies)
    p95 = ordered[min(n - 1, int(0.95 * n))] if n else 0.0
    p50 = statistics.median(latencies) if latencies else 0.0
    print("\n- EVAL -")
    print(f"  success-rate (кейсы):       {case_pass}/{n} = {case_pass / n:.0%}")
    print(f"  1) assert pass:             {assert_pass}/{assert_total}")
    print(f"  2) LLM-as-judge:            {judge_ok}/{judge_runs}")
    print(f"  3) tool-call-correctness:   {tool_ok}/{tool_runs}")
    print("- МЕТРИКИ -")
    print(f"  latency p50={p50:.1f}s  p95={p95:.1f}s")
    print(f"  cost/run avg=${statistics.mean(costs):.5f}  total=${sum(costs):.4f}")

    out_path = os.environ.get("EVAL_OUT")
    if out_path:
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
                "latency_p50_sec": round(p50, 3),
                "latency_p95_sec": round(p95, 3),
                "latency_total_sec": round(sum(latencies), 3),
                "cost_avg_usd": statistics.mean(costs) if costs else 0.0,
                "cost_total_usd": sum(costs),
            },
            "cases": rows,
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> результаты сохранены: {out_path}")

    return 0 if case_pass == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
