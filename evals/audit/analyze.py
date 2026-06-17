"""Build aggregate reports for benchmark run JSON files."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import NamedTuple

REPORT_MD = "round2-analysis.md"
REPORT_HTML = "round2-report.html"
CHARTS_JSON = "round2-charts.json"
SLOW_LATENCY_SEC = 60.0

CATEGORIES: tuple[str, ...] = ("blocked", "legal_consult", "mixed_diagnostic")
CATEGORY_LABELS: dict[str, str] = {
    "blocked": "Блокировка",
    "legal_consult": "Юридические",
    "mixed_diagnostic": "Mixed",
}
MODEL_NAMES: dict[str, str] = {
    "gemini-3.1-flash": "Gemini 3.1 Flash",
    "llama-3.3-70b": "Llama 3.3 70B",
    "qwen-local": "Qwen local",
    "claude-sonnet-high": "Claude Sonnet high",
    "codex-gpt55-medium": "Codex GPT-5.5 medium",
}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    expected_intent: str
    intent: str | None
    ok: bool
    blocked: bool
    must_clarify: bool
    latency_sec: float
    cost_usd: float
    calls: int
    tokens: int
    expected_tools: tuple[str, ...]
    got_tools: tuple[str, ...]
    judge_ok: bool | None
    tool_ok: bool | None


@dataclass(frozen=True)
class RunReport:
    label: str
    models: dict[str, str]
    bench_file: str
    n_cases: int
    summary: dict[str, object]
    cases: tuple[CaseResult, ...]


@dataclass(frozen=True)
class CountRate:
    ok: int
    total: int

    @property
    def rate(self) -> float | None:
        if self.total == 0:
            return None
        return self.ok / self.total


@dataclass(frozen=True)
class ModelMetrics:
    label: str
    display_name: str
    backend_model: str
    bench_file: str
    n_cases: int
    success: CountRate
    categories: dict[str, CountRate]
    tool: CountRate
    multistep: CountRate
    latency_p50_sec: float
    latency_p95_sec: float
    latency_total_sec: float
    cost_total_usd: float
    cost_avg_usd: float
    calls_avg: float
    tokens_avg: float
    slow_cases: tuple[CaseResult, ...]
    mixed_clarification: CountRate


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    input_text: str
    expected_intent: str


@dataclass(frozen=True)
class NotableCase:
    case_id: str
    reason: str


class CaseByModel(NamedTuple):
    label: str
    case: CaseResult


def _as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_bool(value: object, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _as_bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(key, str):
            result[key] = item
    return result


def _case_from_raw(raw: object) -> CaseResult:
    data = _as_dict(raw)
    return CaseResult(
        case_id=_as_str(data.get("id")),
        expected_intent=_as_str(data.get("expected_intent")),
        intent=_as_str(data.get("intent")) if data.get("intent") is not None else None,
        ok=_as_bool(data.get("ok")),
        blocked=_as_bool(data.get("blocked")),
        must_clarify=_as_bool(data.get("must_clarify")),
        latency_sec=_as_float(data.get("latency_sec")),
        cost_usd=_as_float(data.get("cost_usd")),
        calls=_as_int(data.get("calls")),
        tokens=_as_int(data.get("tokens")),
        expected_tools=_as_str_tuple(data.get("expected_tools")),
        got_tools=_as_str_tuple(data.get("got_tools")),
        judge_ok=_as_bool_or_none(data.get("judge_ok")),
        tool_ok=_as_bool_or_none(data.get("tool_ok")),
    )


def _load_run(path: Path) -> RunReport:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: root value must be an object")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list):
        raise ValueError(f"{path}: cases must be a list")
    cases = tuple(_case_from_raw(case) for case in cases_raw)
    return RunReport(
        label=_as_str(raw.get("label"), path.stem),
        models={key: _as_str(value) for key, value in _as_dict(raw.get("models")).items()},
        bench_file=_as_str(raw.get("bench_file")),
        n_cases=_as_int(raw.get("n_cases"), len(cases)),
        summary=_as_dict(raw.get("summary")),
        cases=cases,
    )


def load_runs(audit_dir: Path) -> list[RunReport]:
    paths = sorted(path for path in audit_dir.glob("*.json") if path.name != CHARTS_JSON)
    runs = [_load_run(path) for path in paths]
    return sorted(runs, key=lambda run: run.label)


def _summary_count(summary: dict[str, object], key: str) -> int:
    return _as_int(summary.get(key))


def _summary_float(summary: dict[str, object], key: str) -> float:
    return _as_float(summary.get(key))


def _count_success(cases: Sequence[CaseResult]) -> CountRate:
    return CountRate(ok=sum(1 for case in cases if case.ok), total=len(cases))


def _count_bool(values: Sequence[bool | None]) -> CountRate:
    known = [value for value in values if value is not None]
    return CountRate(ok=sum(1 for value in known if value), total=len(known))


def _count_mixed_clarification(cases: Sequence[CaseResult]) -> CountRate:
    mixed = [case for case in cases if case.expected_intent == "mixed_diagnostic"]
    return CountRate(
        ok=sum(1 for case in mixed if case.intent == "need_clarification"),
        total=len(mixed),
    )


def _model_backend(models: dict[str, str]) -> str:
    values = sorted(set(models.values()))
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values)


def _display_name(label: str) -> str:
    return MODEL_NAMES.get(label, label)


def _mean_int(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _multistep_from_cases(cases: Sequence[CaseResult]) -> CountRate:
    rows = [
        case
        for case in cases
        if case.expected_intent == "mixed_diagnostic" and len(case.expected_tools) >= 2
    ]
    return CountRate(
        ok=sum(1 for case in rows if set(case.expected_tools) <= set(case.got_tools)),
        total=len(rows),
    )


def build_model_metrics(runs: Sequence[RunReport]) -> list[ModelMetrics]:
    metrics: list[ModelMetrics] = []
    for run in runs:
        summary = run.summary
        categories = {
            category: _count_success(
                [case for case in run.cases if case.expected_intent == category]
            )
            for category in CATEGORIES
        }
        tool = CountRate(
            ok=_summary_count(summary, "tool_ok"),
            total=_summary_count(summary, "tool_runs"),
        )
        if tool.total == 0:
            tool = _count_bool(
                [case.tool_ok for case in run.cases if case.expected_intent == "mixed_diagnostic"]
            )
        multistep = CountRate(
            ok=_summary_count(summary, "multistep_ok"),
            total=_summary_count(summary, "multistep_runs"),
        )
        if multistep.total == 0:
            multistep = _multistep_from_cases(run.cases)
        slow_cases = tuple(
            sorted(
                (case for case in run.cases if case.latency_sec > SLOW_LATENCY_SEC),
                key=lambda case: case.latency_sec,
                reverse=True,
            )
        )
        metrics.append(
            ModelMetrics(
                label=run.label,
                display_name=_display_name(run.label),
                backend_model=_model_backend(run.models),
                bench_file=run.bench_file,
                n_cases=run.n_cases,
                success=CountRate(
                    ok=_summary_count(summary, "case_pass") or sum(1 for case in run.cases if case.ok),
                    total=run.n_cases,
                ),
                categories=categories,
                tool=tool,
                multistep=multistep,
                latency_p50_sec=_summary_float(summary, "latency_p50_sec"),
                latency_p95_sec=_summary_float(summary, "latency_p95_sec"),
                latency_total_sec=_summary_float(summary, "latency_total_sec"),
                cost_total_usd=_summary_float(summary, "cost_total_usd"),
                cost_avg_usd=_summary_float(summary, "cost_avg_usd"),
                calls_avg=_mean_int([case.calls for case in run.cases]),
                tokens_avg=_mean_int([case.tokens for case in run.cases]),
                slow_cases=slow_cases,
                mixed_clarification=_count_mixed_clarification(run.cases),
            )
        )
    return metrics


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_benchmark_cases(runs: Sequence[RunReport]) -> dict[str, BenchmarkCase]:
    result: dict[str, BenchmarkCase] = {}
    bench_dir = _repo_root() / "bench"
    for bench_file in sorted({run.bench_file for run in runs if run.bench_file}):
        path = bench_dir / bench_file
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            case_id = _as_str(raw.get("id"))
            if not case_id:
                continue
            result[case_id] = BenchmarkCase(
                case_id=case_id,
                input_text=_as_str(raw.get("input")),
                expected_intent=_as_str(raw.get("expected_intent")),
            )
    return result


def _cases_by_id(runs: Sequence[RunReport]) -> dict[str, list[CaseByModel]]:
    grouped: dict[str, list[CaseByModel]] = defaultdict(list)
    for run in runs:
        for case in run.cases:
            grouped[case.case_id].append(CaseByModel(run.label, case))
    return dict(grouped)


def _all_model_failures(
    grouped: dict[str, list[CaseByModel]],
    model_count: int,
) -> list[str]:
    return sorted(
        case_id
        for case_id, rows in grouped.items()
        if len(rows) == model_count and all(not row.case.ok for row in rows)
    )


def _divergent_cases(grouped: dict[str, list[CaseByModel]]) -> list[str]:
    result: list[str] = []
    for case_id, rows in grouped.items():
        statuses = {row.case.ok for row in rows}
        if len(rows) >= 2 and len(statuses) > 1:
            result.append(case_id)
    return sorted(result)


def _mixed_tool_counters(runs: Sequence[RunReport]) -> tuple[Counter[str], Counter[str]]:
    missing: Counter[str] = Counter()
    extra: Counter[str] = Counter()
    for run in runs:
        for case in run.cases:
            if case.expected_intent != "mixed_diagnostic" or case.tool_ok is not False:
                continue
            expected = set(case.expected_tools)
            got = set(case.got_tools)
            missing.update(sorted(expected - got))
            extra.update(sorted(got - expected))
    return missing, extra


def _case_expected_intent(rows: Sequence[CaseByModel]) -> str:
    for row in rows:
        if row.case.expected_intent:
            return row.case.expected_intent
    return ""


def _top_clarification_cases(grouped: dict[str, list[CaseByModel]]) -> list[str]:
    scored: list[tuple[int, int, float, str]] = []
    for case_id, rows in grouped.items():
        if _case_expected_intent(rows) != "mixed_diagnostic":
            continue
        clarifications = sum(1 for row in rows if row.case.intent == "need_clarification")
        if clarifications == 0:
            continue
        max_tools = max((len(row.case.expected_tools) for row in rows), default=0)
        max_latency = max((row.case.latency_sec for row in rows), default=0.0)
        scored.append((clarifications, max_tools, max_latency, case_id))
    return [case_id for *_score, case_id in sorted(scored, reverse=True)]


def _qwen_slow_cases(runs: Sequence[RunReport]) -> list[str]:
    qwen = next((run for run in runs if run.label == "qwen-local"), None)
    if qwen is None:
        return []
    slow = sorted(qwen.cases, key=lambda case: case.latency_sec, reverse=True)
    return [
        case.case_id
        for case in slow
        if case.latency_sec > SLOW_LATENCY_SEC and case.expected_intent == "mixed_diagnostic"
    ]


def select_notable_cases(
    runs: Sequence[RunReport],
    grouped: dict[str, list[CaseByModel]],
) -> list[NotableCase]:
    selected: list[NotableCase] = []
    selected_ids: set[str] = set()

    def add(case_id: str, reason: str) -> None:
        if case_id in selected_ids or case_id not in grouped or len(selected) >= 8:
            return
        selected.append(NotableCase(case_id=case_id, reason=reason))
        selected_ids.add(case_id)

    for case_id in _all_model_failures(grouped, len(runs))[:3]:
        add(case_id, "падает у всех 5 моделей")
    for case_id in _top_clarification_cases(grouped)[:3]:
        add(case_id, "уход в уточнение вместо вызова инструмента")
    for case_id in _qwen_slow_cases(runs)[:3]:
        add(case_id, "зацикливание Qwen: высокая задержка и много вызовов")
    for case_id in _divergent_cases(grouped):
        add(case_id, "модели расходятся по результату")
        if len(selected) >= 8:
            break
    return selected


def _rate_value(rate: CountRate) -> float:
    return rate.rate if rate.rate is not None else 0.0


def _pct(rate: float | None) -> str:
    if rate is None:
        return "n/a"
    return f"{rate * 100:.1f}%"


def _count_rate(rate: CountRate) -> str:
    return f"{rate.ok}/{rate.total} ({_pct(rate.rate)})"


def _money(value: float) -> str:
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.5f}"
    return f"${value:.4f}"


def _num(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def _short(text: str, limit: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _run_date(audit_dir: Path, runs: Sequence[RunReport]) -> str:
    parts = audit_dir.resolve().parts
    for index in range(len(parts) - 2):
        year, month, day = parts[index : index + 3]
        if (
            len(year) == 4
            and len(month) == 2
            and len(day) == 2
            and year.isdigit()
            and month.isdigit()
            and day.isdigit()
        ):
            return f"{day}.{month}.{year}"
    timestamps = [
        path.stat().st_mtime
        for path in audit_dir.glob("*.json")
        if path.name != CHARTS_JSON and path.is_file()
    ]
    if timestamps:
        return datetime.fromtimestamp(max(timestamps)).strftime("%d.%m.%Y")
    if runs:
        return "дата не определена"
    return "нет данных"


def _best_by_success(metrics: Sequence[ModelMetrics]) -> ModelMetrics | None:
    if not metrics:
        return None
    return max(metrics, key=lambda item: _rate_value(item.success))


def _best_full_run(metrics: Sequence[ModelMetrics]) -> ModelMetrics | None:
    full = [item for item in metrics if item.n_cases == max(metric.n_cases for metric in metrics)]
    if not full:
        return None
    return max(full, key=lambda item: _rate_value(item.success))


def _fastest(metrics: Sequence[ModelMetrics]) -> ModelMetrics | None:
    if not metrics:
        return None
    return min(metrics, key=lambda item: item.latency_p50_sec)


def _best_free(metrics: Sequence[ModelMetrics]) -> ModelMetrics | None:
    free = [item for item in metrics if item.cost_total_usd == 0]
    if not free:
        return None
    return max(free, key=lambda item: _rate_value(item.success))


def chart_data(
    metrics: Sequence[ModelMetrics],
    runs: Sequence[RunReport],
    grouped: dict[str, list[CaseByModel]],
    notable: Sequence[NotableCase],
) -> dict[str, object]:
    missing, extra = _mixed_tool_counters(runs)
    return {
        "models": [
            {
                "label": item.label,
                "display_name": item.display_name,
                "backend_model": item.backend_model,
                "bench_file": item.bench_file,
                "n_cases": item.n_cases,
                "success_rate": _rate_value(item.success),
                "success": {"ok": item.success.ok, "total": item.success.total},
                "categories": {
                    category: {
                        "ok": rate.ok,
                        "total": rate.total,
                        "rate": _rate_value(rate),
                    }
                    for category, rate in item.categories.items()
                },
                "tool_success_rate": _rate_value(item.tool),
                "tool": {"ok": item.tool.ok, "total": item.tool.total},
                "multistep_success_rate": _rate_value(item.multistep),
                "multistep": {"ok": item.multistep.ok, "total": item.multistep.total},
                "mixed_clarification_rate": _rate_value(item.mixed_clarification),
                "latency_p50_sec": item.latency_p50_sec,
                "latency_p95_sec": item.latency_p95_sec,
                "cost_total_usd": item.cost_total_usd,
                "cost_avg_usd": item.cost_avg_usd,
                "calls_avg": item.calls_avg,
                "tokens_avg": item.tokens_avg,
                "slow_cases": [
                    {
                        "id": case.case_id,
                        "latency_sec": case.latency_sec,
                        "calls": case.calls,
                    }
                    for case in item.slow_cases
                ],
            }
            for item in metrics
        ],
        "cross_model": {
            "all_failed": _all_model_failures(grouped, len(metrics)),
            "divergent": _divergent_cases(grouped),
            "notable": [
                {"id": item.case_id, "reason": item.reason}
                for item in notable
            ],
            "mixed_missing_tools": missing.most_common(),
            "mixed_extra_tools": extra.most_common(),
        },
    }


def _line_points(max_value: float, count: int) -> list[tuple[float, str]]:
    if max_value <= 1.0:
        return [(0.0, "0%"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "100%")]
    steps = 4
    return [(max_value * index / steps, f"{max_value * index / steps:.0f}") for index in range(steps + 1)]


def _svg_text(x: float, y: float, text: str, css_class: str = "") -> str:
    cls = f' class="{css_class}"' if css_class else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{cls}>{escape(text)}</text>'


def success_svg(metrics: Sequence[ModelMetrics]) -> str:
    width = 1040
    height = 380
    left = 70
    right = 24
    top = 34
    bottom = 92
    inner_w = width - left - right
    inner_h = height - top - bottom
    colors = {
        "overall": "#2563eb",
        "blocked": "#059669",
        "legal_consult": "#d97706",
        "mixed_diagnostic": "#dc2626",
    }
    series = (
        ("overall", "Общий"),
        ("blocked", "Блокировка"),
        ("legal_consult", "Юридические"),
        ("mixed_diagnostic", "Mixed"),
    )
    group_w = inner_w / max(len(metrics), 1)
    bar_w = min(28.0, group_w / 6)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Success-rate">',
        "<style>.axis{fill:#475569;font:12px Arial}.label{fill:#111827;font:12px Arial}"
        ".value{fill:#111827;font:11px Arial;font-weight:700}.grid{stroke:#e2e8f0}"
        ".legend{fill:#334155;font:12px Arial}</style>",
    ]
    for value, label in _line_points(1.0, 5):
        y = top + inner_h - value * inner_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
        parts.append(_svg_text(18, y + 4, label, "axis"))
    for index, (key, label) in enumerate(series):
        legend_x = left + index * 160
        color = colors[key]
        parts.append(f'<rect x="{legend_x}" y="8" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(_svg_text(legend_x + 18, 19, label, "legend"))
    for model_index, item in enumerate(metrics):
        start_x = left + model_index * group_w + (group_w - bar_w * len(series)) / 2
        values = {
            "overall": _rate_value(item.success),
            **{category: _rate_value(rate) for category, rate in item.categories.items()},
        }
        for bar_index, (key, _label) in enumerate(series):
            value = values[key]
            bar_h = value * inner_h
            bar_x = start_x + bar_index * bar_w
            bar_y = top + inner_h - bar_h
            parts.append(
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w - 3:.1f}" '
                f'height="{bar_h:.1f}" rx="3" fill="{colors[key]}"/>'
            )
            if value >= 0.18:
                parts.append(_svg_text(bar_x - 2, bar_y - 5, f"{value * 100:.0f}", "value"))
        label = item.display_name.replace(" ", "\n")
        label_lines = label.split("\n")
        for line_index, line in enumerate(label_lines[:3]):
            parts.append(
                _svg_text(
                    left + model_index * group_w + group_w / 2 - 48,
                    top + inner_h + 24 + line_index * 14,
                    line,
                    "label",
                )
            )
    parts.append("</svg>")
    return "".join(parts)


def grouped_metric_svg(
    metrics: Sequence[ModelMetrics],
    values: Sequence[tuple[str, str, str]],
    max_value: float,
    aria: str,
) -> str:
    width = 1040
    height = 340
    left = 70
    right = 26
    top = 42
    bottom = 86
    inner_w = width - left - right
    inner_h = height - top - bottom
    colors = ("#2563eb", "#0f766e", "#d97706", "#dc2626")
    group_w = inner_w / max(len(metrics), 1)
    bar_w = min(34.0, group_w / (len(values) + 2))
    safe_max = max(max_value, 1.0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(aria)}">',
        "<style>.axis{fill:#475569;font:12px Arial}.label{fill:#111827;font:12px Arial}"
        ".value{fill:#111827;font:11px Arial;font-weight:700}.grid{stroke:#e2e8f0}"
        ".legend{fill:#334155;font:12px Arial}</style>",
    ]
    for value, label in _line_points(safe_max, 5):
        y = top + inner_h - value / safe_max * inner_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
        parts.append(_svg_text(16, y + 4, label, "axis"))
    for index, (_key, label, _kind) in enumerate(values):
        legend_x = left + index * 190
        parts.append(
            f'<rect x="{legend_x}" y="13" width="12" height="12" '
            f'rx="2" fill="{colors[index]}"/>'
        )
        parts.append(_svg_text(legend_x + 18, 24, label, "legend"))
    for model_index, item in enumerate(metrics):
        start_x = left + model_index * group_w + (group_w - bar_w * len(values)) / 2
        for bar_index, (key, _label, kind) in enumerate(values):
            raw_value = _metric_value(item, key)
            value = raw_value * 100.0 if kind == "rate" else raw_value
            bar_h = min(value / safe_max, 1.0) * inner_h
            bar_x = start_x + bar_index * bar_w
            bar_y = top + inner_h - bar_h
            parts.append(
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w - 4:.1f}" '
                f'height="{bar_h:.1f}" rx="3" fill="{colors[bar_index]}"/>'
            )
            if value > safe_max * 0.12:
                shown = f"{value:.0f}" if kind != "money" else f"{value:.2f}"
                parts.append(_svg_text(bar_x - 2, bar_y - 5, shown, "value"))
        label_lines = item.display_name.split(" ")
        for line_index, line in enumerate(label_lines[:3]):
            parts.append(
                _svg_text(
                    left + model_index * group_w + group_w / 2 - 48,
                    top + inner_h + 24 + line_index * 14,
                    line,
                    "label",
                )
            )
    parts.append("</svg>")
    return "".join(parts)


def _metric_value(item: ModelMetrics, key: str) -> float:
    if key == "tool":
        return _rate_value(item.tool)
    if key == "multistep":
        return _rate_value(item.multistep)
    if key == "clarification":
        return _rate_value(item.mixed_clarification)
    if key == "latency_p50":
        return item.latency_p50_sec
    if key == "latency_p95":
        return item.latency_p95_sec
    if key == "cost_total":
        return item.cost_total_usd
    return 0.0


def markdown_report(
    audit_dir: Path,
    runs: Sequence[RunReport],
    metrics: Sequence[ModelMetrics],
    grouped: dict[str, list[CaseByModel]],
    notable: Sequence[NotableCase],
    bench_cases: dict[str, BenchmarkCase],
) -> str:
    run_date = _run_date(audit_dir, runs)
    all_failed = _all_model_failures(grouped, len(runs))
    divergent = _divergent_cases(grouped)
    missing, extra = _mixed_tool_counters(runs)
    best = _best_by_success(metrics)
    best_full = _best_full_run(metrics)
    fastest = _fastest(metrics)
    free = _best_free(metrics)

    lines = [
        "# Round 2 - анализ прогона 5 моделей",
        "",
        f"Дата прогона: {run_date}. Моделей: {len(runs)}. "
        f"Полный набор: 180 кейсов. CLI-срез: 45 кейсов.",
        "",
        "Сравнение нормализовано по долям: у CLI-моделей меньше кейсов, поэтому в выводах "
        "сравниваются проценты, а не абсолютные счетчики.",
        "",
        "## Сводка по моделям",
        "",
        "| Модель | Набор | Success | Blocked | Legal | Mixed | Tool mixed | Multistep | p50/p95 | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        lines.append(
            "| "
            f"{item.display_name} | {item.n_cases} | {_count_rate(item.success)} | "
            f"{_count_rate(item.categories['blocked'])} | "
            f"{_count_rate(item.categories['legal_consult'])} | "
            f"{_count_rate(item.categories['mixed_diagnostic'])} | "
            f"{_count_rate(item.tool)} | {_count_rate(item.multistep)} | "
            f"{_num(item.latency_p50_sec)}/{_num(item.latency_p95_sec)}s | "
            f"{_money(item.cost_total_usd)} |"
        )
    lines.extend(
        [
            "",
            "## Главные находки",
            "",
        ]
    )
    if best is not None:
        lines.append(
            f"- Лучший общий результат: {best.display_name} - "
            f"{_pct(best.success.rate)} на {best.n_cases} кейсах."
        )
    if best_full is not None:
        lines.append(
            f"- Лучший планировщик на полном наборе: {best_full.display_name} - "
            f"{_pct(best_full.success.rate)}, mixed {_pct(best_full.categories['mixed_diagnostic'].rate)}, "
            f"tool {_pct(best_full.tool.rate)}."
        )
    if fastest is not None:
        lines.append(
            f"- Самая быстрая медиана: {fastest.display_name} - "
            f"p50 {_num(fastest.latency_p50_sec)}s."
        )
    if free is not None:
        lines.append(
            f"- Лучший нулевой API-cost: {free.display_name} - "
            f"{_pct(free.success.rate)}, но p95 {_num(free.latency_p95_sec)}s."
        )
    lines.extend(
        [
            f"- Кейсов, проваленных всеми 5 моделями: {len(all_failed)}.",
            f"- Кейсов с расхождением между моделями: {len(divergent)}.",
            "- Основная зона риска - mixed: инструментальная цепочка и склонность уходить в уточнение.",
            "",
            "## Mixed: типичные провалы инструментов",
            "",
            f"- Чаще всего не вызывается: {_counter_line(missing)}.",
            f"- Лишние инструменты встречаются реже: {_counter_line(extra)}.",
        ]
    )
    for item in metrics:
        lines.append(
            f"- {item.display_name}: уточнение на mixed "
            f"{_count_rate(item.mixed_clarification)}, tool {_count_rate(item.tool)}, "
            f"multistep {_count_rate(item.multistep)}."
        )
    lines.extend(
        [
            "",
            "## Зацикливание и задержки",
            "",
        ]
    )
    for item in metrics:
        if not item.slow_cases:
            lines.append(f"- {item.display_name}: выбросов > {SLOW_LATENCY_SEC:.0f}s нет.")
            continue
        examples = ", ".join(
            f"{case.case_id} ({_num(case.latency_sec)}s, calls {case.calls})"
            for case in item.slow_cases[:4]
        )
        lines.append(
            f"- {item.display_name}: {len(item.slow_cases)} выбросов > "
            f"{SLOW_LATENCY_SEC:.0f}s, примеры: {examples}."
        )
    lines.extend(
        [
            "",
            "## Показательные кейсы",
            "",
        ]
    )
    for notable_case in notable:
        rows = sorted(grouped[notable_case.case_id], key=lambda row: row.label)
        bench = bench_cases.get(notable_case.case_id)
        lines.append(f"### {notable_case.case_id}")
        lines.append("")
        lines.append(f"Причина: {notable_case.reason}.")
        if bench is not None and bench.input_text:
            lines.append(f"Запрос: {_short(bench.input_text)}")
        lines.append("")
        lines.append("| Модель | ok | intent | latency | calls | expected_tools | got_tools |")
        lines.append("|---|---:|---|---:|---:|---|---|")
        for row in rows:
            case = row.case
            lines.append(
                "| "
                f"{_display_name(row.label)} | {'yes' if case.ok else 'no'} | "
                f"{case.intent or 'none'} | {_num(case.latency_sec)}s | {case.calls} | "
                f"{', '.join(case.expected_tools) or '-'} | "
                f"{', '.join(case.got_tools) or '-'} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Выводы",
            "",
            "- Gemini выглядит лучшим планировщиком в этом прогоне: лучший общий процент, "
            "лучший mixed и нормальная скорость.",
            "- Qwen local полезен как бесплатный baseline и второй по полному набору, "
            "но требует контроля лимитов: много вызовов и 23 выброса дольше 60s.",
            "- Llama 70B слабо выбирает инструменты на mixed и часто теряет юридические кейсы.",
            "- Claude Sonnet high и Codex GPT-5.5 medium на CLI-срезе хорошо держат blocked, "
            "но часто переуточняют mixed вместо выполнения цепочки инструментов.",
            "- Системные дыры планировщика видны в all-fail кейсах: сложные цепочки документов, "
            "placements, creative status и attach/download сценарии.",
            "",
        ]
    )
    return "\n".join(lines)


def _counter_line(counter: Counter[str], limit: int = 8) -> str:
    if not counter:
        return "нет данных"
    return ", ".join(f"{name} - {count}" for name, count in counter.most_common(limit))


def html_report(
    audit_dir: Path,
    runs: Sequence[RunReport],
    metrics: Sequence[ModelMetrics],
    grouped: dict[str, list[CaseByModel]],
    notable: Sequence[NotableCase],
    bench_cases: dict[str, BenchmarkCase],
) -> str:
    run_date = _run_date(audit_dir, runs)
    all_failed = _all_model_failures(grouped, len(runs))
    divergent = _divergent_cases(grouped)
    missing, extra = _mixed_tool_counters(runs)
    best = _best_by_success(metrics)
    best_full = _best_full_run(metrics)
    fastest = _fastest(metrics)
    free = _best_free(metrics)
    max_latency = max((item.latency_p95_sec for item in metrics), default=1.0)
    max_cost = max((item.cost_total_usd for item in metrics), default=1.0)

    summary_rows = "\n".join(_summary_html_row(item) for item in metrics)
    legend_rows = "\n".join(
        f"<tr><td>{escape(item.display_name)}</td><td><code>{escape(item.backend_model)}</code></td>"
        f"<td>{escape(item.bench_file)}</td><td class=\"num\">{item.n_cases}</td></tr>"
        for item in metrics
    )
    notable_html = "\n".join(
        _notable_case_html(item, grouped[item.case_id], bench_cases.get(item.case_id))
        for item in notable
    )
    findings = _findings_html(best, best_full, fastest, free, all_failed, divergent)
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Round 2 - отчет по 5 моделям</title>
<style>
:root {{
  --bg:#f6f8fb; --panel:#ffffff; --line:#d9e2ef; --text:#172033; --muted:#64748b;
  --blue:#2563eb; --green:#059669; --amber:#d97706; --red:#dc2626; --violet:#7c3aed;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 Arial, sans-serif}}
.wrap{{max-width:1160px;margin:0 auto;padding:30px 22px 70px}}
h1{{font-size:30px;line-height:1.15;margin:0 0 8px}}
h2{{font-size:21px;margin:34px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}}
h3{{font-size:17px;margin:20px 0 8px}}
p{{margin:8px 0}}
.sub{{color:var(--muted)}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;margin:14px 0}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}}
.stat{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px}}
.stat b{{display:block;font-size:22px;margin-bottom:3px}}
.stat span{{color:var(--muted);font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:14px;background:#fff}}
th,td{{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}
th{{background:#eef3f9;color:#334155;font-size:12px;font-weight:700}}
td.num{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}
code{{background:#eef3f9;padding:1px 5px;border-radius:4px;font-size:12px}}
.chart{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px;margin:12px 0;overflow-x:auto}}
.chart svg{{display:block;width:100%;min-width:820px;height:auto}}
.case{{background:#fff;border:1px solid var(--line);border-radius:8px;margin:12px 0;padding:14px}}
.case-title{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}}
.case-title b{{font-size:16px}}
.reason{{color:#0f766e;font-weight:700}}
.small{{font-size:13px;color:var(--muted)}}
.pill{{display:inline-block;border-radius:999px;padding:2px 8px;font-size:12px;font-weight:700}}
.ok{{background:#dcfce7;color:#166534}}
.bad{{background:#fee2e2;color:#991b1b}}
.findings{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
.finding{{border-left:4px solid var(--blue);background:#fff;padding:12px 14px;border-radius:8px;border-top:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}}
@media (max-width: 760px) {{
  .grid,.findings{{grid-template-columns:1fr}}
  .wrap{{padding:22px 14px 50px}}
  h1{{font-size:25px}}
}}
</style>
</head>
<body>
<div class="wrap">
<h1>Round 2 - аналитика прогона 5 моделей</h1>
<p class="sub">Дата: {escape(run_date)}. Прогон: 5 моделей, полный набор 180 кейсов, CLI-срез 45 кейсов. Сравнение нормализовано по долям.</p>

<div class="grid">
  <div class="stat"><b>{escape(_pct(best.success.rate) if best else "n/a")}</b><span>лучший общий success-rate</span></div>
  <div class="stat"><b>{escape(_pct(best_full.tool.rate) if best_full else "n/a")}</b><span>лучший tool-call на полном наборе</span></div>
  <div class="stat"><b>{len(all_failed)}</b><span>кейсов провалили все 5 моделей</span></div>
  <div class="stat"><b>{len(divergent)}</b><span>кейсов с расхождением моделей</span></div>
</div>

<h2>Легенда моделей</h2>
<div class="panel">
<table>
<tr><th>Модель</th><th>backend:model</th><th>bench</th><th>кейсов</th></tr>
{legend_rows}
</table>
</div>

<h2>Success-rate</h2>
<p class="sub">Общий результат и разбивка по категориям: blocked, legal_consult, mixed_diagnostic.</p>
<div class="chart">{success_svg(metrics)}</div>

<h2>Инструменты, скорость и цена</h2>
<div class="chart">{grouped_metric_svg(metrics, (("tool", "Tool mixed, %", "rate"), ("multistep", "Multistep, %", "rate"), ("clarification", "Уточнение mixed, %", "rate")), 100.0, "Tool success")}</div>
<div class="chart">{grouped_metric_svg(metrics, (("latency_p50", "Latency p50, s", "number"), ("latency_p95", "Latency p95, s", "number")), max_latency, "Latency")}</div>
<div class="chart">{grouped_metric_svg(metrics, (("cost_total", "Cost total, USD", "money"),), max_cost, "Cost")}</div>

<h2>Сводная таблица</h2>
<table>
<tr><th>Модель</th><th>Success</th><th>Blocked</th><th>Legal</th><th>Mixed</th><th>Tool mixed</th><th>Multistep</th><th>p50/p95</th><th>Cost</th><th>calls/tokens</th><th>slow</th></tr>
{summary_rows}
</table>

<h2>Разбор</h2>
<div class="findings">
{findings}
<div class="finding"><b>Missing tools</b><br>{escape(_counter_line(missing))}</div>
<div class="finding"><b>Extra tools</b><br>{escape(_counter_line(extra))}</div>
</div>
{notable_html}

<h2>Выводы</h2>
<div class="panel">
<p>Лучший планировщик в этом прогоне - Gemini 3.1 Flash: лучший общий процент, лучший mixed и низкая медианная задержка.</p>
<p>Qwen local - сильный бесплатный baseline на полном наборе, но именно он дает главный риск зацикливания: 23 кейса дольше 60 секунд и высокая средняя доля вызовов.</p>
<p>Системные провалы связаны не с одним провайдером, а с логикой mixed-планирования: цепочки документов, placements, creative status и сценарии attach/download часто ломаются у всех.</p>
<p>CLI-модели на срезе уверенно держат blocked, но слишком часто уходят в уточнение вместо выполнения доступной инструментальной цепочки.</p>
</div>
</div>
</body>
</html>
"""


def _findings_html(
    best: ModelMetrics | None,
    best_full: ModelMetrics | None,
    fastest: ModelMetrics | None,
    free: ModelMetrics | None,
    all_failed: Sequence[str],
    divergent: Sequence[str],
) -> str:
    rows: list[str] = []
    if best is not None:
        rows.append(
            f"<div class=\"finding\"><b>Лидер общего рейтинга</b><br>"
            f"{escape(best.display_name)} - {escape(_pct(best.success.rate))} "
            f"на {best.n_cases} кейсах.</div>"
        )
    if best_full is not None:
        rows.append(
            f"<div class=\"finding\"><b>Полный набор</b><br>{escape(best_full.display_name)}: "
            f"mixed {escape(_pct(best_full.categories['mixed_diagnostic'].rate))}, "
            f"tool {escape(_pct(best_full.tool.rate))}.</div>"
        )
    if fastest is not None:
        rows.append(
            f"<div class=\"finding\"><b>Скорость</b><br>{escape(fastest.display_name)}: "
            f"p50 {escape(_num(fastest.latency_p50_sec))}s, "
            f"p95 {escape(_num(fastest.latency_p95_sec))}s.</div>"
        )
    if free is not None:
        rows.append(
            f"<div class=\"finding\"><b>Цена</b><br>{escape(free.display_name)}: "
            f"{escape(_pct(free.success.rate))} при API-cost $0.</div>"
        )
    rows.append(
        f"<div class=\"finding\"><b>All-fail</b><br>{len(all_failed)} кейсов: "
        f"{escape(', '.join(all_failed[:8]) or 'нет')}.</div>"
    )
    rows.append(
        f"<div class=\"finding\"><b>Расхождения</b><br>{len(divergent)} кейсов, где часть "
        "моделей прошла, а часть нет.</div>"
    )
    return "\n".join(rows)


def _summary_html_row(item: ModelMetrics) -> str:
    return (
        "<tr>"
        f"<td><b>{escape(item.display_name)}</b><br><span class=\"small\">{escape(item.bench_file)}</span></td>"
        f"<td class=\"num\">{escape(_count_rate(item.success))}</td>"
        f"<td class=\"num\">{escape(_count_rate(item.categories['blocked']))}</td>"
        f"<td class=\"num\">{escape(_count_rate(item.categories['legal_consult']))}</td>"
        f"<td class=\"num\">{escape(_count_rate(item.categories['mixed_diagnostic']))}</td>"
        f"<td class=\"num\">{escape(_count_rate(item.tool))}</td>"
        f"<td class=\"num\">{escape(_count_rate(item.multistep))}</td>"
        f"<td class=\"num\">{escape(_num(item.latency_p50_sec))}/"
        f"{escape(_num(item.latency_p95_sec))}s</td>"
        f"<td class=\"num\">{escape(_money(item.cost_total_usd))}</td>"
        f"<td class=\"num\">{escape(_num(item.calls_avg))}/"
        f"{escape(_num(item.tokens_avg, 0))}</td>"
        f"<td class=\"num\">{len(item.slow_cases)}</td>"
        "</tr>"
    )


def _notable_case_html(
    item: NotableCase,
    rows: Sequence[CaseByModel],
    bench_case: BenchmarkCase | None,
) -> str:
    detail_rows = "\n".join(
        _case_detail_row(row)
        for row in sorted(rows, key=lambda row: row.label)
    )
    prompt = ""
    if bench_case is not None and bench_case.input_text:
        prompt = f"<p class=\"small\">Запрос: {escape(_short(bench_case.input_text))}</p>"
    return f"""
<div class="case">
  <div class="case-title">
    <b>{escape(item.case_id)}</b>
    <span class="reason">{escape(item.reason)}</span>
  </div>
  {prompt}
  <table>
  <tr><th>Модель</th><th>ok</th><th>intent</th><th>latency</th><th>calls</th><th>expected_tools</th><th>got_tools</th></tr>
  {detail_rows}
  </table>
</div>
"""


def _case_detail_row(row: CaseByModel) -> str:
    case = row.case
    ok_class = "ok" if case.ok else "bad"
    ok_text = "yes" if case.ok else "no"
    return (
        "<tr>"
        f"<td>{escape(_display_name(row.label))}</td>"
        f"<td><span class=\"pill {ok_class}\">{ok_text}</span></td>"
        f"<td>{escape(case.intent or 'none')}</td>"
        f"<td class=\"num\">{escape(_num(case.latency_sec))}s</td>"
        f"<td class=\"num\">{case.calls}</td>"
        f"<td><code>{escape(', '.join(case.expected_tools) or '-')}</code></td>"
        f"<td><code>{escape(', '.join(case.got_tools) or '-')}</code></td>"
        "</tr>"
    )


def write_reports(audit_dir: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    runs = load_runs(audit_dir)
    if not runs:
        raise ValueError(f"No JSON run files found in {audit_dir}")
    metrics = build_model_metrics(runs)
    grouped = _cases_by_id(runs)
    bench_cases = load_benchmark_cases(runs)
    notable = select_notable_cases(runs, grouped)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / REPORT_MD
    html_path = out_dir / REPORT_HTML
    charts_path = out_dir / CHARTS_JSON

    md_path.write_text(
        markdown_report(audit_dir, runs, metrics, grouped, notable, bench_cases),
        encoding="utf-8",
    )
    charts_path.write_text(
        json.dumps(chart_data(metrics, runs, grouped, notable), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path.write_text(
        html_report(audit_dir, runs, metrics, grouped, notable, bench_cases),
        encoding="utf-8",
    )
    return md_path, html_path, charts_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate benchmark run JSON files.")
    parser.add_argument("audit_dir", nargs="?", help="Directory with run JSON files.")
    parser.add_argument("--out-dir", help="Output directory. Defaults to audit_dir.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_arg = args.audit_dir if isinstance(args.audit_dir, str) else None
    env_arg = os.environ.get("AUDIT_DIR")
    audit_dir_raw = audit_arg or env_arg
    if not audit_dir_raw:
        raise SystemExit("Pass audit_dir or set AUDIT_DIR.")
    out_dir_raw = args.out_dir if isinstance(args.out_dir, str) else audit_dir_raw
    audit_dir = Path(audit_dir_raw).expanduser().resolve()
    out_dir = Path(out_dir_raw).expanduser().resolve()
    md_path, html_path, charts_path = write_reports(audit_dir, out_dir)
    print(f"markdown: {md_path}")
    print(f"html: {html_path}")
    print(f"charts: {charts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
