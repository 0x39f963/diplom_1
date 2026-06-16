"""Local CLI-agent connector.

Effort mapping, verified against the installed CLI help:

| Provider | low | medium | high |
| --- | --- | --- | --- |
| claude | `--effort low` | `--effort medium` | `--effort high` |
| codex | `-c model_reasoning_effort=low` | `-c model_reasoning_effort=medium` | `-c model_reasoning_effort=high` |

Both providers receive one non-interactive prompt assembled from the system and user
messages. CLI JSON output is used for token and cost metadata when present.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from eva_agent.llm.base import LLMClient, LLMConfigError, LLMResponse
from eva_agent.llm.observability import langfuse_enabled
from eva_agent.settings import Effort

Provider = Literal["claude", "codex"]

_BACKOFF_CAP_SEC = 5.0
_JSON_MODE_INSTRUCTION = "Return strict JSON only. Do not wrap it in Markdown."


@lru_cache(maxsize=1)
def _isolated_cwd() -> str:
    """Пустая рабочая папка для запуска CLI.

    Запуск в нейтральной директории не дает агентскому CLI подхватить рабочие
    настройки и инструкции проекта (он должен отвечать как обычная модель).
    """
    return tempfile.mkdtemp(prefix="eva-cli-sandbox-")


@dataclass(frozen=True)
class _CliOutput:
    text: str
    usage: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class _CliRun:
    output: _CliOutput
    retry_log: list[str]


class _CliProcessError(RuntimeError):
    def __init__(self, *, returncode: int, stderr: str, stdout: str) -> None:
        self.returncode = returncode
        detail = (stderr.strip() or stdout.strip())[:500]
        suffix = f": {detail}" if detail else ""
        super().__init__(f"CLI process failed with code {returncode}{suffix}")


class CliAgentClient(LLMClient):
    def __init__(
        self,
        *,
        provider: Provider,
        model: str,
        effort: Effort = "medium",
        binary: str | None = None,
        timeout: int = 300,
        max_retries: int = 2,
    ) -> None:
        self.provider = provider
        self.model = model
        self.backend = f"{provider}_cli"
        self.effort = effort
        self._binary = binary or provider
        self._timeout = timeout
        self._max_retries = max_retries

    def invoke(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        prompt = _build_prompt(system, user, json_mode=json_mode)
        started = time.monotonic()
        result = self._run_with_observation(prompt, system, user)
        return LLMResponse(
            text=result.output.text,
            model=self.model,
            backend=self.backend,
            walltime_sec=time.monotonic() - started,
            usage=result.output.usage,
            retry_log=result.retry_log,
            raw={
                **result.output.raw,
                "provider": self.provider,
                "effort": self.effort,
                "temperature": temperature,
            },
        )

    def _run_with_observation(self, prompt: str, system: str, user: str) -> _CliRun:
        if not langfuse_enabled():
            return self._run_with_retries(prompt)

        from langfuse import get_client

        client = get_client()
        with client.start_as_current_observation(
            as_type="generation",
            name=f"{self.provider}-cli-generation",
            model=self.model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        ):
            result = self._run_with_retries(prompt)
            client.update_current_generation(
                output=result.output.text,
                usage_details=_langfuse_usage(result.output.usage),
                cost_details=_langfuse_cost(result.output.usage),
            )
            return result

    def _run_with_retries(self, prompt: str) -> _CliRun:
        retry_log: list[str] = []
        last_error: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return _CliRun(output=self._run_once(prompt), retry_log=retry_log)
            except FileNotFoundError as exc:
                raise LLMConfigError(f"CLI binary not found: {self._binary}") from exc
            except subprocess.TimeoutExpired as exc:
                last_error = exc
                retry_log.append(f"attempt {attempt}: timeout after {self._timeout}s")
            except _CliProcessError as exc:
                last_error = exc
                retry_log.append(f"attempt {attempt}: process {exc.returncode}")

            if attempt < self._max_retries:
                self._sleep(attempt)

        if isinstance(last_error, subprocess.TimeoutExpired):
            raise TimeoutError(f"{self.provider} CLI timed out after {self._timeout}s") from last_error
        if last_error is not None:
            raise RuntimeError(str(last_error)) from last_error
        raise RuntimeError("unreachable")

    def _run_once(self, prompt: str) -> _CliOutput:
        if self.provider == "claude":
            return self._run_claude(prompt)
        return self._run_codex(prompt)

    def _run_claude(self, prompt: str) -> _CliOutput:
        command = [
            self._binary,
            "-p",
            prompt,
            "--model",
            self.model,
            "--output-format",
            "json",
            "--effort",
            self.effort,
        ]
        completed = self._run_process(command, stdin=None)
        _raise_for_returncode(completed)
        payload = _parse_json_payload(completed.stdout)
        text = _extract_text(payload) or completed.stdout.strip()
        usage = _extract_usage(payload)
        return _CliOutput(
            text=text,
            usage=usage,
            raw={"stdout": payload, "stderr": completed.stderr, "command": _safe_command(command)},
        )

    def _run_codex(self, prompt: str) -> _CliOutput:
        with tempfile.NamedTemporaryFile(prefix="eva-codex-", suffix=".txt", delete=False) as tmp:
            output_path = Path(tmp.name)
        command = [
            self._binary,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--json",
            "-m",
            self.model,
            "-c",
            f"model_reasoning_effort={self.effort}",
            "-o",
            str(output_path),
            "-",
        ]
        try:
            completed = self._run_process(command, stdin=prompt)
            _raise_for_returncode(completed)
            events = _parse_json_lines(completed.stdout)
            text = output_path.read_text(encoding="utf-8").strip()
            if not text:
                text = _extract_last_text(events) or completed.stdout.strip()
            return _CliOutput(
                text=text,
                usage=_extract_usage(events),
                raw={
                    "stdout_events": events,
                    "stderr": completed.stderr,
                    "command": _safe_command(command),
                },
            )
        finally:
            output_path.unlink(missing_ok=True)

    def _run_process(
        self, command: Sequence[str], *, stdin: str | None
    ) -> subprocess.CompletedProcess[str]:
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            list(command),
            input=stdin,
            capture_output=True,
            text=True,
            timeout=self._timeout,
            check=False,
            cwd=_isolated_cwd(),
        )
        return completed

    @staticmethod
    def _sleep(attempt: int) -> None:
        delay = min(_BACKOFF_CAP_SEC, 1.5 * (2**attempt))
        time.sleep(delay)


def _build_prompt(system: str, user: str, *, json_mode: bool) -> str:
    parts = []
    if system.strip():
        parts.append(f"System instruction:\n{system.strip()}")
    if user.strip():
        parts.append(f"User request:\n{user.strip()}")
    if json_mode:
        parts.append(_JSON_MODE_INSTRUCTION)
    return "\n\n".join(parts)


def _raise_for_returncode(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.returncode != 0:
        raise _CliProcessError(
            returncode=completed.returncode,
            stderr=completed.stderr,
            stdout=completed.stdout,
        )


def _parse_json_payload(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {"text": stdout}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"items": payload}
    return {"value": payload}


def _parse_json_lines(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _extract_last_text(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        text = _extract_text(event)
        if text:
            return text
    return ""


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if not isinstance(value, dict):
        return ""

    for key in ("result", "text", "response", "output"):
        text = _extract_text(value.get(key))
        if text:
            return text
    for key in ("message", "content"):
        text = _extract_text(value.get(key))
        if text:
            return text
    choices = value.get("choices")
    if isinstance(choices, list) and choices:
        return _extract_text(choices[0])
    return ""


def _extract_usage(value: Any) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    _merge_usage(value, usage)
    prompt_tokens = _as_int(usage.get("prompt_tokens"))
    completion_tokens = _as_int(usage.get("completion_tokens"))
    if usage.get("total_tokens") is None and prompt_tokens is not None and completion_tokens is not None:
        usage["total_tokens"] = prompt_tokens + completion_tokens
    return {key: item for key, item in usage.items() if item is not None}


def _merge_usage(value: Any, usage: dict[str, Any]) -> None:
    if isinstance(value, list):
        for item in value:
            _merge_usage(item, usage)
        return
    if not isinstance(value, dict):
        return

    _copy_first(value, usage, ("prompt_tokens", "input_tokens"), "prompt_tokens")
    _copy_first(value, usage, ("completion_tokens", "output_tokens"), "completion_tokens")
    _copy_first(value, usage, ("total_tokens",), "total_tokens")
    _copy_first(value, usage, ("cost_usd", "total_cost_usd", "cost"), "cost_usd")

    for nested in value.values():
        if isinstance(nested, dict | list):
            _merge_usage(nested, usage)


def _copy_first(source: dict[str, Any], target: dict[str, Any], keys: tuple[str, ...], name: str) -> None:
    if target.get(name) is not None:
        return
    for key in keys:
        if key in source and source[key] is not None:
            target[name] = source[key]
            return


def _langfuse_usage(usage: dict[str, Any]) -> dict[str, int]:
    details: dict[str, int] = {}
    prompt_tokens = _as_int(usage.get("prompt_tokens"))
    completion_tokens = _as_int(usage.get("completion_tokens"))
    if prompt_tokens is not None:
        details["input"] = prompt_tokens
    if completion_tokens is not None:
        details["output"] = completion_tokens
    return details


def _langfuse_cost(usage: dict[str, Any]) -> dict[str, float]:
    cost = _as_float(usage.get("cost_usd"))
    if cost is None:
        return {}
    return {"total": cost}


def _safe_command(command: Sequence[str]) -> list[str]:
    return [
        part
        if not (part.startswith("System instruction:") or part.startswith("User request:"))
        else "<prompt>"
        for part in command
    ]


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
