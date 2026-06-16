from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from eva_agent.llm import cli_agent
from eva_agent.llm.base import LLMConfigError
from eva_agent.llm.cli_agent import CliAgentClient
from eva_agent.llm.config import get_client
from eva_agent.settings import settings


def _completed(
    args: Sequence[str],
    *,
    stdout: str,
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=list(args),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_codex_command_maps_effort_and_reads_last_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        command = list(args)
        calls.append(command)
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text("codex answer", encoding="utf-8")
        assert kwargs["input"].startswith("System instruction:\nsystem")
        stdout = json.dumps({"usage": {"input_tokens": 11, "output_tokens": 4}})
        return _completed(command, stdout=f"{stdout}\n")

    monkeypatch.setattr(cli_agent.subprocess, "run", fake_run)
    client = CliAgentClient(
        provider="codex",
        model="gpt-5.5",
        effort="high",
        binary="codex-test",
        timeout=5,
        max_retries=0,
    )

    response = client.invoke("system", "user")

    command = calls[0]
    assert command[:2] == ["codex-test", "exec"]
    assert "--json" in command
    assert command[command.index("-m") + 1] == "gpt-5.5"
    assert command[command.index("-c") + 1] == "model_reasoning_effort=high"
    assert command[-1] == "-"
    assert response.text == "codex answer"
    assert response.model == "gpt-5.5"
    assert response.backend == "codex_cli"
    assert response.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
    }


def test_claude_command_maps_effort_and_parses_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        command = list(args)
        calls.append(command)
        stdout = json.dumps(
            {
                "result": "claude answer",
                "usage": {"input_tokens": 7, "output_tokens": 3},
                "total_cost_usd": 0.02,
            }
        )
        return _completed(command, stdout=stdout)

    monkeypatch.setattr(cli_agent.subprocess, "run", fake_run)
    client = CliAgentClient(
        provider="claude",
        model="opus",
        effort="medium",
        binary="claude-test",
        timeout=5,
        max_retries=0,
    )

    response = client.invoke("system", "user", json_mode=True)

    command = calls[0]
    assert command[0] == "claude-test"
    assert "-p" in command
    assert "--model" in command
    assert command[command.index("--model") + 1] == "opus"
    assert command[command.index("--effort") + 1] == "medium"
    assert command[command.index("--output-format") + 1] == "json"
    assert "Return strict JSON only" in command[command.index("-p") + 1]
    assert response.text == "claude answer"
    assert response.backend == "claude_cli"
    assert response.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "cost_usd": 0.02,
        "total_tokens": 10,
    }


def test_timeout_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=list(args), timeout=1)

    monkeypatch.setattr(cli_agent.subprocess, "run", fake_run)
    client = CliAgentClient(
        provider="claude",
        model="sonnet",
        effort="low",
        binary="claude-test",
        timeout=1,
        max_retries=0,
    )

    with pytest.raises(TimeoutError, match="claude CLI timed out after 1s"):
        client.invoke("system", "user")


def test_process_error_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return _completed(args, stdout="", stderr="bad flag", returncode=2)

    monkeypatch.setattr(cli_agent.subprocess, "run", fake_run)
    client = CliAgentClient(
        provider="claude",
        model="sonnet",
        effort="low",
        binary="claude-test",
        timeout=1,
        max_retries=0,
    )

    with pytest.raises(RuntimeError, match="CLI process failed with code 2: bad flag"):
        client.invoke("system", "user")


def test_missing_binary_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(cli_agent.subprocess, "run", fake_run)
    client = CliAgentClient(
        provider="codex",
        model="gpt-5.5",
        effort="medium",
        binary="missing-codex",
        timeout=1,
        max_retries=0,
    )

    with pytest.raises(LLMConfigError, match="CLI binary not found: missing-codex"):
        client.invoke("system", "user")


def test_get_client_dispatches_claude_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return _completed(args, stdout=json.dumps({"result": "ok"}))

    monkeypatch.setattr(cli_agent.subprocess, "run", fake_run)
    monkeypatch.setattr(settings, "llm_backend_default", "claude_cli")
    monkeypatch.setattr(settings, "llm_model_default", "sonnet")
    monkeypatch.setattr(settings, "llm_effort_default", "low")
    monkeypatch.setattr(settings, "eva_cli_claude_bin", "claude-test")
    monkeypatch.setattr(settings, "llm_provider_max_retries", 0)

    response = get_client("default").invoke("system", "user")

    assert response.text == "ok"
    assert response.backend == "claude_cli"
    assert response.model == "sonnet"
