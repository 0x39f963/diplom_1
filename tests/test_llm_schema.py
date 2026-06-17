from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from jsonschema import validate

from eva_agent.llm.base import LLMClient, LLMResponse
from eva_agent.llm.cli_agent import CliAgentClient
from eva_agent.llm.config import _TrackedClient
from eva_agent.llm.ollama_local import OllamaLocalClient
from eva_agent.llm.openrouter import OpenRouterClient
from tests.test_cli_agent import _completed

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


def test_ollama_sends_schema_as_format(monkeypatch) -> None:
    client = OllamaLocalClient("test-model", base_url="http://ollama.local")
    captured: dict[str, Any] = {}

    def fake_call(payload: dict[str, Any]) -> dict[str, Any]:
        captured.update(payload)
        return {"message": {"content": '{"name":"ok"}'}}

    monkeypatch.setattr(client, "_call", fake_call)

    response = client.invoke("system", "user", json_mode=True, schema=_SCHEMA)

    assert captured["format"] == _SCHEMA
    validate(json.loads(response.text), _SCHEMA)


def test_openrouter_sends_schema_response_format() -> None:
    completions = _FakeCompletions()
    client = OpenRouterClient(
        "test-model",
        api_key="test-key",
        base_url="https://openrouter.example/api/v1",
        max_retries=0,
    )
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    response = client.invoke("system", "user", json_mode=True, schema=_SCHEMA)

    assert completions.kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "frame", "schema": _SCHEMA, "strict": True},
    }
    validate(json.loads(response.text), _SCHEMA)


def test_cli_agent_adds_schema_to_prompt(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: Any):
        calls.append(list(args))
        return _completed(args, stdout=json.dumps({"result": '{"name":"ok"}'}))

    import eva_agent.llm.cli_agent as cli_agent

    monkeypatch.setattr(cli_agent.subprocess, "run", fake_run)
    client = CliAgentClient(
        provider="claude",
        model="sonnet",
        binary="claude-test",
        timeout=5,
        max_retries=0,
    )

    response = client.invoke("system", "user", schema=_SCHEMA)

    prompt = calls[0][calls[0].index("-p") + 1]
    assert "JSON Schema" in prompt
    assert '"required": ["name"]' in prompt
    validate(json.loads(response.text), _SCHEMA)


def test_tracked_client_passes_schema() -> None:
    inner = _InnerClient()
    client = _TrackedClient(inner)

    client.invoke("system", "user", schema=_SCHEMA)

    assert inner.schema == _SCHEMA


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return _FakeOpenRouterResponse()


class _FakeOpenRouterResponse:
    def model_dump(self) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": '{"name":"ok"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


class _InnerClient(LLMClient):
    backend = "fake"
    model = "fake"

    def __init__(self) -> None:
        self.schema: dict[str, Any] | None = None

    def invoke(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del system, user, temperature, json_mode
        self.schema = schema
        return LLMResponse(text='{"name":"ok"}', model=self.model, backend=self.backend)
