from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import _CodexCompletionsAdapter, _build_call_kwargs, call_llm


def _structured_extra_body() -> dict:
    return {
        "reasoning": {"effort": "low"},
        "hermes_structured_output": {
            "type": "json_schema",
            "name": "role_output",
            "schema": {"type": "object", "properties": {"status": {"type": "string"}}},
            "strict": False,
        },
    }


def _chat_response(text: str = "ok"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class _FakeChatClient:
    def __init__(self, side_effects):
        self.calls = []
        self._side_effects = list(side_effects)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.base_url = "https://api.openai.com/v1"
        self.api_key = "sk-test"

    def create(self, **kwargs):
        self.calls.append(kwargs)
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


def test_build_call_kwargs_translates_neutral_intent_to_chat_response_format():
    kwargs = _build_call_kwargs(
        provider="openrouter",
        model="openai/gpt-test",
        messages=[{"role": "user", "content": "hi"}],
        extra_body=_structured_extra_body(),
        base_url="https://openrouter.ai/api/v1",
    )

    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "role_output",
            "schema": {"type": "object", "properties": {"status": {"type": "string"}}},
            "strict": False,
        },
    }
    assert kwargs["extra_body"] == {"reasoning": {"effort": "low"}}
    assert "text" not in kwargs
    assert "hermes_structured_output" not in kwargs["extra_body"]


def test_build_call_kwargs_strips_neutral_intent_for_anthropic_wire():
    kwargs = _build_call_kwargs(
        provider="custom",
        model="claude-test",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        extra_body=_structured_extra_body(),
        base_url="https://api.anthropic.com/v1",
    )

    assert "response_format" not in kwargs
    assert "text" not in kwargs
    assert kwargs["extra_body"] == {"reasoning": {"effort": "low"}}
    assert "hermes_structured_output" not in kwargs["extra_body"]


def test_codex_adapter_translates_neutral_intent_to_responses_text_format():
    captured = {}
    message_item = SimpleNamespace(
        type="message",
        role="assistant",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="hi")],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_item.done", item=message_item),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed", id="resp_test")),
    ]

    class _FakeCreateStream:
        def __iter__(self):
            return iter(events)

        def close(self):
            pass

    def create(**kwargs):
        captured.update(kwargs)
        return _FakeCreateStream()

    real_client = MagicMock()
    real_client.responses.create = create
    adapter = _CodexCompletionsAdapter(real_client, "gpt-5.4")

    adapter.create(messages=[{"role": "user", "content": "hi"}], extra_body=_structured_extra_body())

    assert captured["text"] == {
        "format": {
            "type": "json_schema",
            "name": "role_output",
            "schema": {"type": "object", "properties": {"status": {"type": "string"}}},
            "strict": False,
        }
    }
    assert captured["reasoning"] == {"effort": "low", "summary": "auto"}
    assert "response_format" not in captured
    assert "hermes_structured_output" not in str(captured)


def test_codex_adapter_translates_chat_response_format_to_responses_text_format():
    captured = {}
    message_item = SimpleNamespace(
        type="message",
        role="assistant",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="hi")],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_item.done", item=message_item),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed", id="resp_test")),
    ]

    class _FakeCreateStream:
        def __iter__(self):
            return iter(events)

        def close(self):
            pass

    def create(**kwargs):
        captured.update(kwargs)
        return _FakeCreateStream()

    real_client = MagicMock()
    real_client.responses.create = create
    adapter = _CodexCompletionsAdapter(real_client, "gpt-5.4")

    adapter.create(
        messages=[{"role": "user", "content": "hi"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "role_output",
                "schema": {"type": "object"},
                "strict": False,
            },
        },
    )

    assert captured["text"] == {
        "format": {
            "type": "json_schema",
            "name": "role_output",
            "schema": {"type": "object"},
            "strict": False,
        }
    }
    assert "response_format" not in captured

def test_call_llm_retries_once_without_structured_output_when_chat_provider_rejects_response_format():
    client = _FakeChatClient([
        RuntimeError("unsupported_parameter: response_format is not supported"),
        _chat_response("ok"),
    ])

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("custom", "gpt-test", "https://api.openai.com/v1", "sk-test", None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "gpt-test")),
    ):
        result = call_llm(
            task="hermes_trading_role",
            messages=[{"role": "user", "content": "hi"}],
            extra_body=_structured_extra_body(),
        )

    assert result.choices[0].message.content == "ok"
    assert len(client.calls) == 2
    assert "response_format" in client.calls[0]
    assert "response_format" not in client.calls[1]
    assert client.calls[1]["extra_body"] == {"reasoning": {"effort": "low"}}


def test_call_llm_does_not_retry_unrelated_errors():
    client = _FakeChatClient([RuntimeError("Internal Server Error")])

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("custom", "gpt-test", "https://api.openai.com/v1", "sk-test", None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "gpt-test")),
    ):
        with pytest.raises(RuntimeError, match="Internal Server Error"):
            call_llm(
                task="hermes_trading_role",
                messages=[{"role": "user", "content": "hi"}],
                extra_body=_structured_extra_body(),
            )

    assert len(client.calls) == 1
