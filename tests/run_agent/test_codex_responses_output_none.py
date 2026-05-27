"""Regression tests for the ChatGPT Codex ``response.output = None`` crash.

When Hermes uses the ``openai-codex`` provider against the ChatGPT backend
(``https://chatgpt.com/backend-api/codex``), the OpenAI SDK's Responses
streaming accumulator (``parse_response``) raises
``TypeError("'NoneType' object is not iterable")`` because it runs
``for output in response.output`` and that backend emits an event whose
``response.output`` is ``None`` rather than ``[]``.

``_run_codex_stream`` now treats that accumulator TypeError the same way it
already treats the prelude/postlude RuntimeErrors — fall back to the manual
``responses.create(stream=True)`` path, which never calls the accumulator.
The match is narrow (``"is not iterable"``) so genuine TypeErrors in our own
callbacks still propagate, mirroring
``test_codex_stream_unrelated_runtimeerror_still_raises``.

The fallback's empty-output backfill also now treats ``output is None`` like
an empty list, so it synthesizes from collected items/deltas in that case too.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_codex_agent():
    """Build a minimal AIAgent wired for codex_responses streaming tests."""
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-5.5",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "codex_responses"
    agent.provider = "openai-codex"
    agent._interrupt_requested = False
    return agent


# ---------------------------------------------------------------------------
# Change 1: accumulator TypeError -> create(stream=True) fallback
# ---------------------------------------------------------------------------


def test_codex_stream_accumulator_typeerror_falls_back_to_create_stream():
    """The SDK accumulator's ``'NoneType' object is not iterable`` must route
    to the non-stream fallback instead of surfacing as a non-retryable error."""
    agent = _make_codex_agent()

    accumulator_error = TypeError("'NoneType' object is not iterable")

    mock_client = MagicMock()
    mock_client.responses.stream.side_effect = accumulator_error

    fallback_response = SimpleNamespace(
        output=[SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="fallback ok")],
        )],
        status="completed",
    )

    with patch.object(
        agent, "_run_codex_create_stream_fallback", return_value=fallback_response
    ) as mock_fallback:
        result = agent._run_codex_stream({}, client=mock_client)

    assert result is fallback_response
    mock_fallback.assert_called_once_with({}, client=mock_client)


def test_codex_stream_unrelated_typeerror_still_raises():
    """TypeErrors that aren't the accumulator's iterable error (e.g. a bug in
    one of our own stream callbacks) must propagate, not be swallowed."""
    agent = _make_codex_agent()

    mock_client = MagicMock()
    mock_client.responses.stream.side_effect = TypeError(
        "build_api_kwargs() got an unexpected keyword argument"
    )

    with patch.object(agent, "_run_codex_create_stream_fallback") as mock_fallback:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            agent._run_codex_stream({}, client=mock_client)

    mock_fallback.assert_not_called()


# ---------------------------------------------------------------------------
# Change 2: fallback backfill treats output=None like an empty list
# ---------------------------------------------------------------------------


def _completed_event(response):
    return SimpleNamespace(type="response.completed", response=response)


# A request that satisfies the Codex Responses preflight validator
# (model + instructions + input are required before the stream is opened).
_VALID_CODEX_KWARGS = {
    "model": "gpt-5.5",
    "instructions": "be brief",
    "input": [{"role": "user", "content": "ping"}],
}


def test_fallback_synthesizes_output_when_terminal_output_is_none():
    """A terminal ``response.completed`` whose ``output`` is ``None`` must be
    backfilled from the text deltas collected during the manual stream."""
    agent = _make_codex_agent()

    terminal_response = SimpleNamespace(output=None, status="completed")
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="po"),
        SimpleNamespace(type="response.output_text.delta", delta="ng"),
        _completed_event(terminal_response),
    ]

    mock_client = MagicMock()
    # create(stream=True) returns an iterable of events (no ``.output`` attr,
    # so the compatibility shim does not short-circuit).
    mock_client.responses.create.return_value = iter(events)

    result = agent._run_codex_create_stream_fallback(_VALID_CODEX_KWARGS, client=mock_client)

    assert result is terminal_response
    assert isinstance(result.output, list) and result.output
    assert result.output[0].content[0].text == "pong"


def test_fallback_backfills_items_when_terminal_output_is_none():
    """When output items arrived via ``response.output_item.done`` they take
    precedence over synthesized deltas, even if terminal output is None."""
    agent = _make_codex_agent()

    done_item = SimpleNamespace(
        type="message",
        role="assistant",
        content=[SimpleNamespace(type="output_text", text="real item")],
    )
    terminal_response = SimpleNamespace(output=None, status="completed")
    events = [
        SimpleNamespace(type="response.output_item.done", item=done_item),
        _completed_event(terminal_response),
    ]

    mock_client = MagicMock()
    mock_client.responses.create.return_value = iter(events)

    result = agent._run_codex_create_stream_fallback(_VALID_CODEX_KWARGS, client=mock_client)

    assert result is terminal_response
    assert result.output == [done_item]
