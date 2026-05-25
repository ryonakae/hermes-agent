"""Regression tests for the Codex Responses stream activity watchdog.

The TTFB watchdog (see test_codex_ttfb_watchdog.py) handles the case where the
backend accepts a connection but never emits a single stream event. Once *any*
event has arrived, the connection is healthy and the stale-call detector
takes over.

The bug this file covers: prior to the activity-watchdog fix, that stale
detector measured against ``_call_start`` even for ``codex_responses``, which
streams events under the hood of what looks like a single non-streaming call.
So a long-but-healthy turn (extended reasoning, multi-tool-call sequences)
crossed the wall-clock stale threshold while events were still flowing and
got misreported as ``No response from provider for Ns (non-streaming, ...)``.

The fix: when ``api_mode == "codex_responses"`` and a stream event has been
observed, measure staleness from ``agent._codex_stream_last_event_ts`` rather
than from the call start. Non-codex modes keep wall-clock semantics.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _make_codex_agent(tmp_path, monkeypatch, *, stale_timeout: float = 1.0):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent.api_mode = "codex_responses"
    # Disable TTFB so this file exclusively exercises the activity watchdog.
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0")
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda *a, **k: stale_timeout
    )
    agent._emitted_status: list = []
    monkeypatch.setattr(
        agent, "_emit_status", lambda msg: agent._emitted_status.append(msg)
    )
    return agent


def _wire_client_closes(agent, monkeypatch):
    """Capture every connection-close reason so the test can assert on them."""
    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    return closes


def test_stale_kicks_in_when_events_stop_flowing(tmp_path, monkeypatch):
    """Events arrived (so TTFB does not apply), then the stream went silent
    for longer than the stale timeout — the activity watchdog kills the call
    and the surfaced error/status reflect "no new event" rather than the
    legacy "non-streaming, no response" wording."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch, stale_timeout=1.0)
    closes = _wire_client_closes(agent, monkeypatch)

    stop = {"flag": False}

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        # Stream a single event, then go silent until either killed or 30s pass.
        agent._codex_stream_last_event_ts = time.time()
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    t0 = time.time()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        elapsed = time.time() - t0
        assert "no new event" in str(excinfo.value).lower()
        assert "stale_call_kill" in closes
        # Must fire near the 1s threshold + 2s join grace — well under 30s.
        assert elapsed < 10, f"activity watchdog took {elapsed:.1f}s"
        # The user-facing status must say "stream events", not "non-streaming".
        emitted = " ".join(agent._emitted_status)
        assert "stream events" in emitted
        assert "non-streaming" not in emitted
    finally:
        stop["flag"] = True


def test_long_healthy_stream_is_not_killed(tmp_path, monkeypatch):
    """Events keep flowing past the wall-clock stale threshold — total elapsed
    > stale_timeout, but the gap between the last event and now is always
    short, so the watchdog must NOT kill the connection."""
    from agent import chat_completion_helpers as h

    # 0.4s stale window; we will run ~1.2s of streaming with ~50ms event gap.
    agent = _make_codex_agent(tmp_path, monkeypatch, stale_timeout=0.4)
    closes = _wire_client_closes(agent, monkeypatch)
    sentinel = SimpleNamespace(ok=True)

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        # 24 events × 50ms = 1.2s real elapsed, but the last-event gap is
        # always ~50ms — well under the 0.4s stale window.
        for _ in range(24):
            agent._codex_stream_last_event_ts = time.time()
            time.sleep(0.05)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    t0 = time.time()
    resp = h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
    elapsed = time.time() - t0
    assert resp is sentinel
    assert "stale_call_kill" not in closes
    assert elapsed >= 1.0, "fake stream should have run > 1s"
    # No timeout status should have been emitted while events were flowing.
    assert not any("stream events" in m for m in agent._emitted_status)


def test_non_codex_mode_keeps_wall_clock_semantics(tmp_path, monkeypatch):
    """The activity watchdog is gated on ``codex_responses``. Other api_modes
    (chat_completions non-stream, anthropic, bedrock) have no stream-event
    signal and must continue to fire stale based on wall-clock elapsed."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch, stale_timeout=1.0)
    agent.api_mode = "chat_completions"  # NOT codex_responses
    closes = _wire_client_closes(agent, monkeypatch)

    # Even if some other code path stamped this marker, it must not influence
    # non-codex_responses requests. Set it to "now" — under codex rules the
    # call would be considered freshly alive; under wall-clock it still dies.
    agent._codex_stream_last_event_ts = time.time()

    stop = {"flag": False}

    def fake_blocking_call(*a, **k):
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            # Keep refreshing the codex marker so a buggy activity check would
            # see the call as "healthy" and never fire.
            agent._codex_stream_last_event_ts = time.time()
            time.sleep(0.05)
        raise RuntimeError("connection closed")

    # interruptible_api_call falls through to the chat_completions branch and
    # calls openai_client.chat.completions.create(**api_kwargs); we stub that.
    fake_openai = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_blocking_call)
        )
    )
    monkeypatch.setattr(agent, "openai_client", fake_openai, raising=False)
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: fake_openai)

    t0 = time.time()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]})
        elapsed = time.time() - t0
        # Wall-clock path: message must say "non-streaming", not "no new event".
        assert "non-streaming" in str(excinfo.value).lower()
        assert "no new event" not in str(excinfo.value).lower()
        # Should still fire near the 1s threshold + 2s join grace.
        assert elapsed < 10, f"wall-clock watchdog took {elapsed:.1f}s"
    finally:
        stop["flag"] = True
