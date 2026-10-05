"""Tests for gateway /goal verdict-message delivery.

The judge verdict message ("✓ Goal achieved", "⏸ budget exhausted", etc.)
must reach the user after each turn. Before this fix the code checked
``hasattr(adapter, "send_message")`` — but adapters expose ``send()``,
never ``send_message``, so the check always evaluated False and users
never saw verdicts. This test locks in the fix.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    # Pre-warm the SessionDB cache from this SYNC context. The tests call
    # GoalManager.set() on the event-loop thread, where _get_session_db()
    # refuses to construct SessionDB inline (loop-liveness guard) and only
    # waits _DB_BOOTSTRAP_LOOP_WAIT_S for a background bootstrap. On a loaded
    # CI runner the init overruns that window, the goal write is silently
    # dropped by design, and the continuation path no-ops — the recurring
    # sends == [] flake. Warming here uses the direct construction path, so
    # the loop-thread set() always finds a cached DB.
    goals._get_session_db()
    yield home
    goals._DB_CACHE.clear()


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


class _RecordingAdapter:
    """Minimal adapter that records send() invocations."""

    def __init__(self) -> None:
        self._pending_messages: dict = {}
        self.sends: list[dict] = []

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None):
        self.sends.append({"chat_id": chat_id, "content": content, "metadata": metadata})

        class _R:
            success = True
            message_id = "mock-msg"

        return _R()


class _CallbackRecordingAdapter(_RecordingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.callbacks: dict = {}

    def register_post_delivery_callback(self, session_key, callback, *, generation=None):
        self.callbacks[session_key] = (generation, callback)


def _make_runner_with_adapter(session_id: str = None):
    from gateway.run import GatewayRunner
    import uuid

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._queued_events = {}

    src = _make_source()
    # Default to a unique session_id so xdist parallel runs on the same worker
    # don't see each other's GoalManager state (DEFAULT_DB_PATH gets frozen at
    # module-import time, defeating per-test HERMES_HOME monkeypatches).
    session_entry = SessionEntry(
        session_key=build_session_key(src),
        session_id=session_id or f"goal-sess-{uuid.uuid4().hex[:8]}",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._generate_session_key.return_value = build_session_key(src)

    adapter = _RecordingAdapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    return runner, adapter, session_entry, src


async def _drain_until(condition, timeout=5.0):
    """Yield to the event loop until ``condition()`` is truthy (bounded).

    The goal-continuation path finishes its sends/enqueues on spawned tasks;
    a fixed 0.05s sleep raced them on loaded CI runners (#88975). Returns as
    soon as the condition holds — the asserts after the call stay exact.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while not condition() and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_goal_verdict_continue_enqueues_continuation(hermes_home):
    """When the judge says continue, both the 'continuing' status and the
    continuation-prompt event must be delivered. The continuation prompt is
    routed through the adapter's pending-messages FIFO so the goal loop
    proceeds on the next turn."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("polish the docs")

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "still needs work", False, None, False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="here's a partial edit",
        )
        await _drain_until(lambda: adapter.sends and adapter._pending_messages)

    # Status line sent back
    assert len(adapter.sends) == 1
    assert "Continuing toward goal" in adapter.sends[0]["content"]
    # Continuation prompt enqueued for next turn
    assert adapter._pending_messages, "continuation prompt must be enqueued in pending_messages"


@pytest.mark.asyncio
async def test_goal_verdict_budget_exhausted_sends_pause(hermes_home):
    """When the budget is exhausted, a '⏸ Goal paused' message must be sent
    and no further continuation enqueued."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager, save_goal

    mgr = GoalManager(session_entry.session_id, default_max_turns=2)
    state = mgr.set("tiny goal", max_turns=2)
    state.turns_used = 2
    save_goal(session_entry.session_id, state)

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "keep going", False, None, False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="still partial",
        )
        await _drain_until(lambda: adapter.sends)

    assert len(adapter.sends) == 1
    content = adapter.sends[0]["content"]
    assert "paused" in content.lower()
    assert "turns used" in content.lower()
    # No continuation enqueued when budget is exhausted
    assert not adapter._pending_messages


@pytest.mark.asyncio
async def test_goal_verdict_streamed_done_sends_status_immediately(hermes_home):
    """A streamed body has already landed, so its goal status cannot wait for a later send callback."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()
    adapter = _CallbackRecordingAdapter()
    runner.adapters[Platform.TELEGRAM] = adapter

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("ship the feature")

    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("done", "the feature shipped", False, None, False),
    ):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="I shipped the feature.",
            response_already_delivered=True,
        )

    assert len(adapter.sends) == 1
    assert "Goal achieved" in adapter.sends[0]["content"]
    assert adapter.callbacks == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [True, False])
async def test_post_turn_hooks_deliver_goal_notice_after_body(hermes_home, streamed):
    from types import SimpleNamespace
    from hermes_cli.goals import GoalManager

    runner, _, session_entry, src = _make_runner_with_adapter()
    adapter = _CallbackRecordingAdapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    GoalManager(session_entry.session_id).set("ship the feature")
    body = "I shipped the feature."
    event = SimpleNamespace(_streamed_final_response=body if streamed else None)
    if streamed:
        await adapter.send(src.chat_id, body)

    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("done", "the feature shipped", False, None, False),
    ):
        await runner._run_post_turn_hooks(
            agent_result=None if streamed else body,
            source=src, is_internal=False, event=event,
        )

    if streamed:
        assert adapter.callbacks == {}
    else:
        assert adapter.sends == []
        assert len(adapter.callbacks) == 1
        await adapter.send(src.chat_id, body)
        _, callback = adapter.callbacks.pop(build_session_key(src))
        await callback()

    assert len(adapter.sends) == 2
    assert adapter.sends[0]["content"] == body
    assert "Goal achieved" in adapter.sends[1]["content"]
    assert adapter.callbacks == {}


