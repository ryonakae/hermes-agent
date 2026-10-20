"""Regression for #118349: ``message_changed`` must not replay an already-answered message.

``_processed_message_ts`` lives in memory and is bounded, so after a restart (or once a claim is
evicted) it cannot tell which messages were already answered. Slack keeps emitting
``message_changed`` for those messages on its own (thread reply metadata, agent sessions,
unfurls, file state). Before the fix each one became a new user turn with the original text,
and the bot answered it again.
"""

import asyncio
import importlib
import sys
import time
from importlib.machinery import PathFinder
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _load_installed_package(name):
    if PathFinder.find_spec(name) is None:
        return None
    prefix = f"{name}."
    displaced = {
        m: sys.modules.pop(m)
        for m in tuple(sys.modules)
        if (m == name or m.startswith(prefix)) and not isinstance(sys.modules[m], ModuleType)
    }
    try:
        return importlib.import_module(name)
    except ImportError:
        sys.modules.update(displaced)
        return None


_load_installed_package("slack_bolt")
_load_installed_package("slack_sdk")

SlackAdapter = importlib.import_module("plugins.platforms.slack.adapter").SlackAdapter

BOT = "U0BOT00001"
USER = "U0USER0001"
TEXT = f"<@{BOT}> summarize the release notes"


def _slack_ts(seconds: float) -> str:
    return f"{seconds:.6f}"


def _adapter(delivered):
    """A freshly started adapter: what the gateway has right after a restart."""
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._bot_user_id = BOT
    adapter._resolve_user_name = AsyncMock(return_value="user")

    async def _capture(event):
        delivered.append(event)

    adapter.handle_message = _capture
    return adapter


def _changed_event(message_ts, *, edited_ts=None, previous_text=None, text=TEXT):
    """``message_changed`` for ``message_ts``, emitted now."""
    change_ts = _slack_ts(time.time())
    message = {
        "type": "message", "user": USER, "text": text, "ts": message_ts,
        "client_msg_id": "cmid-1", "reply_count": 1, "latest_reply": change_ts,
    }
    if edited_ts:
        message["edited"] = {"user": USER, "ts": edited_ts}
    event = {
        "type": "message", "subtype": "message_changed", "hidden": True,
        "channel": "C0CHAN0001", "channel_type": "channel", "team": "T0TEAM0001",
        "ts": change_ts, "event_ts": change_ts, "message": message,
    }
    if previous_text is not None:
        event["previous_message"] = {
            "type": "message", "user": USER, "text": previous_text, "ts": message_ts}
    return event


def _run(adapter, event):
    asyncio.run(adapter._handle_slack_message(event, {"team_id": "T0TEAM0001"}))


def _answered_then_evicted(adapter, delivered):
    """A message answered by this adapter whose claim was later evicted from the bounded map."""
    adapter._PROCESSED_MESSAGE_TS_MAX = 2
    message_ts = _slack_ts(time.time())
    original = {
        "type": "message", "user": USER, "text": TEXT, "ts": message_ts,
        "channel": "C0CHAN0001", "channel_type": "channel", "team": "T0TEAM0001",
        "client_msg_id": "cmid-1"}
    _run(adapter, original)
    assert len(delivered) == 1
    for offset in (1, 2):
        adapter._remember_processed_message_ts(_slack_ts(time.time() + offset))
    assert message_ts not in adapter._processed_message_ts
    return _changed_event(message_ts, previous_text=TEXT)


@pytest.mark.parametrize(
    "shape", ["no_previous_message", "same_text", "edited_before_restart", "claim_evicted"])
def test_change_slack_made_to_an_already_answered_message_is_not_a_new_turn(shape):
    delivered = []
    adapter = _adapter(delivered)
    posted = time.time() - 600  # answered by the previous process, before this adapter started
    message_ts = _slack_ts(posted)
    event = {
        "no_previous_message": lambda: _changed_event(message_ts),
        "same_text": lambda: _changed_event(message_ts, previous_text=TEXT),
        "edited_before_restart": lambda: _changed_event(
            message_ts, edited_ts=_slack_ts(posted + 30), previous_text=TEXT),
        "claim_evicted": lambda: _answered_then_evicted(adapter, delivered),
    }[shape]()
    already_delivered = len(delivered)

    _run(adapter, event)

    assert len(delivered) == already_delivered


def test_edit_made_after_the_restart_still_reaches_the_agent():
    """An @mention edited into an older, unanswered message still wakes the bot once."""
    delivered = []
    adapter = _adapter(delivered)
    message_ts = _slack_ts(time.time() - 600)
    event = _changed_event(message_ts, previous_text="summarize the release notes")
    event["message"]["edited"] = {"user": USER, "ts": event["event_ts"]}

    _run(adapter, event)

    assert len(delivered) == 1
    assert delivered[0].message_id == message_ts


@pytest.mark.parametrize("with_previous", [False, True])
def test_metadata_after_an_evicted_post_restart_edit_is_not_redelivered(monkeypatch, with_previous):
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    delivered = []
    adapter = _adapter(delivered)
    adapter._PROCESSED_MESSAGE_TS_MAX = 2
    monkeypatch.setattr(time, "time", lambda: 1100.0)
    edit = _changed_event("500.000000", previous_text="summarize the release notes", edited_ts="1100.000000")
    _run(adapter, edit)
    assert len(delivered) == 1
    for now, original_ts in ((1110.0, "600.000000"), (1120.0, "700.000000")):
        monkeypatch.setattr(time, "time", lambda now=now: now)
        adapter._remember_processed_message_ts(original_ts)
    assert "500.000000" not in adapter._processed_message_ts
    monkeypatch.setattr(time, "time", lambda: 1200.0)
    metadata = _changed_event("500.000000", edited_ts="1100.000000")
    if with_previous:
        metadata["previous_message"] = dict(edit["message"])
    _run(adapter, metadata)
    assert len(delivered) == 1


@pytest.mark.parametrize("new_mention", [True, False])
def test_old_block_only_mention_change_is_distinguished_from_metadata(new_mention):
    delivered = []
    adapter = _adapter(delivered)
    text = "summarize the release notes"
    event = _changed_event(_slack_ts(time.time() - 600), text=text, previous_text=text)
    blocks = [{"type": "rich_text", "elements": [{
        "type": "rich_text_section", "elements": [{"type": "user", "user_id": BOT}]}]}]
    event["message"]["blocks"] = blocks
    event["previous_message"]["blocks"] = [] if new_mention else blocks
    _run(adapter, event)
    assert len(delivered) == int(new_mention)


def test_repeated_flat_mention_in_current_blocks_is_not_new_mention():
    delivered = []
    adapter = _adapter(delivered)
    event = _changed_event(
        _slack_ts(time.time() - 600), text=TEXT, previous_text=TEXT)
    blocks = [{"type": "rich_text", "elements": [{
        "type": "rich_text_section", "elements": [{"type": "user", "user_id": BOT}]}]}]
    event["message"]["blocks"] = blocks
    event["previous_message"]["blocks"] = []

    _run(adapter, event)

    assert len(delivered) == 0


@pytest.mark.parametrize("with_previous", [False, True])
def test_current_edit_timestamp_still_allows_an_old_message(with_previous):
    delivered = []
    adapter = _adapter(delivered)
    event = _changed_event(_slack_ts(time.time() - 600))
    event["message"]["edited"] = {"user": USER, "ts": event["event_ts"]}
    if with_previous:
        event["previous_message"] = {"text": TEXT}
    _run(adapter, event)
    assert len(delivered) == 1


def test_advancing_edit_timestamp_allows_delayed_edit_event(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    delivered = []
    adapter = _adapter(delivered)
    monkeypatch.setattr(time, "time", lambda: 1200.0)
    event = _changed_event("500.000000", previous_text=TEXT, edited_ts="1100.000000")
    event["previous_message"]["edited"] = {"user": USER, "ts": "900.000000"}
    _run(adapter, event)
    assert len(delivered) == 1
