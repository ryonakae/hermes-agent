"""A failed native draft must not orphan its visible Slack message."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["prefix_mismatch", "append_error"])
@pytest.mark.parametrize("delivery", ["edit", "replacement", "failed_replacement"])
async def test_failed_draft_keeps_preview_until_replacement(failure, delivery):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._app = MagicMock()
    client = AsyncMock()
    client.chat_startStream.return_value = {"ok": True, "ts": "123.456"}
    client.chat_postMessage.return_value = {"ok": True, "ts": "999.111"}
    client.chat_update.return_value = {"ok": True, "ts": "123.456"}
    client.chat_stopStream.return_value = {"ok": True}
    client.chat_delete.return_value = {"ok": True}
    adapter._get_client = MagicMock(return_value=client)
    adapter.stop_typing = AsyncMock()

    if failure == "prefix_mismatch":
        prefix, tail = "```python\nprint(", "'hello')\n```"
    else:
        prefix, tail = "**Checking", " the result.**"
        client.chat_appendStream.side_effect = RuntimeError("append failed")
    if delivery != "edit":
        client.chat_update.side_effect = RuntimeError("edit failed")
    consumer = GatewayStreamConsumer(
        adapter, "C1",
        config=StreamConsumerConfig(transport="auto", edit_interval=0, buffer_threshold=1),
        metadata={"thread_id": "111.000", "user_id": "U1"},
    )
    frame_delivered = asyncio.Event()
    original_send_draft = adapter.send_draft

    async def send_draft(*args, **kwargs):
        result = await original_send_draft(*args, **kwargs)
        frame_delivered.set()
        return result

    adapter.send_draft = send_draft
    task = asyncio.create_task(consumer.run())
    try:
        client.chat_postMessage.return_value = {"ok": True, "ts": "100.001"}
        consumer.on_commentary("Checking prerequisites.")
        assert await asyncio.to_thread(consumer.flush_pending_sync)
        client.chat_postMessage.assert_awaited_once()
        client.chat_postMessage.reset_mock()
        client.chat_postMessage.return_value = {"ok": True, "ts": "999.111"}
        if delivery == "failed_replacement":
            client.chat_postMessage.side_effect = RuntimeError("post failed")
        consumer.on_delta(prefix)
        await asyncio.wait_for(frame_delivered.wait(), timeout=5)
        client.chat_startStream.assert_awaited_once()
        frame_delivered.clear()
        consumer.on_delta(tail)
        await asyncio.wait_for(frame_delivered.wait(), timeout=5)
        # Failed edits must not erase the only visible response before a final lands.
        client.chat_delete.assert_not_awaited()
        consumer.finish(prefix + tail)
        await asyncio.wait_for(task, timeout=5)
    finally:
        if not task.done():
            task.cancel()
            await task

    if delivery == "edit":
        client.chat_postMessage.assert_not_awaited()
        assert client.chat_update.await_args.kwargs["ts"] == "123.456"
        assert client.chat_update.await_args.kwargs["text"] == adapter.format_message(prefix + tail)
        client.chat_delete.assert_not_awaited()
        assert consumer.delivered_final_matches(prefix + tail)
    elif delivery == "replacement":
        client.chat_postMessage.assert_awaited_once()
        assert client.chat_postMessage.await_args.kwargs["text"] == adapter.format_message(prefix + tail)
        client.chat_delete.assert_awaited_once_with(channel="C1", ts="123.456")
        assert consumer.delivered_final_matches(prefix + tail)
    else:
        client.chat_postMessage.assert_awaited()
        client.chat_delete.assert_not_awaited()
        assert not consumer.final_content_delivered
