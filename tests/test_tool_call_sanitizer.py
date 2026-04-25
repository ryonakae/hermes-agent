"""Unit tests for the Pass-0 sanitizer that drops tool_calls with empty function.name.

Regression coverage for the bug where some providers emit a streamed tool_call
with ``id="call_xxx"`` but ``function.name=""``.  Such malformed calls were
previously silently dropped by the Responses-API adapter while the matching
``tool_result`` was retained, which produced gateway 400 errors of the form::

    No tool call found for function call output with call_id ...

The fix lives in ``AIAgent._sanitize_api_messages`` (run_agent.py) — Pass 0
strips the malformed call so the existing orphan-result logic then removes
its (now unpaired) ``tool`` message.
"""

from __future__ import annotations

import types

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assistant_dict_call(call_id: str, name: str = "terminal", arguments: str = "{}") -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def assistant_obj_call(call_id: str, name: str = "terminal", arguments: str = "{}"):
    """SDK-style object (SimpleNamespace) tool_call."""
    tc = types.SimpleNamespace()
    tc.id = call_id
    tc.function = types.SimpleNamespace(name=name, arguments=arguments)
    return tc


def tool_result(call_id: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# ---------------------------------------------------------------------------
# Direct sanitizer tests
# ---------------------------------------------------------------------------

class TestEmptyFunctionNameSanitizer:
    """Phase 0 — strip tool_calls whose function.name is empty/missing."""

    def test_empty_name_call_dropped_with_orphan_result(self):
        msgs = [
            {"role": "user", "content": "do stuff"},
            {"role": "assistant", "tool_calls": [
                assistant_dict_call("c_good", name="terminal"),
                assistant_dict_call("c_bad", name=""),
            ]},
            tool_result("c_good", "first"),
            tool_result("c_bad", "second"),
        ]
        out = AIAgent._sanitize_api_messages(msgs)

        # Good call survives, bad call is gone, its tool_result is gone too.
        assistant_msgs = [m for m in out if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        surviving_call_ids = [tc["id"] for tc in assistant_msgs[0]["tool_calls"]]
        assert surviving_call_ids == ["c_good"]

        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "c_good"

    def test_missing_function_field_dropped(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                assistant_dict_call("c1", name="read_file"),
                {"id": "c_no_fn"},  # no function key at all
            ]},
            tool_result("c1"),
            tool_result("c_no_fn"),
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        ids_left = {m.get("tool_call_id") for m in out if m.get("role") == "tool"}
        assert ids_left == {"c1"}

    def test_whitespace_only_name_dropped(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                assistant_dict_call("c_ws", name="   "),
            ]},
            tool_result("c_ws"),
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        # Stub will be inserted for orphaned call — but here the call itself
        # is removed as malformed, so no stub and no tool result.
        assert all(m.get("tool_call_id") != "c_ws" for m in out)
        assert all(
            not (m.get("role") == "assistant" and any(
                (tc.get("function") or {}).get("name", "").strip() == ""
                for tc in (m.get("tool_calls") or [])
            ))
            for m in out
        )

    def test_object_style_tool_call_with_empty_name_dropped(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                assistant_obj_call("c_ok", name="search_files"),
                assistant_obj_call("c_obj_bad", name=""),
            ]},
            tool_result("c_ok"),
            tool_result("c_obj_bad"),
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        tool_ids = {m.get("tool_call_id") for m in out if m.get("role") == "tool"}
        assert tool_ids == {"c_ok"}

    def test_all_normal_calls_unchanged_regression(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [
                assistant_dict_call("c1", name="terminal"),
                assistant_dict_call("c2", name="read_file"),
            ]},
            tool_result("c1", "first"),
            tool_result("c2", "second"),
            {"role": "assistant", "content": "all done"},
        ]
        # Snapshot deep state before
        before_ids = [
            tc["id"]
            for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")
            for tc in m["tool_calls"]
        ]
        before_results = [m["tool_call_id"] for m in msgs if m.get("role") == "tool"]

        out = AIAgent._sanitize_api_messages(msgs)

        after_ids = [
            tc["id"]
            for m in out if m.get("role") == "assistant" and m.get("tool_calls")
            for tc in m["tool_calls"]
        ]
        after_results = [m["tool_call_id"] for m in out if m.get("role") == "tool"]

        assert after_ids == before_ids
        assert after_results == before_results
        assert len(out) == len(msgs)

    def test_multiple_assistant_messages_independent(self):
        msgs = [
            {"role": "assistant", "tool_calls": [assistant_dict_call("a1")]},
            tool_result("a1"),
            {"role": "assistant", "tool_calls": [
                assistant_dict_call("a2"),
                assistant_dict_call("a_bad", name=""),
            ]},
            tool_result("a2"),
            tool_result("a_bad"),
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        all_call_ids = {
            tc["id"]
            for m in out if m.get("role") == "assistant" and m.get("tool_calls")
            for tc in m["tool_calls"]
        }
        all_result_ids = {m["tool_call_id"] for m in out if m.get("role") == "tool"}
        assert "a_bad" not in all_call_ids
        assert "a_bad" not in all_result_ids
        assert {"a1", "a2"}.issubset(all_call_ids)


# ---------------------------------------------------------------------------
# Idempotency — running sanitizer twice must produce identical output
# ---------------------------------------------------------------------------

def test_sanitizer_is_idempotent_on_corrupted_input():
    msgs = [
        {"role": "assistant", "tool_calls": [
            assistant_dict_call("c_keep"),
            assistant_dict_call("c_drop", name=""),
        ]},
        tool_result("c_keep"),
        tool_result("c_drop"),
    ]
    once = AIAgent._sanitize_api_messages(msgs)
    twice = AIAgent._sanitize_api_messages([dict(m) for m in once])
    assert once == twice


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
