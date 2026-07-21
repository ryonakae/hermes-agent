"""Unit tests for repairing tool_calls with empty function.name.

Regression coverage for the bug where some providers emit a streamed tool_call
with ``id="call_xxx"`` but ``function.name=""``.  Such malformed calls were
previously silently dropped by the Responses-API adapter while the matching
``tool_result`` was retained, which produced gateway 400 errors of the form::

    No tool call found for function call output with call_id ...

The fix lives in ``AIAgent._sanitize_api_messages`` (run_agent.py). It renames
the malformed call to ``invalid_tool_call`` so the call and matching result
stay paired and the model can receive the anti-priming error result.
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
    """Repair tool_calls whose function.name is empty or missing."""

    def test_empty_name_call_repaired_with_result_preserved(self):
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

        assistant_msgs = [m for m in out if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        surviving_call_ids = [tc["id"] for tc in assistant_msgs[0]["tool_calls"]]
        assert surviving_call_ids == ["c_good", "c_bad"]
        names = [tc["function"]["name"] for tc in assistant_msgs[0]["tool_calls"]]
        assert names == ["terminal", "invalid_tool_call"]

        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["c_good", "c_bad"]

    def test_missing_function_field_repaired(self):
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
        assert ids_left == {"c1", "c_no_fn"}
        assistant = next(m for m in out if m.get("role") == "assistant")
        assert assistant["tool_calls"][1]["function"]["name"] == "invalid_tool_call"

    def test_whitespace_only_name_repaired(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                assistant_dict_call("c_ws", name="   "),
            ]},
            tool_result("c_ws"),
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assistant = next(m for m in out if m.get("role") == "assistant")
        assert assistant["tool_calls"][0]["function"]["name"] == "invalid_tool_call"
        assert any(m.get("tool_call_id") == "c_ws" for m in out)

    def test_object_style_tool_call_with_empty_name_repaired(self):
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
        assert tool_ids == {"c_ok", "c_obj_bad"}
        assistant = next(m for m in out if m.get("role") == "assistant")
        assert assistant["tool_calls"][1].function.name == "invalid_tool_call"

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
        assert all_call_ids == {"a1", "a2", "a_bad"}
        assert all_result_ids == {"a1", "a2", "a_bad"}
        repaired = next(
            tc
            for m in out if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
            if tc["id"] == "a_bad"
        )
        assert repaired["function"]["name"] == "invalid_tool_call"


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
