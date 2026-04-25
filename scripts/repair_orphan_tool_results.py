#!/usr/bin/env python3
"""Repair orphan tool_result / malformed tool_call entries in a Hermes session file.

A Hermes session is a JSON-Lines file (one message dict per line) stored under
``~/.hermes/sessions/``.  Two kinds of corruption can leave the session in a
state that the gateway's Responses-API adapter refuses (HTTP 400
``No tool call found for function call output with call_id ...``):

1.  **Malformed tool_call** — an ``assistant`` message carries a ``tool_calls``
    entry whose ``function.name`` is empty/missing.  Some providers emit this
    when a streamed response is truncated.
2.  **Orphan tool_result** — a ``tool`` message whose ``tool_call_id`` does
    not match any surviving ``assistant`` tool_call id.  This typically
    appears *after* fix (1) strips the malformed call.

Both chat-completions format
(``role="assistant"`` + ``tool_calls`` / ``role="tool"`` + ``tool_call_id``)
and Responses-API style (``type="function_call"`` + ``call_id`` /
``type="function_call_output"`` + ``call_id``) are handled.

Usage::

    python scripts/repair_orphan_tool_results.py <session.jsonl>           # dry-run
    python scripts/repair_orphan_tool_results.py <session.jsonl> --apply   # rewrite (with .bak)

Dry-run is the default — nothing is written unless ``--apply`` is passed.
When applying, the original file is copied to ``<path>.bak.<timestamp>`` first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _extract_tool_call_name_and_id(tc: Any) -> Tuple[str, str]:
    """Return (name, id) for a chat-style tool_call dict."""
    if not isinstance(tc, dict):
        return "", ""
    fn = tc.get("function") or {}
    name = fn.get("name") if isinstance(fn, dict) else None
    cid = tc.get("id") or ""
    return (name or "").strip(), cid or ""


def _strip_malformed_tool_calls(msg: Dict[str, Any]) -> int:
    """Mutate ``msg`` in place.  Returns number of removed tool_calls."""
    tcs = msg.get("tool_calls")
    if not isinstance(tcs, list) or not tcs:
        return 0
    kept = []
    for tc in tcs:
        name, _ = _extract_tool_call_name_and_id(tc)
        if name:
            kept.append(tc)
    removed = len(tcs) - len(kept)
    if removed:
        if kept:
            msg["tool_calls"] = kept
        else:
            # Assistant message had only malformed calls: keep the message
            # itself only if it has content; otherwise drop tool_calls key
            # (caller decides whether to keep the whole message).
            msg.pop("tool_calls", None)
    return removed


def analyze(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Walk messages and collect repair decisions.

    Returns a dict with:
      - ``cleaned``     : new list of messages to keep/rewrite
      - ``malformed``   : list of (line_idx, call_id) for empty-name tool_calls
      - ``orphans``     : list of (line_idx, call_id) for orphan tool_results
      - ``dropped_msgs``: list of (line_idx, reason) for whole messages removed
    """
    malformed: List[Tuple[int, str]] = []
    orphans: List[Tuple[int, str]] = []
    dropped_msgs: List[Tuple[int, str]] = []

    # --- Pass 1: strip malformed tool_calls / function_call items ---
    pass1: List[Tuple[int, Dict[str, Any]]] = []  # (original_idx, msg)
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            dropped_msgs.append((idx, "non-dict line"))
            continue

        # Responses-API style: type=function_call with empty name
        if msg.get("type") == "function_call":
            name = (msg.get("name") or "").strip()
            if not name:
                cid = msg.get("call_id") or msg.get("id") or ""
                malformed.append((idx, cid))
                dropped_msgs.append((idx, f"function_call empty name (call_id={cid})"))
                continue

        # Chat style: assistant with tool_calls list
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            original_tcs = list(msg["tool_calls"])
            for tc in original_tcs:
                name, cid = _extract_tool_call_name_and_id(tc)
                if not name:
                    malformed.append((idx, cid))
            # Mutate a copy so we don't scribble the input list
            msg = dict(msg)
            msg["tool_calls"] = list(original_tcs)
            removed = _strip_malformed_tool_calls(msg)
            # If the assistant had *only* malformed calls and no content,
            # the message becomes useless — drop it.
            if removed and not msg.get("tool_calls"):
                has_content = bool(
                    (msg.get("content") or "").strip()
                    if isinstance(msg.get("content"), str)
                    else msg.get("content")
                )
                if not has_content:
                    dropped_msgs.append(
                        (idx, "assistant msg left empty after stripping malformed tool_calls")
                    )
                    continue

        pass1.append((idx, msg))

    # --- Pass 2: collect surviving call ids (both formats) ---
    surviving_call_ids = set()
    for _, msg in pass1:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                _, cid = _extract_tool_call_name_and_id(tc)
                if cid:
                    surviving_call_ids.add(cid)
        if msg.get("type") == "function_call":
            cid = msg.get("call_id") or msg.get("id")
            if cid:
                surviving_call_ids.add(cid)

    # --- Pass 3: drop orphan tool_results ---
    cleaned: List[Dict[str, Any]] = []
    for idx, msg in pass1:
        if msg.get("role") == "tool":
            cid = msg.get("tool_call_id") or ""
            if cid and cid not in surviving_call_ids:
                orphans.append((idx, cid))
                dropped_msgs.append((idx, f"orphan tool_result (call_id={cid})"))
                continue
        if msg.get("type") == "function_call_output":
            cid = msg.get("call_id") or ""
            if cid and cid not in surviving_call_ids:
                orphans.append((idx, cid))
                dropped_msgs.append((idx, f"orphan function_call_output (call_id={cid})"))
                continue
        cleaned.append(msg)

    return {
        "cleaned": cleaned,
        "malformed": malformed,
        "orphans": orphans,
        "dropped_msgs": dropped_msgs,
    }


def load_session(path: Path) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  warning: line {lineno} is not valid JSON ({e}); kept as raw string",
                      file=sys.stderr)
                msgs.append({"__raw__": line})
    return msgs


def write_session(path: Path, messages: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for msg in messages:
            if "__raw__" in msg:
                fh.write(msg["__raw__"] + "\n")
            else:
                fh.write(json.dumps(msg, ensure_ascii=False) + "\n")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session", type=Path, help="Path to session .jsonl file")
    parser.add_argument("--apply", action="store_true",
                        help="Actually rewrite the file (default: dry-run)")
    args = parser.parse_args(argv)

    path: Path = args.session
    if not path.is_file():
        print(f"error: {path} is not a file", file=sys.stderr)
        return 2

    messages = load_session(path)
    total = len(messages)
    report = analyze(messages)
    cleaned = report["cleaned"]
    malformed = report["malformed"]
    orphans = report["orphans"]
    dropped_msgs = report["dropped_msgs"]

    print(f"session: {path}")
    print(f"  scanned lines : {total}")
    print(f"  malformed calls (empty function.name): {len(malformed)}")
    print(f"  orphan tool_results                  : {len(orphans)}")
    print(f"  messages to drop                     : {len(dropped_msgs)}")
    print(f"  messages kept                        : {len(cleaned)}")

    if dropped_msgs:
        print("\n  would drop:")
        for idx, reason in dropped_msgs[:50]:
            print(f"    line {idx}: {reason}")
        if len(dropped_msgs) > 50:
            print(f"    ... ({len(dropped_msgs) - 50} more)")

    if not dropped_msgs and not malformed:
        print("\nnothing to repair.")
        return 0

    if not args.apply:
        print("\n[dry-run] no changes written. Re-run with --apply to rewrite.")
        return 0

    backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    shutil.copy2(path, backup)
    write_session(path, cleaned)
    print(f"\napplied. backup -> {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
