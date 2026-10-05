from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from hermes_state import SessionDB
from hermes_state_common import (
    FTS_CJK_PROJECTION_PENDING_KEY,
    FTS_PROJECTION_PENDING_KEY,
    FTS_TRIGRAM_PROJECTION_PENDING_KEY,
)
from hermes_cli.session_recovery import recover_session_database


_PENDING_PROJECTION_KEYS = {
    FTS_PROJECTION_PENDING_KEY,
    FTS_TRIGRAM_PROJECTION_PENDING_KEY,
    FTS_CJK_PROJECTION_PENDING_KEY,
}


def _make_old_source(path: Path) -> tuple[str, list[dict[str, object]]]:
    payload = [
        {"type": "text", "text": "legacy line one\nlegacy line two"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,OLD"}},
    ]
    db = SessionDB(db_path=path)
    try:
        db.create_session("legacy-session", source="cli")
        message_id = db.append_message("legacy-session", "user", payload)
        db.set_meta(FTS_PROJECTION_PENDING_KEY, "old-base-pending")
        db.set_meta(FTS_TRIGRAM_PROJECTION_PENDING_KEY, "old-trigram-pending")
        db.set_meta(FTS_CJK_PROJECTION_PENDING_KEY, "old-cjk-pending")
    finally:
        db.close()

    # SQLite's transaction context does not close the connection. Finish the
    # fixture before recovery so later GC cannot checkpoint its source WAL.
    with closing(sqlite3.connect(str(path), isolation_level=None)) as conn:
        raw_content = conn.execute(
            "SELECT content FROM messages WHERE id = ?", (message_id,)
        ).fetchone()[0]
        # Reproduce a pre-projection source: the old canonical table had no
        # fts_content column, and its derived FTS objects are not canonical.
        objects = conn.execute(
            "SELECT type, name FROM sqlite_master WHERE name LIKE 'messages_fts%'"
        ).fetchall()
        for object_type, name in objects:
            if object_type == "trigger":
                conn.execute(f'DROP TRIGGER IF EXISTS "{name}"')
        for object_type, name in objects:
            if object_type == "view":
                conn.execute(f'DROP VIEW IF EXISTS "{name}"')
        for object_type, name in objects:
            if object_type == "table":
                conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        conn.execute("ALTER TABLE messages DROP COLUMN fts_content")
    return raw_content, payload


def test_recovery_rebuilds_missing_projection_without_mutating_replay_or_pending_meta(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-state.db"
    output = tmp_path / "recovered-state.db"
    raw_content, payload = _make_old_source(source)

    report = recover_session_database(source, output, work_dir=tmp_path)

    assert report["complete"] is True
    assert report["verified"] is True
    assert report["verification"]["integrity_check"] == ["ok"]
    assert report["verification"]["foreign_key_check"] == []
    assert report["verification"]["fts_checks"]
    assert all(value == "ok" for value in report["verification"]["fts_checks"].values())

    with sqlite3.connect(str(output)) as conn:
        raw_recovered, projection = conn.execute(
            "SELECT content, fts_content FROM messages WHERE id = 1"
        ).fetchone()
        assert raw_recovered == raw_content
        assert projection == "legacy line one\nlegacy line two"
        assert conn.execute(
            "SELECT content FROM messages_fts WHERE rowid = 1"
        ).fetchone()[0] == projection
        # The strict rank=1 probe re-reads the external content source.
        conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
        )
        keys = {row[0] for row in conn.execute("SELECT key FROM state_meta")}

    assert not keys & _PENDING_PROJECTION_KEYS

    recovered = SessionDB(db_path=output)
    try:
        assert recovered.get_messages("legacy-session")[0]["content"] == payload
    finally:
        recovered.close()
